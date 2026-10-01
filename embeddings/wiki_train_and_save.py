"""
wiki_train_and_save.py

Trains Euclidean / Poincare / Tropical embeddings on the FULL Wikipedia AI
category graph (no held-out split this time -- for the demo we want the
best possible embedding of the graph we actually have, not a generalization
test) and saves each to a .pt file for the query demo to load without
retraining.

Dimension: 100 -- chosen from the independent held-out link-prediction
dimension sweep (wiki_kg_train_eval.py), where tropical's held-out mAP
(0.261) clearly beats both Euclidean (0.107) and Poincare (0.119) at this
dimension -- the clearest margin of the three tested dims (20/60/100). This
choice is made from that held-out task alone, before looking at any RAG/
RAGAS output, to avoid picking a dimension that happens to flatter one
geometry on the downstream eval.
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wiki_kg_train_eval import (
    load_graph, build_relation_structures,
    euclidean_dist, poincare_dist, tropical_dist,
    make_temperature_schedule, Embedding, DATA_DIR,
)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

DIM = 100
EPOCHS = 400


def sample_negatives(N, u_idx, v_idx, num_neg):
    negs = []
    while len(negs) < num_neg:
        cand = random.randrange(N)
        if cand != u_idx and cand != v_idx:
            negs.append(cand)
    return negs


def train_full_graph(N, all_pairs_directed, dist_fn, dim, epochs, lr,
                      riemannian=False, ball_eps=1e-3, burn_in_epochs=20, burn_in_factor=10,
                      batch_size=256, num_neg=10, temperature_schedule=None, verbose=True):
    model = Embedding(N, dim)
    opt = None if riemannian else torch.optim.Adam(model.parameters(), lr=lr)
    pairs = all_pairs_directed[:]
    for epoch in range(epochs):
        cur_lr = lr / burn_in_factor if epoch < burn_in_epochs else lr
        random.shuffle(pairs)
        total_loss = 0.0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            u_idx = torch.tensor([p[0] for p in batch])
            v_idx = torch.tensor([p[1] for p in batch])
            neg_idx = torch.tensor([sample_negatives(N, p[0], p[1], num_neg) for p in batch])

            u = model(u_idx)
            v_pos = model(v_idx)
            v_negs = model(neg_idx)

            if temperature_schedule is not None:
                t = temperature_schedule(epoch)
                from wiki_kg_train_eval import soft_tropical_dist
                d_pos = soft_tropical_dist(u, v_pos, t)
                d_negs = soft_tropical_dist(u.unsqueeze(1).expand_as(v_negs), v_negs, t)
            else:
                d_pos = dist_fn(u, v_pos)
                d_negs = dist_fn(u.unsqueeze(1).expand_as(v_negs), v_negs)

            logits = -torch.cat([d_pos.unsqueeze(1), d_negs], dim=1)
            labels = torch.zeros(len(batch), dtype=torch.long)
            loss = F.cross_entropy(logits, labels)

            model.zero_grad() if riemannian else opt.zero_grad()
            loss.backward()

            if riemannian:
                with torch.no_grad():
                    sq_norm = model.emb.pow(2).sum(-1, keepdim=True).clamp(max=1 - ball_eps)
                    riemannian_scale = (1 - sq_norm).pow(2) / 4
                    model.emb -= cur_lr * riemannian_scale * model.emb.grad
                    norms = model.emb.norm(dim=-1, keepdim=True)
                    too_big = (norms > 1 - ball_eps).squeeze(-1)
                    model.emb[too_big] = model.emb[too_big] / norms[too_big] * (1 - ball_eps)
            else:
                opt.step()

            total_loss += loss.item() * len(batch)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}  loss {total_loss / len(pairs):.4f}")
    return model


if __name__ == "__main__":
    nodes, node_idx, N, edges = load_graph()
    related = build_relation_structures(N, edges)
    all_pairs_undirected = list({frozenset((i, j)) for i in related for j in related[i]})
    all_pairs_directed = [tuple(p) for p in all_pairs_undirected] + \
                          [(b, a) for a, b in all_pairs_undirected]
    print(f"Training on full graph: {N} nodes, {len(all_pairs_directed)} directed relation pairs")

    sched = make_temperature_schedule(t_start=1.0, t_end=0.02, anneal_epochs=250)
    specs = [
        ("euclidean", euclidean_dist, dict(lr=1e-2, riemannian=False)),
        ("poincare", poincare_dist, dict(lr=5e-1, riemannian=True)),
        ("tropical", tropical_dist, dict(lr=1e-2, riemannian=False, temperature_schedule=sched)),
    ]

    for name, dist_fn, kwargs in specs:
        print(f"\nTraining {name} (dim={DIM})...")
        model = train_full_graph(N, all_pairs_directed, dist_fn, DIM, EPOCHS, **kwargs)
        torch.save({"emb": model.emb.detach(), "nodes": nodes}, os.path.join(DATA_DIR, f"wiki_emb_{name}.pt"))
        print(f"Saved wiki_emb_{name}.pt")

    print("\nAll three embeddings trained and saved.")
