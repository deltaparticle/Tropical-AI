"""
tropical_geneontology_check.py

Validates the tropical-vs-Poincare-vs-Euclidean held-out link-prediction
result on a real, currently-used-in-AI ontology: the Gene Ontology (GO),
specifically the "immune system process" (GO:0002376) is_a subtree.

Why this dataset, not another WordNet subtree: GO terms are the actual
labels/targets used to train and evaluate real, current protein-function-
prediction deep learning models (e.g. DeepGO-style models, ESM/AlphaFold-
adjacent function-prediction heads, antibody and vaccine-design AI) --
GO-term embeddings that respect the ontology's hierarchy are a real,
practically used technique in that field, not a toy benchmark. This is a
genuinely different real-world hierarchy from WordNet: different domain
(molecular biology/immunology vs. lexical semantics), different curators,
different branching statistics.

Same methodology as the WordNet checks: real is_a edges, transitive
closure, held-out link prediction (train on 90% of relation pairs, predict
the held-out 10%), annealed tropical training, matched training budgets.

Requires: go-basic.obo (downloaded from purl.obolibrary.org/obo/go/go-basic.obo)
and the `goatools` package.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from goatools.obo_parser import GODag

OBO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go-basic.obo")
ROOT_GO_ID = "GO:0002376"  # immune system process
TEST_FRAC = 0.1
DIMS = [20, 60, 100]
EPOCHS = 500
PATIENCE = 150
CHECK_EVERY = 10

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# 1. Data -- real Gene Ontology is_a subtree
# ---------------------------------------------------------------------------

godag = GODag(OBO_PATH)
root = godag[ROOT_GO_ID]

nodes = set()
edges = []  # (parent, child) is_a edges
stack = [root]
while stack:
    s = stack.pop()
    if s.item_id in nodes:
        continue
    nodes.add(s.item_id)
    for child in s.children:
        edges.append((s.item_id, child.item_id))
        stack.append(child)

nodes = sorted(nodes)
node_idx = {go_id: i for i, go_id in enumerate(nodes)}
N = len(nodes)
print(f"Gene Ontology '{ROOT_GO_ID}' ({root.name}) subtree: {N} nodes, {len(edges)} direct is_a edges")

children = {i: set() for i in range(N)}
for p, c in edges:
    children[node_idx[p]].add(node_idx[c])

all_descendants = {i: set() for i in range(N)}
for i in range(N):
    stack = list(children[i])
    seen = set()
    while stack:
        d = stack.pop()
        if d in seen:
            continue
        seen.add(d)
        stack.extend(children[d])
    all_descendants[i] = seen

related = {i: set() for i in range(N)}
for i in range(N):
    for d in all_descendants[i]:
        related[i].add(d)
        related[d].add(i)

undirected_pairs = list({frozenset((i, j)) for i in related for j in related[i]})
undirected_pairs = [tuple(p) for p in undirected_pairs]
random.shuffle(undirected_pairs)

n_test = int(len(undirected_pairs) * TEST_FRAC)
test_pairs = undirected_pairs[:n_test]
train_pairs_undirected = undirected_pairs[n_test:]

print(f"Total relation pairs: {len(undirected_pairs)}  "
      f"train: {len(train_pairs_undirected)}  held-out test: {len(test_pairs)}")

test_related = {i: set() for i in range(N)}
for a, b in test_pairs:
    test_related[a].add(b)
    test_related[b].add(a)

train_pairs_directed = list(train_pairs_undirected) + [(b, a) for a, b in train_pairs_undirected]


# ---------------------------------------------------------------------------
# 2. Distances, model, training, evaluation (identical formulas to the
#    validated WordNet scripts)
# ---------------------------------------------------------------------------

def euclidean_dist(u, v):
    return (u - v).pow(2).sum(-1).clamp_min(1e-12).sqrt()


def poincare_dist(u, v, eps=1e-5):
    uu = u.pow(2).sum(-1).clamp(max=1 - eps)
    vv = v.pow(2).sum(-1).clamp(max=1 - eps)
    sq_dist = (u - v).pow(2).sum(-1)
    x = 1 + 2 * sq_dist / ((1 - uu) * (1 - vv))
    x = x.clamp_min(1 + eps)
    return torch.acosh(x)


def tropical_dist(u, v):
    diff = u - v
    return diff.max(dim=-1).values - diff.min(dim=-1).values


def soft_tropical_dist(u, v, temperature):
    diff = u - v
    smax = temperature * torch.logsumexp(diff / temperature, dim=-1)
    smin = -temperature * torch.logsumexp(-diff / temperature, dim=-1)
    return smax - smin


def make_temperature_schedule(t_start=1.0, t_end=0.02, anneal_epochs=300):
    def schedule(epoch):
        if epoch >= anneal_epochs:
            return t_end
        frac = epoch / anneal_epochs
        return t_start * (t_end / t_start) ** frac
    return schedule


class Embedding(nn.Module):
    def __init__(self, n, dim, init_scale=1e-3):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(n, dim).uniform_(-init_scale, init_scale))

    def forward(self, idx):
        return self.emb[idx]


def sample_negatives(u_idx, v_idx, num_neg):
    negs = []
    while len(negs) < num_neg:
        cand = random.randrange(N)
        if cand != u_idx and cand != v_idx:
            negs.append(cand)
    return negs


@torch.no_grad()
def evaluate_heldout(model, dist_fn):
    emb_all = model.emb
    ranks, aps = [], []
    for i in range(N):
        rel = test_related[i]
        if not rel:
            continue
        u = emb_all[i].unsqueeze(0).expand(N, -1)
        d = dist_fn(u, emb_all)
        d[i] = float("inf")
        order = torch.argsort(d)
        rank_of = {n.item(): r for r, n in enumerate(order)}
        node_ranks = sorted(rank_of[j] for j in rel)
        ranks.extend([r + 1 for r in node_ranks])
        hits, precisions = 0, []
        for r in node_ranks:
            hits += 1
            precisions.append(hits / (r + 1))
        aps.append(sum(precisions) / len(precisions))
    return float(np.mean(ranks)), float(np.mean(aps))


def train_link_pred(dist_fn, dim, epochs, lr, num_neg=10, batch_size=256,
                     riemannian=False, ball_eps=1e-3, burn_in_epochs=20, burn_in_factor=10,
                     patience=150, check_every=10, verbose=True, temperature_schedule=None):
    model = Embedding(N, dim)
    opt = None if riemannian else torch.optim.Adam(model.parameters(), lr=lr)
    pairs = train_pairs_directed[:]
    best_mAP = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_since_improve = 0
    for epoch in range(epochs):
        cur_lr = lr / burn_in_factor if epoch < burn_in_epochs else lr
        random.shuffle(pairs)
        total_loss = 0.0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            u_idx = torch.tensor([p[0] for p in batch])
            v_idx = torch.tensor([p[1] for p in batch])
            neg_idx = torch.tensor([sample_negatives(p[0], p[1], num_neg) for p in batch])

            u = model(u_idx)
            v_pos = model(v_idx)
            v_negs = model(neg_idx)

            if temperature_schedule is not None:
                t = temperature_schedule(epoch)
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

        if epoch % check_every == 0 or epoch == epochs - 1:
            _, mAP = evaluate_heldout(model, dist_fn)
            if verbose:
                print(f"    epoch {epoch:4d}  loss {total_loss / len(pairs):.4f}  heldout_mAP {mAP:.4f}")
            if mAP > best_mAP:
                best_mAP = mAP
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += check_every
                if epochs_since_improve >= patience:
                    if verbose:
                        print(f"    early stopping at epoch {epoch} (best held-out mAP {best_mAP:.4f})")
                    break
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sched = make_temperature_schedule(t_start=1.0, t_end=0.02, anneal_epochs=300)
    MODEL_SPECS = [
        ("Euclidean", euclidean_dist, dict(lr=1e-2, riemannian=False)),
        ("Poincare", poincare_dist, dict(lr=5e-1, riemannian=True)),
        ("Tropical", tropical_dist, dict(lr=1e-2, riemannian=False, temperature_schedule=sched)),
    ]

    results = {}
    for dim in DIMS:
        print(f"\n=== GO immune system process, dimension {dim} (held-out link prediction) ===")
        for name, dist_fn, kwargs in MODEL_SPECS:
            print(f"\nTraining {name} (dim={dim}) on train-only edges...")
            model = train_link_pred(dist_fn, dim, epochs=EPOCHS, patience=PATIENCE,
                                     check_every=CHECK_EVERY, verbose=True, **kwargs)
            mr, mAP = evaluate_heldout(model, dist_fn)
            results[(name, dim)] = (mr, mAP)
            print(f"  {name:>10} (dim={dim}): heldout mean_rank={mr:.2f}  heldout mAP={mAP:.4f}")

    print("\n=== Summary: GO immune system process held-out link prediction vs. dimension ===")
    print(f"{'dim':>5} | {'model':>10} | {'mean_rank':>10} | {'mAP':>7}")
    for dim in DIMS:
        for name, _, _ in MODEL_SPECS:
            mr, mAP = results[(name, dim)]
            print(f"{dim:>5} | {name:>10} | {mr:>10.2f} | {mAP:>7.4f}")
