"""
tropical_hierarchy_link_prediction.py

Held-out link-prediction test for the WordNet mammal hierarchy embeddings.

The reconstruction experiment (tropical_hierarchy_embeddings.py) trains and
evaluates on the SAME full transitive closure -- by design, matching Nickel
& Kiela's "reconstruction" protocol: it measures how well a geometry can
compress a KNOWN, FIXED graph into d dimensions, not whether it generalizes.
That experiment showed near-perfect reconstruction (mAP -> 1.0) for BOTH
tropical and, increasingly, Euclidean at high dimension (dim=40) -- which is
exactly what you'd expect from any sufficiently flexible geometry once its
raw parameter count is large relative to the graph size (1170 nodes), not
uniquely convincing evidence that tropical geometry is special.

This script runs the harder, complementary test: hold out a random 10% of
the tree's ancestor/descendant relation pairs, train ONLY on the rest, and
check whether the embedding correctly ranks the HELD-OUT (never seen during
training) relations close together. If a model does well here, it's doing
real geometric generalization, not just satisfying the training objective
via brute-force parameter capacity.

Reuses the WordNet data loading, model, and distance functions from
tropical_hierarchy_embeddings.py (importing it only runs its data-loading
and function-definition code, not its __main__ sweep).

Produces:
  - hierarchy_link_prediction.png -- held-out mean rank / mAP vs. dimension
  - printed results table
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import tropical_hierarchy_embeddings as base

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

N = base.N
related = base.related
Embedding = base.Embedding
sample_negatives = base.sample_negatives
euclidean_dist = base.euclidean_dist
poincare_dist = base.poincare_dist
tropical_dist = base.tropical_dist
soft_tropical_dist = base.soft_tropical_dist
make_temperature_schedule = base.make_temperature_schedule

# ---------------------------------------------------------------------------
# 1. Split the tree's relation pairs into train / held-out test
# ---------------------------------------------------------------------------

undirected_pairs = list({frozenset((i, j)) for i in related for j in related[i]})
undirected_pairs = [tuple(p) for p in undirected_pairs]
random.shuffle(undirected_pairs)

TEST_FRAC = 0.1
n_test = int(len(undirected_pairs) * TEST_FRAC)
test_pairs = undirected_pairs[:n_test]
train_pairs_undirected = undirected_pairs[n_test:]

print(f"Total relation pairs: {len(undirected_pairs)}  "
      f"train: {len(train_pairs_undirected)}  held-out test: {len(test_pairs)}")

train_related = {i: set() for i in range(N)}
for a, b in train_pairs_undirected:
    train_related[a].add(b)
    train_related[b].add(a)

test_related = {i: set() for i in range(N)}
for a, b in test_pairs:
    test_related[a].add(b)
    test_related[b].add(a)

train_pairs_directed = list(train_pairs_undirected) + [(b, a) for a, b in train_pairs_undirected]


# ---------------------------------------------------------------------------
# 2. Training loop -- identical mechanics to base.train, but trains on
#    train-only edges and monitors/early-stops on HELD-OUT mAP
# ---------------------------------------------------------------------------

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
                     patience=150, check_every=40, verbose=True, temperature_schedule=None):
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
            loss = torch.nn.functional.cross_entropy(logits, labels)

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
    DIMS = [5, 20, 40]
    MAX_EPOCHS = 1000
    PATIENCE = 150

    MODEL_SPECS = [
        ("Euclidean", euclidean_dist, dict(lr=1e-2, riemannian=False)),
        ("Poincare", poincare_dist, dict(lr=5e-1, riemannian=True)),
        ("Tropical", tropical_dist, dict(
            lr=1e-2, riemannian=False,
            temperature_schedule=make_temperature_schedule(t_start=1.0, t_end=0.02, anneal_epochs=300),
        )),
    ]

    results = {}
    for dim in DIMS:
        print(f"\n=== dimension {dim} (held-out link prediction) ===")
        for name, dist_fn, kwargs in MODEL_SPECS:
            print(f"\nTraining {name} (dim={dim}) on train-only edges...")
            model = train_link_pred(dist_fn, dim, epochs=MAX_EPOCHS, patience=PATIENCE, **kwargs)
            mr, mAP = evaluate_heldout(model, dist_fn)
            results[(name, dim)] = (mr, mAP)
            print(f"  {name:>10} (dim={dim}): heldout mean_rank={mr:.2f}  heldout mAP={mAP:.4f}")

    print("\n=== Summary: HELD-OUT link-prediction quality vs. dimension ===")
    print(f"{'dim':>5} | {'model':>10} | {'mean_rank':>10} | {'mAP':>7}")
    for dim in DIMS:
        for name, _, _ in MODEL_SPECS:
            mr, mAP = results[(name, dim)]
            print(f"{dim:>5} | {name:>10} | {mr:>10.2f} | {mAP:>7.4f}")

    colors = {"Euclidean": "#888888", "Poincare": "#4477aa", "Tropical": "#cc6677"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, _, _ in MODEL_SPECS:
        ranks = [results[(name, d)][0] for d in DIMS]
        maps = [results[(name, d)][1] for d in DIMS]
        axes[0].plot(DIMS, ranks, "o-", label=name, color=colors[name])
        axes[1].plot(DIMS, maps, "o-", label=name, color=colors[name])
    axes[0].set_xlabel("Embedding dimension")
    axes[0].set_ylabel("Held-out mean rank (lower is better)")
    axes[0].set_title("Held-out link prediction: mean rank")
    axes[0].legend()
    axes[1].set_xlabel("Embedding dimension")
    axes[1].set_ylabel("Held-out mAP (higher is better)")
    axes[1].set_title("Held-out link prediction: mAP")
    axes[1].legend()
    plt.suptitle(f"WordNet mammal subtree HELD-OUT link prediction "
                 f"({N} real nodes, {len(test_pairs)} held-out relations)")
    plt.tight_layout()
    plt.savefig("plots/hierarchy_link_prediction.png", dpi=150)
    print("\nSaved hierarchy_link_prediction.png")
