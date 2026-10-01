"""
tropical_hierarchy_link_prediction_finegrained.py

Verifies the held-out link-prediction results at dim=40 with a finer
evaluation grid (every 10 epochs instead of 40, matching the dim=20
verification that confirmed tropical's win there wasn't a coarse-grid
checkpoint artifact), and extends the dimension sweep up to 100.

Early stopping is effectively disabled here (huge patience) so we get the
FULL peak-then-decline curve at every dimension, not just the model's
final answer -- this is what lets us tell a genuine, reproducible peak
apart from checkpoint noise.

Reuses tropical_hierarchy_link_prediction.py's data split, models, and
training loop.
"""

import numpy as np
import matplotlib.pyplot as plt

import tropical_hierarchy_link_prediction as lp

DIMS = [128, 256]
EPOCHS = 300
CHECK_EVERY = 10

sched = lp.make_temperature_schedule(t_start=1.0, t_end=0.02, anneal_epochs=300)
MODEL_SPECS = [
    ("Euclidean", lp.euclidean_dist, dict(lr=1e-2, riemannian=False)),
    ("Poincare", lp.poincare_dist, dict(lr=5e-1, riemannian=True)),
    ("Tropical", lp.tropical_dist, dict(lr=1e-2, riemannian=False, temperature_schedule=sched)),
]

if __name__ == "__main__":
    results = {}  # (name, dim) -> (mean_rank, mAP) at best checkpoint
    for dim in DIMS:
        print(f"\n=== dimension {dim} (fine-grained, check_every={CHECK_EVERY}) ===")
        for name, dist_fn, kwargs in MODEL_SPECS:
            print(f"\nTraining {name} (dim={dim})...")
            model = lp.train_link_pred(
                dist_fn, dim=dim, epochs=EPOCHS, patience=100000,
                check_every=CHECK_EVERY, verbose=True, **kwargs,
            )
            mr, mAP = lp.evaluate_heldout(model, dist_fn)
            results[(name, dim)] = (mr, mAP)
            print(f"  {name:>10} (dim={dim}) BEST CHECKPOINT: mean_rank={mr:.2f}  mAP={mAP:.4f}")

    print("\n=== Summary: fine-grained held-out link prediction, dims 40-100 ===")
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
    axes[0].set_title("Held-out link prediction (fine-grained): mean rank")
    axes[0].legend()
    axes[1].set_xlabel("Embedding dimension")
    axes[1].set_ylabel("Held-out mAP (higher is better)")
    axes[1].set_title("Held-out link prediction (fine-grained): mAP")
    axes[1].legend()
    plt.suptitle(f"WordNet mammal subtree HELD-OUT link prediction, dims 128-256 "
                 f"({lp.N} real nodes, {len(lp.test_pairs)} held-out relations)")
    plt.tight_layout()
    plt.savefig("plots/hierarchy_link_prediction_dims128to256.png", dpi=150)
    print("\nSaved hierarchy_link_prediction_dims128to256.png")
