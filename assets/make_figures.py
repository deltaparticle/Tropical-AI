"""
make_figures.py

Generates the README figures from confirmed, already-run experimental
numbers (not placeholders). Run once from this folder: python make_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt

COLORS = {"Euclidean": "#888888", "Poincare": "#4477aa", "Tropical": "#cc6677"}

# -----------------------------------------------------------------------
# 1. Why hyperbolic/tropical: volume growth vs. tree branching
# -----------------------------------------------------------------------
r = np.linspace(0.1, 6, 200)
branching_factor = 2.5
tree_nodes = branching_factor ** r            # exponential, matches a branching tree
euclid_vol_d10 = r ** 10                      # polynomial (Euclidean ball, d=10)
hyper_vol = np.exp((10 - 1) * r) * 0.001      # exponential (hyperbolic ball, d=10), scaled to fit

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(r, np.log(tree_nodes), label="Tree node count (branching factor 2.5)", color="black", linewidth=2)
ax.plot(r, np.log(euclid_vol_d10), label="Euclidean ball volume (dim 10)", color=COLORS["Euclidean"], linewidth=2)
ax.plot(r, np.log(hyper_vol), label="Hyperbolic ball volume (dim 10)", color=COLORS["Poincare"], linewidth=2, linestyle="--")
ax.set_xlabel("Radius r")
ax.set_ylabel("log(count or volume)")
ax.set_title("Why Euclidean space is the wrong shape for a tree")
ax.legend(fontsize=9)
ax.text(0.98, 0.04,
        "Hyperbolic volume grows at the same exponential rate as a branching tree.\n"
        "Euclidean volume only grows polynomially -- it falls further behind as r grows.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, style="italic", color="#444")
plt.tight_layout()
plt.savefig("growth_rates.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 2. Wikipedia AI graph: held-out mAP vs. embedding dimension (real numbers,
#    embeddings/wiki_kg_train_eval.py dimension sweep)
# -----------------------------------------------------------------------
dims = [20, 60, 100]
mAP = {
    "Euclidean": [0.0838, 0.1071, 0.1071],
    "Poincare":  [0.1035, 0.1255, 0.1188],
    "Tropical":  [0.1816, 0.2018, 0.2606],
}

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, vals in mAP.items():
    ax.plot(dims, vals, "o-", label=name, color=COLORS[name], linewidth=2, markersize=7)
ax.set_xlabel("Embedding dimension")
ax.set_ylabel("Held-out mAP (higher is better)")
ax.set_title("Wikipedia AI category graph: held-out link prediction")
ax.set_xticks(dims)
ax.legend()
plt.tight_layout()
plt.savefig("wiki_dim_sweep.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 3. RAGAS final results: 4 metrics x 3 geometries (confirmed final numbers,
#    evaluation/ragas_results.json)
# -----------------------------------------------------------------------
metrics = ["Faithfulness", "Answer\nRelevancy", "Answer\nCorrectness", "Answer\nSimilarity"]
ragas = {
    "Euclidean": [0.918, 0.514, 0.328, 0.756],
    "Poincare":  [0.941, 0.354, 0.268, 0.753],
    "Tropical":  [0.862, 0.612, 0.361, 0.782],
}

x = np.arange(len(metrics))
width = 0.26
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (name, vals) in enumerate(ragas.items()):
    ax.bar(x + (i - 1) * width, vals, width, label=name, color=COLORS[name])
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score (mean over 15 questions)")
ax.set_title("RAGAS evaluation of the knowledge-graph RAG pipeline")
ax.legend()
plt.tight_layout()
plt.savefig("ragas_results.png", dpi=150)
plt.close()

print("Saved growth_rates.png, wiki_dim_sweep.png, ragas_results.png")
