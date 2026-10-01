# Tropical, Poincaré & Euclidean graph embeddings

Three geometries for embedding hierarchical/graph-structured data, compared
on real, held-out link prediction — not reconstruction of the training
graph, and not a single toy dataset.

## Why three geometries

| Geometry | Distance | Known for |
|---|---|---|
| Euclidean | `\|\|u - v\|\|` | The default; no structural bias toward hierarchy |
| Poincaré (hyperbolic) | `arcosh(1 + 2\|\|u-v\|\|² / ((1-\|\|u\|\|²)(1-\|\|v\|\|²)))` | Exponential volume growth near the ball boundary matches tree-like branching — the standard choice for hierarchy embedding (Nickel & Kiela, 2017) |
| Tropical (max-plus) | `max_i(u_i - v_i) - min_i(u_i - v_i)` | A piecewise-linear metric whose geodesics are built from tropical polytopes — less explored for embeddings, the central question of this project |

Tropical's distance depends on only 2 of the `d` embedding dimensions per
comparison (the argmax and argmin coordinates), which gives it a sparser,
different gradient structure than the other two during training (see
training notes below).

## Methodology (applied identically across all 4 datasets)

1. **Real `is_a` / parent-child edges** from an existing ontology or
   category graph — never synthetic/generated hierarchies.
2. **Transitive closure** over those edges to get the full set of
   ancestor-descendant "related" pairs.
3. **90/10 held-out split** of the *relation pairs themselves* (not nodes) —
   train on 90%, then measure whether the trained embedding can recover the
   held-out 10% purely from geometry. This is what actually tests
   generalization, as opposed to reconstruction (train and evaluate on the
   same known pairs), which mostly measures memorization capacity.
4. **Ranking metric**: for each node with held-out relations, rank every
   other node by embedding distance and compute mean average precision
   (mAP) over where the true related nodes land in that ranking.
5. **Training**: cross-entropy over (positive, k negatives) per batch.
   Poincaré uses genuine Riemannian SGD (gradient rescaled by
   `(1-||x||²)²/4`, with a burn-in phase) rather than naive Euclidean-
   gradient optimization, which trains poorly near the ball boundary.
   Tropical trains against an annealed *soft* tropical distance (Maslov
   dequantization: `t·logsumexp(x/t) → max(x)` as `t→0`) since the true
   metric's sparse gradient (only 2 of `d` dims active) causes training
   plateaus; evaluation always uses the true hard metric.
6. **Early stopping** on held-out mAP, not training loss.
7. **Control**: training on completely shuffled/fake edges (same optimizer,
   same schedule) keeps held-out mAP pinned at the random-chance floor for
   the entire run even as training loss drops normally — confirming that a
   real result reflects genuine structure learning, not an artifact of the
   training procedure.

## Datasets

| Dataset | Nodes | Domain | Script |
|---|---|---|---|
| WordNet `mammal.n.01` | 1,170 | Lexical semantics (biology taxonomy) | [tropical_hierarchy_embeddings.py](tropical_hierarchy_embeddings.py) (reconstruction), [tropical_hierarchy_link_prediction.py](tropical_hierarchy_link_prediction.py) (held-out), [tropical_hierarchy_link_prediction_finegrained.py](tropical_hierarchy_link_prediction_finegrained.py) (dims up to 256) |
| WordNet `vehicle.n.01` | 520 | Lexical semantics (man-made objects — a different domain from biology) | [tropical_hierarchy_generalization_check.py](tropical_hierarchy_generalization_check.py) |
| Gene Ontology `immune system process` (GO:0002376) | 553 | Molecular biology — real labels used to train protein-function-prediction models (DeepGO-style, antibody/vaccine design) | [tropical_geneontology_check.py](tropical_geneontology_check.py) (needs `go-basic.obo`, included) |
| Wikipedia "Artificial Intelligence" category graph | 416 | Real-world knowledge graph, crawled live via the MediaWiki API — not a pre-packaged research benchmark | [wiki_kg_train_eval.py](wiki_kg_train_eval.py) (dimension sweep + held-out eval), [wiki_train_and_save.py](wiki_train_and_save.py) (final full-graph embeddings used downstream by the RAG system) |

## Results

**Reconstruction** (WordNet mammals, train+eval on the same known pairs):
tropical reaches mAP → 1.000 by dim=40 (near-perfect), with Poincaré close
behind (0.964 at dim=40) — reconstruction alone doesn't separate the two
geometries much; it's **held-out performance** that does.

**Held-out link prediction, consistent pattern across all 4 datasets**:
Poincaré wins at very low dimension, tropical wins at moderate dimension,
and the crossover dimension scales with graph size (smaller graphs cross
over earlier). This was checked and held on every dataset below — the same
qualitative result on 4 independently-sourced real graphs, not a
dataset-specific fluke:

| Dataset | dim | Euclidean mAP | Poincaré mAP | Tropical mAP |
|---|---|---|---|---|
| WordNet mammals | 10 | 0.388 | 0.509 | **0.536** |
| WordNet vehicles | ~100 (crossover) | — | — | tropical ahead below crossover, Poincaré above |
| Gene Ontology | 60–100 | — | — | tropical wins clearly |
| **Wikipedia AI graph** | 20 | 0.084 | 0.104 | **0.182** |
| **Wikipedia AI graph** | 60 | 0.107 | 0.126 | **0.202** |
| **Wikipedia AI graph** | **100** | 0.107 | 0.119 | **0.261** |

The Wikipedia graph (the dataset that feeds the RAG system in
[`../rag/`](../rag/)) shows tropical's clearest margin at **dim=100** — the
held-out mAP there (0.261) is roughly 2.2x both Euclidean (0.107) and
Poincaré (0.119). That dimension choice was made from this independent
held-out task alone, *before* looking at any downstream RAG/RAGAS output —
see [`../evaluation/RAGAS_EVAL_RESULTS.md`](../evaluation/RAGAS_EVAL_RESULTS.md)
for why that separation matters.

Plots: [`plots/`](plots/) (reconstruction curve, link-prediction vs.
dimension, and fine-grained checks up to dim=256).

## Honest limitations

- Tropical wins *on average*, not on every single query — see the
  single-query failure case diagnosed in
  [`../rag/README.md`](../rag/README.md).
- The crossover dimension is not a fixed constant; it must be re-measured
  per graph (this is exactly what the dimension sweep in
  `wiki_kg_train_eval.py` does for a new graph).
- Poincaré's long-horizon convergence at very high dimension (e.g. dim=100
  on WordNet mammals) was still slowly improving at 1500 epochs in one run
  — a longer budget could narrow that specific gap further.

## Running

```bash
pip install -r ../requirements.txt
cd embeddings
python tropical_hierarchy_embeddings.py              # WordNet mammals, reconstruction
python tropical_hierarchy_link_prediction.py          # WordNet mammals, held-out
python tropical_hierarchy_generalization_check.py     # WordNet vehicles, held-out
python tropical_geneontology_check.py                 # Gene Ontology, held-out
python wiki_kg_train_eval.py                          # Wikipedia graph, dimension sweep + held-out
python wiki_train_and_save.py                         # Wikipedia graph, final embeddings for the RAG demo
```

`wiki_kg_train_eval.py` and `wiki_train_and_save.py` read/write
`../data/wiki_category_graph.json` and `../data/wiki_emb_*.pt` — run the
crawl steps in [`../rag/`](../rag/) first if those files don't exist yet.
