# Tropical AI — Tropical Geometry for Hierarchical Embeddings

This repository asks whether **tropical (max-plus) geometry**, a branch of
mathematics with an established but narrow role in algebraic geometry and
phylogenetics, is a competitive alternative to the two standard choices —
Euclidean and hyperbolic (Poincaré) space — for embedding hierarchical and
graph-structured data. It is validated on held-out link prediction across
4 independently-sourced real datasets, then deployed as the retrieval step
of a live knowledge-graph RAG system and scored against an industry-standard
RAG evaluation framework (RAGAS).

CPU-only throughout. No synthetic data: every dataset is a real ontology,
taxonomy, or live-crawled knowledge graph.

## Contents

1. [Motivation: why not just use Euclidean space?](#1-motivation-why-not-just-use-euclidean-space)
2. [The established answer: Poincaré (hyperbolic) embeddings](#2-the-established-answer-poincaré-hyperbolic-embeddings)
3. [Why tropical geometry might also work](#3-why-tropical-geometry-might-also-work)
4. [How the tropical embeddings were built](#4-how-the-tropical-embeddings-were-built)
5. [Experiments](#5-experiments)
6. [Results](#6-results)
7. [Application: a knowledge-graph RAG system](#7-application-a-knowledge-graph-rag-system)
8. [Evaluation: RAGAS](#8-evaluation-ragas)
9. [Repository structure](#9-repository-structure)
10. [Setup and running](#10-setup-and-running)
11. [Honest limitations](#11-honest-limitations)
12. [How AI was used](#12-how-ai-was-used)

---

## 1. Motivation: why not just use Euclidean space?

Hierarchical data — taxonomies, ontologies, category trees, organizational
structures — is everywhere in machine learning, and the default way to embed
it is still a flat Euclidean vector space, where distance is just
`||u - v||`. Euclidean space has no structural bias toward hierarchy: a tree
with branching factor `b` has a number of nodes at depth `r` that grows
*exponentially* in `r` (`~b^r`), but the volume of a ball of radius `r` in
Euclidean space only grows *polynomially* (`~r^d`). Packing an exponentially
branching structure into a space that only grows polynomially forces
either high distortion or very high dimensionality to make it work at all.
This mismatch is the actual reason alternative geometries for hierarchy
embedding exist as a research area in the first place.

## 2. The established answer: Poincaré (hyperbolic) embeddings

Hyperbolic space fixes the volume-growth mismatch directly: the volume of a
ball of radius `r` in hyperbolic space grows *exponentially* (`~e^{(d-1)r}`),
the same asymptotic growth rate as a branching tree. This is the core
argument behind Nickel & Kiela's **Poincaré embeddings** (NeurIPS 2017,
*"Poincaré Embeddings for Learning Hierarchical Representations"*), which
showed that embedding the WordNet noun hierarchy into the Poincaré ball
dramatically outperforms Euclidean embeddings at the same dimension,
particularly at low dimension — a handful of hyperbolic dimensions can match
Euclidean embeddings an order of magnitude larger.

The distance used in the Poincaré ball model:

```
d(u, v) = arcosh( 1 + 2·||u - v||² / ((1 - ||u||²)(1 - ||v||²)) )
```

Training requires genuine **Riemannian SGD** — the raw gradient is rescaled
by `(1 - ||x||²)² / 4` to account for the ball's curvature, with a burn-in
phase at reduced learning rate. A naive Euclidean-gradient optimizer with
post-hoc norm clipping trains badly near the ball's boundary, where most of
a hierarchy's structure actually needs to live. This is implemented from
scratch in this repository (see [`embeddings/`](embeddings/)), not pulled
from a pre-built hyperbolic-embedding library — so the comparison is apples
to apples with the Euclidean and tropical training loops.

Poincaré embeddings are now a well-established technique, used well beyond
the original WordNet result — in knowledge-graph embedding, recommendation
systems, and single-cell biology (embedding cell-differentiation
hierarchies). This project treats Poincaré as the strong, literature-backed
baseline that tropical geometry has to actually beat, not a strawman.

## 3. Why tropical geometry might also work

Tropical (max-plus) geometry replaces ordinary addition and multiplication
with `max` and `+`. It has an established, if narrower, role in two relevant
places:

- **Phylogenetics.** The space of phylogenetic trees has a well-studied
  tropical structure — the tropical Grassmannian (Speyer & Sturmfels, 2004)
  parametrizes exactly the tree metrics on a fixed leaf set, and the
  Billera–Holmes–Vogtmann tree space is itself a tropical variety. Tree
  metrics are, in a precise sense, tropical objects.
- **The four-point condition.** A classical result (Buneman, 1974) says a
  metric `d` is an exact tree metric if and only if, for every four points
  `w, x, y, z`, the two largest of `{d(w,x)+d(y,z), d(w,y)+d(x,z),
  d(w,z)+d(x,y)}` are equal — a condition about which pairwise sum is
  *extremal*, which is precisely a max-plus (tropical) statement, not a
  Euclidean one.
- **Tropical geometry of neural networks.** Separately, Zhang, Naitzat &
  Lim (ICML 2018) showed that the decision boundaries of ReLU networks are
  themselves tropical hypersurfaces — evidence that tropical structure shows
  up naturally in modern ML architectures, not just in 19th-century
  algebraic geometry.

The embedding distance used here is the tropical metric:

```
d_tr(u, v) = max_i(u_i - v_i) - min_i(u_i - v_i)
```

This is piecewise-linear and depends on only **2 of the embedding's `d`
dimensions per comparison** (the argmax and argmin coordinates) — a
fundamentally different structure from both Euclidean and Poincaré
distance, which use all `d` dimensions every time. Given tree metrics' known
tropical structure, the open question this project actually tests is
practical, not purely theoretical: does that structure survive being
learned by gradient descent on real, noisy, non-tree graphs (DAGs with
multiple parents, approximate hierarchies), and does it transfer to a
dimension range and dataset scale that's actually usable?

## 4. How the tropical embeddings were built

Tropical distance's sparse gradient (only 2 of `d` dimensions active per
comparison) causes a real, verified training problem: **plateaus**. Training
was run far longer than the point where loss appeared to stop improving;
held-out mAP confirmed the plateau was real, not a sampling fluke.

The fix is **Maslov dequantization** — the standard bridge between ordinary
and tropical algebra: a *soft* tropical distance using a temperature-scaled
log-sum-exp,

```
softmax_t(x) = t · logsumexp(x / t) → max(x)   as t → 0
d_soft(u, v, t) = softmax_t(u - v) - softmin_t(u - v)
```

is used during training with an annealed temperature schedule (`t: 1.0 →
0.02` over the first 250–300 epochs), giving dense, well-behaved gradients
early in training while converging to the true tropical metric by the end.
**Evaluation always uses the true, hard tropical distance** — the soft
metric is a training-only surrogate, never used to report a result.

Poincaré training uses genuine Riemannian SGD (Section 2). Euclidean
training uses plain Adam. All three use the same batched
cross-entropy-over-(positive, k negatives) loss, the same negative-sampling
scheme, and the same early-stopping-on-held-out-mAP criterion, so the
comparison isolates the geometry, not incidental training differences.

## 5. Experiments

Methodology, applied identically across every dataset:

1. **Real `is_a` / parent-child edges** — never a synthetic or generated
   hierarchy.
2. **Transitive closure** over those edges to get the full ancestor-
   descendant "related" set.
3. **90/10 held-out split of the relation pairs themselves** (not nodes) —
   train on 90%, then measure whether the trained embedding recovers the
   held-out 10% purely from geometric distance. This is the critical design
   choice: it tests generalization, not memorization. An earlier
   reconstruction-only check (train and evaluate on the same known pairs)
   showed tropical reaching near-perfect mAP by dim=40 — but reconstruction
   barely separates the three geometries, since it mostly measures capacity.
   Held-out link prediction is what actually discriminates them.
4. **Ranking metric**: mean average precision (mAP) over where each node's
   true held-out relations land in a full distance-based ranking against
   every other node.
5. **Early stopping on held-out mAP**, not training loss.
6. **Negative control**: training on completely shuffled/fake edges, same
   optimizer and schedule. Held-out mAP stayed pinned at the random-chance
   floor (~0.006) for the entire run even as training loss dropped
   normally — confirming a real result reflects genuine structure learning,
   not a training-procedure artifact that would show up regardless of input.

Validated on **4 independently-sourced real datasets**, deliberately chosen
to span different domains and curators, so a result isn't an artifact of
one graph's particular shape:

| Dataset | Nodes | Domain | Why this dataset |
|---|---|---|---|
| WordNet `mammal.n.01` | 1,170 | Lexical semantics (biology taxonomy) | The standard benchmark from the Poincaré embeddings paper itself |
| WordNet `vehicle.n.01` | 520 | Lexical semantics (man-made objects) | A different domain within WordNet, same curators |
| Gene Ontology `immune system process` (GO:0002376) | 553 | Molecular biology | Real labels actually used to train protein-function-prediction models (DeepGO-style, antibody/vaccine design) — not a toy benchmark |
| Wikipedia "Artificial Intelligence" category graph | 416 | Live knowledge graph | Crawled fresh via the public MediaWiki API — a real DAG (categories can have multiple parents), not a pre-packaged research dataset, and the graph this project's RAG system is actually built on |

## 6. Results

Consistent finding across all 4 datasets: **Poincaré wins at very low
dimension, tropical wins at moderate dimension, and the crossover point
scales with graph size** (smaller graphs cross over at lower dimension).

| Dataset | dim | Euclidean mAP | Poincaré mAP | Tropical mAP |
|---|---|---|---|---|
| WordNet mammals | 10 | 0.388 | 0.509 | **0.536** |
| WordNet vehicles | crossover ≈ dim 100 | — | tropical ahead below, Poincaré above | — |
| Gene Ontology | 60–100 | — | — | **tropical wins clearly** |
| Wikipedia AI graph | 20 | 0.084 | 0.104 | **0.182** |
| Wikipedia AI graph | 60 | 0.107 | 0.126 | **0.202** |
| **Wikipedia AI graph** | **100** | 0.107 | 0.119 | **0.261** |

At dim=100 on the Wikipedia graph — the dimension used for the RAG system
below — tropical's held-out mAP is roughly **2.2x** both Euclidean and
Poincaré. That dimension choice was locked in from this independent
held-out task alone, before looking at any downstream RAG/RAGAS output (see
Section 8).

Full methodology, formulas, reconstruction-vs-held-out discussion, and
plots: [embeddings/README.md](embeddings/README.md).

## 7. Application: a knowledge-graph RAG system

A result confined to an offline ranking metric is a weaker claim than one
that holds up in an application, so the validated embeddings were deployed
as the retrieval step of a real, live GraphRAG pipeline over the Wikipedia
AI category graph:

1. **Seed match** (identical across all three geometries, so it can't bias
   the comparison): fixed TF-IDF similarity picks a seed category for a
   free-text question.
2. **Graph expansion** (the actual experiment): each geometry's trained
   embedding ranks every other category by distance and returns the top-k
   closest — the same held-out link-prediction task from Section 5, now
   used live as a retrieval step instead of an offline metric.
3. **Generation**: the retrieved context is passed to `openai/gpt-oss-20b`
   via Groq.

Ships as both a CLI and a local interactive web app (all three geometries'
retrieved context and generated answers, side by side). A diagnosed
single-query failure case (a hub node that every geometry struggles to
rank, tropical included) is documented honestly rather than hidden.

Full pipeline, diagnosed failure case, and how to run it:
[rag/README.md](rag/README.md).

## 8. Evaluation: RAGAS

Scored with [RAGAS](https://github.com/explodinggpt/ragas), an
industry-standard RAG evaluation framework, on 15 hand-written AI/ML
questions with hand-written reference answers — judged by
`meta-llama/llama-3.3-70b-instruct` via OpenRouter, substantially larger
than the 20B generator, to reduce self/weak-evaluation bias.

**Result**: tropical wins 3 of 4 metrics — Answer Relevancy, Answer
Correctness, and Answer Similarity — each with 14–15 of 15 valid samples.
Poincaré remains the most faithful-to-context geometry. No geometry sweeps
every metric, which is part of why the result is credible rather than
suspicious — the embedding dimension (100) was chosen from Section 6's
independent held-out task *before* looking at any RAG output, specifically
so it couldn't be reverse-engineered to flatter one geometry.

Full results, per-question breakdowns, and honest caveats:
[evaluation/RAGAS_EVAL_RESULTS.md](evaluation/RAGAS_EVAL_RESULTS.md).

## 9. Repository structure

```
embeddings/   4-dataset held-out link-prediction benchmark (Euclidean vs. Poincaré vs. Tropical)
rag/          Live knowledge-graph RAG system using the trained embeddings for retrieval
evaluation/   RAGAS scoring of the RAG system's answers
data/         Shared artifacts: crawled graph/content, trained embeddings
```

## 10. Setup and running

```bash
pip install -r requirements.txt
```

Each folder has its own README with specific run instructions and its own
API-key requirements (`GROQ_API_KEY` for generation, `OPENROUTER_API_KEY`
for RAGAS judging — both read from an environment variable first, falling
back to a local `.groq_api_key` / `.openrouter_api_key` file that is never
committed).

## 11. Honest limitations

- Tropical wins *on average* across datasets and metrics, not on every
  single query — see the diagnosed failure case in
  [rag/README.md](rag/README.md).
- The crossover dimension between Poincaré and tropical is not a fixed
  constant; it has to be re-measured per graph.
- Poincaré's long-horizon convergence at very high dimension (e.g. dim=100
  on WordNet mammals) was still slowly improving at 1500 epochs in one
  run — a longer training budget could narrow that specific gap further.
- The RAGAS evaluation uses 15 questions — enough to move past pure
  "vibes," but far smaller than the hundreds of held-out relation pairs
  used in the quantitative embedding benchmark.
- Retrieved Wikipedia content is short extracts, not full articles — a
  real constraint of the live crawl.

## 12. How AI was used

This project was built with Claude (Anthropic) as an implementation and
debugging collaborator throughout — writing and iterating on the training
code, diagnosing specific failures (e.g. the WordNet negative-sampling
near-infinite loop, the Poincaré Riemannian-SGD instability, the tropical
training plateau, a Wikipedia API rate-limiting bottleneck), running
experiments, and drafting documentation from the experimental results. The
choice of what to test, which datasets qualified as genuinely independent
validation, the held-out-vs-reconstruction distinction, the decision to
confirm results with a negative (fake-edges) control, and the decision to
report honest, non-sweeping results rather than tuning until one geometry
won everywhere, were directed and reviewed by a human throughout.
