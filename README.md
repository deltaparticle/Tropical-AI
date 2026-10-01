# Tropical AI

Tropical (max-plus) geometry embeddings for hierarchical/graph-structured
data — benchmarked against Euclidean and Poincaré (hyperbolic) embeddings on
real, held-out link prediction across 4 independent real-world datasets,
then deployed as the retrieval step of a live knowledge-graph RAG system and
scored with an industry-standard RAG evaluation framework (RAGAS).

CPU-only throughout. No synthetic data: every dataset is a real ontology,
taxonomy, or live-crawled knowledge graph.

## The question

Poincaré (hyperbolic) embeddings are the well-established choice for
embedding hierarchies — their exponential volume growth near the ball
boundary matches tree-like branching. **Tropical (max-plus) geometry** is
far less explored for this purpose. Its distance metric,
`d(u,v) = max_i(u_i - v_i) - min_i(u_i - v_i)`, is piecewise-linear and
depends on only 2 of the embedding's `d` dimensions per comparison — a very
different structure from both Euclidean and Poincaré distance. Does it
actually work, and if so, where?

## What this project does, in three parts

### 1. [`embeddings/`](embeddings/) — does tropical geometry generalize?

Trains all three geometries on real `is_a`/category-graph edges and tests
**held-out link prediction** (train on 90% of relation pairs, predict the
held-out 10% from geometry alone) — not reconstruction of the training
graph, which mostly measures memorization. Validated on 4 independently-
sourced real datasets: WordNet mammals (1,170 nodes), WordNet vehicles
(520 nodes, a different domain), the Gene Ontology's immune-system-process
subtree (553 nodes — real labels used in protein-function-prediction
models), and a live-crawled Wikipedia category graph (416 nodes).

**Consistent finding across all 4**: Poincaré wins at very low dimension,
tropical wins at moderate dimension, and the crossover point scales with
graph size. On the Wikipedia graph at dim=100, tropical's held-out mAP
(0.261) is roughly **2.2x** both Euclidean (0.107) and Poincaré (0.119).
Confirmed not to be an artifact: training on completely shuffled/fake edges
keeps held-out mAP pinned at the random-chance floor the entire time, even
as training loss drops normally.

→ [embeddings/README.md](embeddings/README.md) for methodology, formulas,
full results table, and plots.

### 2. [`rag/`](rag/) — does it matter in a real application?

A live knowledge-graph RAG system over the Wikipedia "Artificial
Intelligence" category graph (crawled via the public MediaWiki API). A
fixed TF-IDF seed match (identical across all three geometries, so it can't
bias the comparison) is followed by geometry-specific graph expansion — the
exact held-out link-prediction task from part 1, now used live as a
retrieval step — then generation via an LLM (`openai/gpt-oss-20b` via Groq).
Ships as both a CLI and a local interactive web app showing all three
geometries' retrieved context and generated answers side by side.

→ [rag/README.md](rag/README.md) for the pipeline, a diagnosed failure
case, and how to run it.

### 3. [`evaluation/`](evaluation/) — is the RAG difference real, by an objective standard?

Scored with [RAGAS](https://github.com/explodinggpt/ragas), an
industry-standard RAG evaluation framework, on 15 hand-written AI/ML
questions with hand-written reference answers — judged by a 70B model
(`meta-llama/llama-3.3-70b-instruct` via OpenRouter, substantially larger
than the 20B generator, to reduce self/weak-evaluation bias). The embedding
dimension (100) was chosen from part 1's independent held-out task *before*
looking at any RAG output, specifically so the choice couldn't be reverse-
engineered to flatter one geometry.

**Result**: tropical wins 3 of 4 metrics — Answer Relevancy, Answer
Correctness, and Answer Similarity — each with 14-15 of 15 valid samples.
Poincaré remains the most faithful-to-context geometry. No geometry sweeps
every metric, which is part of why the result is credible rather than
suspicious.

→ [evaluation/RAGAS_EVAL_RESULTS.md](evaluation/RAGAS_EVAL_RESULTS.md) for
full results, per-question breakdowns, and honest caveats.

## Repository layout

```
embeddings/   4-dataset held-out link-prediction benchmark (Euclidean vs. Poincaré vs. Tropical)
rag/          Live knowledge-graph RAG system using the trained embeddings for retrieval
evaluation/   RAGAS scoring of the RAG system's answers
data/         Shared artifacts: crawled graph/content, trained embeddings
```

## Tech stack

PyTorch (custom Riemannian SGD for Poincaré, annealed Maslov-dequantized
training for tropical), scikit-learn (TF-IDF retrieval), Flask, the public
MediaWiki API, Groq (LLM generation), OpenRouter (LLM-as-judge), RAGAS,
goatools (Gene Ontology parsing), NLTK WordNet. All CPU-only — no GPU
required anywhere in this project.

## Setup

```bash
pip install -r requirements.txt
```

Each of the three folders has its own README with specific run instructions
and its own API-key requirements (`GROQ_API_KEY` for generation,
`OPENROUTER_API_KEY` for RAGAS judging — both read from an environment
variable first, falling back to a local `.groq_api_key` /
`.openrouter_api_key` file that is never committed).

## Honest limitations

- Tropical wins *on average* across datasets and metrics, not on every
  single query — see the diagnosed failure case in
  [rag/README.md](rag/README.md).
- The RAGAS evaluation uses 15 questions — enough to move past pure
  "vibes," but far smaller than the hundreds of held-out relation pairs
  used in the quantitative embedding benchmark.
- Retrieved Wikipedia content is short extracts, not full articles — a
  real constraint of the live crawl, documented in
  [evaluation/RAGAS_EVAL_RESULTS.md](evaluation/RAGAS_EVAL_RESULTS.md).

See each folder's README for the full, dataset-by-dataset detail.
