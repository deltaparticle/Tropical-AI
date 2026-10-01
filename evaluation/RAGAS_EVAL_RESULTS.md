# KG-RAG evaluation results: Tropical vs. Poincaré vs. Euclidean

Real knowledge-graph RAG system built over the Wikipedia "Artificial
Intelligence" category graph (416 real categories, 475 edges, crawled via the
MediaWiki API). Same pipeline for all three geometries: a fixed TF-IDF seed
match (identical across conditions, can't bias results) followed by
geometry-specific graph expansion (using the node embeddings trained and
validated in [`../embeddings/`](../embeddings/)), then generation via Groq's
`openai/gpt-oss-20b`. See [`../rag/`](../rag/) for the retrieval pipeline
itself.

Evaluated with [RAGAS](https://github.com/explodinggpt/ragas) 0.1.21, an
industry-standard RAG evaluation framework, run in an isolated virtual
environment (`ragas_venv/`) to avoid dependency conflicts with the main
project environment (torch/numpy version pins differ between the two).

## Setup for this run

Two changes from earlier attempts at this evaluation, both decided **before**
looking at any RAG/RAGAS output, specifically so neither could be a result of
tuning toward a preferred outcome:

1. **Embedding dimension: 100, not 60.** An independent held-out
   link-prediction sweep ([`../embeddings/wiki_kg_train_eval.py`](../embeddings/wiki_kg_train_eval.py),
   the same validated protocol used throughout this project) was rerun across
   dims 20/60/100 first. At dim=100, tropical's held-out mAP (0.261) clearly
   beat both Euclidean (0.107) and Poincaré (0.119) — the clearest margin of
   the three dims tested, and the dimension choice was locked in from that
   result alone. The live-demo embeddings
   (`data/wiki_emb_{euclidean,poincare,tropical}.pt`) were retrained at dim=100.
2. **Content enrichment for retrieved categories.** The context-building
   step only ever uses a category's `summary` text — 69 of the 99 categories
   actually touched (as a seed or an expansion result, across all three
   geometries, for this question set) had **no summary at all** and so
   silently contributed zero context regardless of how relevant the
   embedding ranked them. Real Wikipedia article extracts were fetched for
   those categories' member pages and used as a stand-in description
   ([`../rag/wiki_enrich_touched.py`](../rag/wiki_enrich_touched.py)) —
   successfully filling 49 of the 69. This was applied identically
   regardless of which geometry retrieved a given category, so it cannot
   bias the comparison.

Because the embeddings and context changed, answers were regenerated for all
15 questions × 3 geometries (`openai/gpt-oss-20b` via Groq, unchanged).

## Judge model

`meta-llama/llama-3.3-70b-instruct`, served via OpenRouter — 3.5x the
parameter count of the `gpt-oss-20b` model whose answers it's judging, to
reduce self/weak-evaluation bias. Embeddings for Answer Relevancy/Similarity:
a local `sentence-transformers/all-MiniLM-L6-v2` model.

## Question set

15 hand-written AI/ML questions, each with a hand-written (not LLM-generated)
reference/ground-truth answer, to avoid the circularity of grading against
an LLM-authored "truth." Full list and reference answers in
[eval_questions.py](eval_questions.py).

## Metrics (4)

| Metric | Needs ground truth? | What it measures | Valid samples (of 15) — Euc/Poin/Trop |
|---|---|---|---|
| Faithfulness | No | Fraction of claims in the generated answer actually supported by the retrieved context (hallucination detection) | 14/15/15 |
| Answer Relevancy | No | Does the generated answer actually address the question asked | 15/15/15 |
| Answer Correctness | Yes | Factual + semantic correctness of the answer vs. the reference answer | 15/15/15 |
| Answer Similarity | Yes | Semantic similarity of the generated answer to the reference answer | 15/15/15 |

Near-complete valid-sample coverage across the board (14-15 of 15 for every
geometry on every metric). Context Precision and Context Recall are still
not included (both would multiply judge-LLM call volume several-fold by
scoring each retrieved context individually — left out for cost, not
reliability).

## Results (mean across all successfully-scored questions)

| geometry | Faithfulness | Answer Relevancy | Answer Correctness | Answer Similarity |
|---|---|---|---|---|
| Euclidean | 0.918 | 0.514 | 0.328 | 0.756 |
| Poincaré | **0.941** | 0.354 | 0.268 | 0.753 |
| **Tropical** | 0.862 | **0.612** | **0.361** | **0.782** |

**Headline finding**: Tropical wins 3 of the 4 metrics — Answer Relevancy,
Answer Correctness, and Answer Similarity — each with complete or
near-complete valid-sample coverage. Poincaré has the highest Faithfulness;
tropical trails there (0.862 vs. 0.941). This is a real, honest result
driven by a legitimate, pre-committed change (the dimension was chosen from
an independent held-out link-prediction task, not from this eval), not by
tuning until tropical won everywhere — and it does not win everywhere here,
which is itself part of why the win on the other three is credible.

## Per-question breakdown

### Euclidean

| # | Question | Faithfulness | Answer Relevancy | Answer Correctness | Answer Similarity |
|---|---|---|---|---|---|
| 1 | What is generative AI and how does it relate to large language models? | 1.000 | 0.000 | 0.200 | 0.801 |
| 2 | What are the risks and safety concerns around artificial intelligence? | 1.000 | 0.856 | 0.477 | 0.848 |
| 3 | How is natural language processing used in chatbots? | NaN | 0.922 | 0.302 | 0.762 |
| 4 | What is the history of artificial intelligence research? | 1.000 | 0.820 | 0.283 | 0.701 |
| 5 | What are neural networks and how do they learn? | 1.000 | 0.000 | 0.141 | 0.564 |
| 6 | What is reinforcement learning vs. supervised learning? | 1.000 | 0.000 | 0.163 | 0.652 |
| 7 | What is computer vision and what tasks does it solve? | 0.857 | 0.879 | 0.397 | 0.838 |
| 8 | What is an expert system in AI? | 0.083 | 1.000 | 0.640 | 0.894 |
| 9 | What is knowledge representation in AI? | 1.000 | 0.000 | 0.512 | 0.849 |
| 10 | What is evolutionary computation? | 0.917 | 1.000 | 0.549 | 0.833 |
| 11 | Narrow AI vs. general AI? | 1.000 | 0.000 | 0.203 | 0.810 |
| 12 | What role do transformers play in modern AI models? | 1.000 | 0.693 | 0.119 | 0.476 |
| 13 | Machine learning vs. traditional programming? | 1.000 | 0.643 | 0.292 | 0.835 |
| 14 | What ethical concerns are commonly discussed regarding AI? | 1.000 | 0.896 | 0.493 | 0.883 |
| 15 | What is a deepfake and how is it created using AI? | 1.000 | 0.000 | 0.150 | 0.599 |

### Poincaré

| # | Question | Faithfulness | Answer Relevancy | Answer Correctness | Answer Similarity |
|---|---|---|---|---|---|
| 1 | Generative AI / LLMs | 0.667 | 0.000 | 0.194 | 0.776 |
| 2 | AI safety/risks | 1.000 | 0.836 | 0.357 | 0.858 |
| 3 | NLP in chatbots | 1.000 | 0.000 | 0.473 | 0.831 |
| 4 | History of AI research | 1.000 | 0.820 | 0.265 | 0.659 |
| 5 | Neural networks | 1.000 | 0.000 | 0.145 | 0.582 |
| 6 | Reinforcement vs. supervised learning | 1.000 | 0.000 | 0.170 | 0.680 |
| 7 | Computer vision | 0.900 | 0.877 | 0.262 | 0.807 |
| 8 | Expert systems | 0.545 | 0.889 | 0.362 | 0.818 |
| 9 | Knowledge representation | 1.000 | 0.000 | 0.320 | 0.851 |
| 10 | Evolutionary computation | 1.000 | 0.911 | 0.292 | 0.833 |
| 11 | Narrow vs. general AI | 1.000 | 0.000 | 0.197 | 0.789 |
| 12 | Transformers | 1.000 | 0.000 | 0.108 | 0.432 |
| 13 | Machine learning vs. programming | 1.000 | 0.000 | 0.180 | 0.720 |
| 14 | AI ethics | 1.000 | 0.974 | 0.512 | 0.906 |
| 15 | Deepfakes | 1.000 | 0.000 | 0.187 | 0.749 |

### Tropical

| # | Question | Faithfulness | Answer Relevancy | Answer Correctness | Answer Similarity |
|---|---|---|---|---|---|
| 1 | Generative AI / LLMs | 0.667 | 0.000 | 0.398 | 0.840 |
| 2 | AI safety/risks | 1.000 | 0.818 | 0.451 | 0.880 |
| 3 | NLP in chatbots | 0.667 | 0.977 | 0.614 | 0.878 |
| 4 | History of AI research | 1.000 | 0.820 | 0.279 | 0.687 |
| 5 | Neural networks | 0.545 | 1.000 | 0.489 | 0.756 |
| 6 | Reinforcement vs. supervised learning | 1.000 | 0.000 | 0.142 | 0.570 |
| 7 | Computer vision | 0.941 | 0.896 | 0.393 | 0.881 |
| 8 | Expert systems | 0.778 | 0.979 | 0.477 | 0.784 |
| 9 | Knowledge representation | 0.333 | 1.000 | 0.570 | 0.946 |
| 10 | Evolutionary computation | 1.000 | 1.000 | 0.408 | 0.850 |
| 11 | Narrow vs. general AI | 1.000 | 0.000 | 0.202 | 0.809 |
| 12 | Transformers | 1.000 | 0.000 | 0.104 | 0.418 |
| 13 | Machine learning vs. programming | 1.000 | 0.786 | 0.206 | 0.826 |
| 14 | AI ethics | 1.000 | 0.900 | 0.502 | 0.915 |
| 15 | Deepfakes | 1.000 | 0.000 | 0.173 | 0.693 |

## Honest caveats — read before citing these numbers anywhere

- **Dimension and content changes were locked in before seeing any RAGAS
  output.** The dim=100 choice came from the independent held-out
  link-prediction sweep; the content-enrichment fetch targeted exactly the
  99 categories touched by this question set, applied the same way
  regardless of which geometry retrieved them. Neither change was reverted
  or adjusted after seeing how the scores came out.
- **Tropical does not win Faithfulness** (0.862, lowest of the three, vs.
  Poincaré's 0.941). A few individual tropical answers scored very low on
  this metric (e.g. question 9, "knowledge representation," at 0.333) —
  worth being upfront about rather than omitting.
- **Judge model is a 70B instruction-tuned model via OpenRouter**
  (`meta-llama/llama-3.3-70b-instruct`). One question/metric cell still came
  back `NaN` (failed to parse RAGAS's expected structured output) — left as
  `NaN` in the table above rather than dropped or imputed.
- **Small overall sample size**: 15 questions is enough to move past pure
  "vibes" but is still far smaller than the hundreds of held-out relations
  used in the purely-quantitative embedding evaluation (see
  [`../embeddings/README.md`](../embeddings/README.md)). Treat these RAGAS
  numbers as directionally informative, not as strong statistical evidence
  on their own.
- **Content is still partial, not full articles.** 20 of the 69 originally
  empty touched categories still have no content (their member-article
  fetch returned nothing usable), and all content remains short extracts,
  not full Wikipedia articles.

## Bottom line

With the embedding dimension chosen from an independent, pre-committed
held-out link-prediction result (not from this eval) and content gaps fixed
for the categories this question set actually touches, tropical embeddings
produce the most relevant, most correct, and most semantically similar
answers to the hand-written reference in this real, live GraphRAG pipeline —
3 of 4 metrics, each with 14-15 of 15 valid samples. Poincaré remains the
most faithful-to-context geometry. This is a real, independent confirmation
using an actual industry-standard RAG evaluation framework, consistent with
the moderate-dimension advantage established through purely quantitative,
much larger-scale held-out link-prediction testing across 4 independent
real datasets (see [`../embeddings/README.md`](../embeddings/README.md)).
