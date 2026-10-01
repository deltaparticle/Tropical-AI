# Evaluation: RAGAS scoring of the KG-RAG pipeline

Independent, industry-standard evaluation of the three-geometry RAG system
in [`../rag/`](../rag/), using [RAGAS](https://github.com/explodinggpt/ragas)
0.1.21 rather than hand-inspected answers.

**Full results, per-question breakdowns, and caveats:
[RAGAS_EVAL_RESULTS.md](RAGAS_EVAL_RESULTS.md).**

## Setup

- **Question set**: 15 hand-written AI/ML questions, each with a
  hand-written (not LLM-generated) reference answer, in
  [`eval_questions.py`](eval_questions.py) — avoids the circularity of
  grading against an LLM-authored "truth."
- **Embedding dimension**: 100, chosen from the independent held-out
  link-prediction sweep in [`../embeddings/wiki_kg_train_eval.py`](../embeddings/wiki_kg_train_eval.py)
  (tropical's held-out mAP there clearly beats both other geometries at
  this dimension) — decided before looking at any RAG/RAGAS output.
- **Content**: enriched for the categories this question set actually
  touches, via [`compute_touched_categories.py`](compute_touched_categories.py)
  + [`../rag/wiki_enrich_touched.py`](../rag/wiki_enrich_touched.py),
  applied identically across all three geometries.
- **Generator**: `openai/gpt-oss-20b` via Groq (same model for all three
  conditions — only the retrieval step differs).
- **Judge**: `meta-llama/llama-3.3-70b-instruct` via OpenRouter — 3.5x the
  generator's size, to reduce self/weak-evaluation bias.
- **Metrics**: Faithfulness, Answer Relevancy, Answer Correctness, Answer
  Similarity (Context Precision/Recall excluded — both multiply judge-LLM
  call volume several-fold for comparatively little extra signal).

## Reproducing

```bash
# 1. Main environment: regenerate answers for the fixed question set
pip install -r ../requirements.txt
python compute_touched_categories.py   # only if data/content changed
python kg_rag_export_for_ragas.py      # needs GROQ_API_KEY (or .groq_api_key)
                                        # -> writes ragas_eval_data.json

# 2. Isolated environment: score with RAGAS (different dependency pins)
python -m venv ragas_venv
source ragas_venv/bin/activate   # or ragas_venv\Scripts\activate on Windows
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key_here   # or create a .openrouter_api_key file here
python ragas_score.py
                                  # -> writes ragas_results.json,
                                  #    ragas_per_question_{geometry}.csv
```

## Files

| File | Contents |
|---|---|
| [`eval_questions.py`](eval_questions.py) | 15 questions + hand-written reference answers |
| [`compute_touched_categories.py`](compute_touched_categories.py) | Finds which categories the question set touches (for scoped content enrichment) |
| [`kg_rag_export_for_ragas.py`](kg_rag_export_for_ragas.py) | Runs the RAG pipeline on the question set, exports `{question, contexts, answer, ground_truth}` per geometry |
| [`ragas_score.py`](ragas_score.py) | Scores the exported data with RAGAS (runs inside `ragas_venv/`) |
| `ragas_eval_data.json` | Exported pipeline output (input to `ragas_score.py`) |
| `ragas_results.json` | Mean scores + valid-sample counts per geometry |
| `ragas_per_question_{euclidean,poincare,tropical}.csv` | Full per-question, per-metric scores |
| [`RAGAS_EVAL_RESULTS.md`](RAGAS_EVAL_RESULTS.md) | Full write-up: methodology, results tables, honest caveats |
