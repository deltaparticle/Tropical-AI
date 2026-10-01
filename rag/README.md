# Knowledge-graph RAG: tropical vs. Poincaré vs. Euclidean retrieval

A real, live GraphRAG system that uses the graph embeddings from
[`../embeddings/`](../embeddings/) as the *retrieval* step of a
retrieval-augmented-generation pipeline — not another synthetic benchmark.

## Data source

Wikipedia's "Artificial Intelligence" category tree, crawled live via the
public MediaWiki API: **416 categories, 475 edges**. Chosen so the pipeline
could be evaluated on content any technical reader can judge for themselves,
without specialist domain literacy.

- [`wiki_kg_rag.py`](wiki_kg_rag.py) — crawler primitives (`_api_get` with
  retry/backoff, `get_subcategories`, `get_page_summary`,
  `fetch_category_graph`).
- [`wiki_crawl_step1.py`](wiki_crawl_step1.py) — BFS crawl from the root
  category → `../data/wiki_category_graph.json`.
- [`wiki_crawl_step2.py`](wiki_crawl_step2.py) — batched content fetch
  (MediaWiki's `prop=extracts` allows up to 50 titles per request) →
  `../data/wiki_content.json`.
- [`wiki_enrich_touched.py`](wiki_enrich_touched.py) — targeted content
  enrichment: the context-building step only ever uses a category's
  `summary` text, and many categories (pure category pages, no matching
  Wikipedia article) have none. For the categories that the evaluation
  question set actually touches (see
  [`../evaluation/compute_touched_categories.py`](../evaluation/compute_touched_categories.py)),
  this fetches real member-article extracts as a stand-in description —
  applied identically regardless of which geometry retrieved a given
  category, so it can't bias the comparison.

## Pipeline

A free-text question goes through two stages, and **only the second stage
differs between the three geometries**:

1. **Seed match** (identical across all three conditions, so it can't bias
   the comparison): fixed TF-IDF similarity over category
   titles + summaries picks the single best-matching "seed" category.
2. **Graph expansion** (the actual experiment): from that seed, each
   geometry's trained embedding (`../data/wiki_emb_{euclidean,poincare,tropical}.pt`)
   ranks every other category by distance and returns the top-k closest —
   exactly the held-out link-prediction task from `../embeddings/`, now
   used live as a retrieval step instead of an offline metric.
3. **Generation**: the seed + expanded categories' content is assembled into
   a context block and passed to `openai/gpt-oss-20b` via Groq
   ([`groq_client.py`](groq_client.py), rate-limited and retry-safe).

## Files

- [`kg_rag_demo.py`](kg_rag_demo.py) — interactive CLI:
  `python kg_rag_demo.py "your question"`, or run with no arguments for a
  loop.
- [`webapp.py`](webapp.py) — local Flask site (`python webapp.py`, then open
  `http://127.0.0.1:5000`): type a question, see all three geometries'
  retrieved context and generated answer side by side.
- [`groq_client.py`](groq_client.py) — minimal rate-limited Groq client.

## Diagnosed failure case

A "history of AI" query's true-answer node was a direct 1-hop child of the
graph's root, yet Poincaré ranked it 213th and tropical 155th out of 416
(Euclidean got it right at rank 9). Root cause: high-degree hub nodes are
intrinsically hard to rank for any embedding geometry, and tropical's
specific failure mode traces to its metric depending on only 2 of `d`
dimensions per comparison — a structural property of the geometry, not a
bug. Worth stating plainly: tropical wins *on average* in the held-out and
RAGAS evaluations, not on every single query.

## Running

```bash
pip install -r ../requirements.txt
cd rag
python wiki_crawl_step1.py      # only needed if ../data/wiki_category_graph.json doesn't exist
python wiki_crawl_step2.py      # only needed if ../data/wiki_content.json doesn't exist
```

Train embeddings first (see [`../embeddings/`](../embeddings/)), then:

```bash
export GROQ_API_KEY=your_key_here   # or create a .groq_api_key file in this folder
python kg_rag_demo.py "What is reinforcement learning?"
python webapp.py                    # http://127.0.0.1:5000
```

Without a Groq key, the demo/webapp still runs in retrieval-only mode (shows
the retrieved categories, skips generation).
