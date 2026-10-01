"""
wiki_enrich_touched.py

Targeted content enrichment, NOT a full 416-category re-crawl (that was
dropped earlier for being too slow/un-batchable). Scope: only the 99
categories actually touched (as seed or expansion result, across ALL THREE
geometries and all 15 eval questions) when running the real eval question
set through the live KG-RAG pipeline (touched_categories.json).

Root cause being fixed: build_context() in kg_rag_demo.py only ever uses
content[name]["summary"] -- categories with an empty summary (69 of the 99
touched ones) silently contribute ZERO context today, regardless of how
relevant the embedding ranked them. For those, fetch up to 3 real member
articles via the MediaWiki API and use their extracts as a stand-in
description, so the category has *some* real retrievable text.

This only ADDS content for categories that had none -- it does not touch
categories that already had a real summary, and it is applied identically
regardless of which geometry retrieved the category, so it cannot bias the
comparison between Euclidean/Poincare/Tropical.
"""

import json
import os
from wiki_crawl_step2 import batch_get_category_pages, batch_get_summaries

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MAX_ARTICLES_PER_CATEGORY = 3

with open(os.path.join(DATA_DIR, "touched_categories.json"), encoding="utf-8") as f:
    touched = json.load(f)

with open(os.path.join(DATA_DIR, "wiki_content.json"), encoding="utf-8") as f:
    content = json.load(f)

empty = [c for c in touched if not content.get(c, {}).get("summary", "").strip()]
print(f"Touched categories: {len(touched)}  |  empty summary: {len(empty)}")

print("Fetching up to 3 member articles per empty category...")
cat_to_articles = batch_get_category_pages(empty)

all_article_titles = sorted({t for arts in cat_to_articles.values() for t in arts})
print(f"Fetching extracts for {len(all_article_titles)} member articles (batched)...")
article_summaries = batch_get_summaries(all_article_titles)

filled = 0
for cat in empty:
    titles = cat_to_articles.get(cat, [])[:MAX_ARTICLES_PER_CATEGORY]
    pieces = []
    articles_dict = {}
    for t in titles:
        s = article_summaries.get(t, "").strip()
        if s:
            pieces.append(f"{t}: {s[:300]}")
            articles_dict[t] = s[:300]
    if pieces:
        content.setdefault(cat, {})["summary"] = " | ".join(pieces)[:800]
        content[cat]["articles"] = articles_dict
        filled += 1

print(f"Filled real content for {filled}/{len(empty)} previously-empty touched categories")

out_path = os.path.join(DATA_DIR, "wiki_content.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(content, f, indent=1)
print(f"Saved {out_path}")
