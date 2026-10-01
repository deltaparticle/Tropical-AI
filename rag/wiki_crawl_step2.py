"""Step 2 (rewritten for speed): batch-fetch real Wikipedia article/category
summaries. MediaWiki's API allows up to 50 titles per prop=extracts request,
so instead of one HTTP call per article (which was crawling into rate-limit
backoff and taking hours), we fetch in batches of 50 -- a ~10-50x reduction
in request count. Saves to wiki_content.json."""

import json
import os
import time
from wiki_kg_rag import _api_get

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MAX_ARTICLES_PER_CATEGORY = 3
BATCH_SIZE = 50


def batch_get_summaries(titles, batch_size=BATCH_SIZE, delay=0.5):
    """titles: list[str] -> dict[str, str] (title -> plain-text intro extract)."""
    out = {}
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        data = _api_get({
            "action": "query", "prop": "extracts", "exintro": True, "explaintext": True,
            "titles": "|".join(batch), "format": "json",
        })
        if data is not None:
            for _, page in data.get("query", {}).get("pages", {}).items():
                title = page.get("title")
                extract = page.get("extract", "") or ""
                if title:
                    out[title] = extract[:800]
        time.sleep(delay)
    return out


def batch_get_category_pages(cat_titles, delay=0.3):
    """cat_titles: list[str] -> dict[str, list[str]] (category -> article titles in it).
    categorymembers doesn't support multi-title batching, but each call is cheap
    (no text content, just titles) so this stays fast."""
    out = {}
    for cat in cat_titles:
        data = _api_get({
            "action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
            "cmtype": "page", "cmlimit": MAX_ARTICLES_PER_CATEGORY, "format": "json",
        })
        if data is not None:
            out[cat] = [m["title"] for m in data["query"]["categorymembers"]]
        else:
            out[cat] = []
        time.sleep(delay)
    return out


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "wiki_category_graph.json"), encoding="utf-8") as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    print(f"Fetching content for {len(nodes)} categories (batched)...")

    print("Step 2a: category summaries (batched, ~1 request per 50 categories)...")
    cat_pseudo_titles = [f"Category:{c}" for c in nodes]
    cat_summaries_raw = batch_get_summaries(cat_pseudo_titles)
    cat_summaries = {c: cat_summaries_raw.get(f"Category:{c}", "") for c in nodes}
    # fallback: article of the same name, for categories with no category-page text
    missing = [c for c in nodes if not cat_summaries[c]]
    if missing:
        print(f"  {len(missing)} categories had no category-page text, trying article fallback...")
        fallback = batch_get_summaries(missing)
        for c in missing:
            cat_summaries[c] = fallback.get(c, "")

    # Per-category article lists (Step 2b/2c) were dropped: Wikipedia's
    # categorymembers endpoint can't batch across categories, so it required
    # 416 separate slow requests (observed ~20-30s each -- a Wikipedia-side
    # throttle on this account/IP, not something client-side batching can
    # fix) -- projected 2+ hours for what was meant to be a quick step.
    # Category summaries alone (real Wikipedia text, already fetched above,
    # batched) are sufficient content for the demo.
    content = {cat: {"summary": cat_summaries.get(cat, ""), "articles": {}} for cat in nodes}

    out_path = os.path.join(DATA_DIR, "wiki_content.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=1)
    print(f"Saved {out_path}")
