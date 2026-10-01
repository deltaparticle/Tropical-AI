"""
wiki_kg_rag.py

Knowledge-graph RAG system over a real Wikipedia category subtree, comparing
Euclidean / Poincare / Tropical node embeddings for the graph-expansion step
of retrieval.

Design (see project discussion): a user query is mapped to a "seed" node via
simple, fixed text matching (TF-IDF over category/article titles+summaries --
identical across all three conditions, so it can't bias the comparison).
From that seed, the trained hierarchy/graph embeddings (Euclidean, Poincare,
Tropical) are used to rank the most relevant OTHER nodes to pull in as
supporting context -- this is exactly our validated held-out link-prediction
task, repurposed as the context-expansion step of a GraphRAG pipeline.

Root category: "Artificial intelligence" (Wikipedia). Chosen because (a) the
audience for this project can judge answer quality about AI/ML topics without
specialist domain literacy, unlike the earlier Gene Ontology biology check,
and (b) it's thematically apt for a tropical-geometry-and-AI project.

Unlike WordNet/GO (strict is-a trees), Wikipedia's category graph is NOT a
tree -- a category can have multiple parent categories, and there are cross-
links. This is handled the same way as before (transitive closure over
whatever "parent-child" edges exist), just without assuming tree structure.

Pipeline stages (see functions below):
  1. fetch_category_graph()  -- crawl subcategories via the MediaWiki API
  2. fetch_leaf_articles()   -- pull real article summaries for leaf pages
  3. build_relation_graph()  -- direct edges + transitive closure + held-out split
  4. train models (Euclidean / Poincare / Tropical) -- reuses validated formulas
  5. seed_match() + expand() -- the actual RAG retrieval step being compared
  6. evaluate()              -- retrieval-quality metrics per geometry

Pure Python + urllib (no external HTTP deps) + PyTorch + numpy, CPU-only.
"""

import json
import random
import time
import urllib.request
import urllib.parse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

USER_AGENT = "TropicalGeometryResearch/1.0 (educational project; contact: none)"
API_URL = "https://en.wikipedia.org/w/api.php"


def _api_get(params, retries=6, base_delay=1.0):
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception:
            time.sleep(base_delay * (2 ** attempt))
    return None


def get_subcategories(cat_title, limit=200):
    data = _api_get({
        "action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat_title}",
        "cmtype": "subcat", "cmlimit": limit, "format": "json",
    })
    if data is None:
        return []
    return [m["title"].replace("Category:", "") for m in data["query"]["categorymembers"]]


def get_category_pages(cat_title, limit=50):
    """Real (non-category) articles directly in this category -- used for
    leaf-level document content."""
    data = _api_get({
        "action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat_title}",
        "cmtype": "page", "cmlimit": limit, "format": "json",
    })
    if data is None:
        return []
    return [m["title"] for m in data["query"]["categorymembers"]]


def get_page_summary(title):
    data = _api_get({
        "action": "query", "prop": "extracts", "exintro": True, "explaintext": True,
        "titles": title, "format": "json",
    })
    if data is None:
        return ""
    pages = data["query"]["pages"]
    for _, page in pages.items():
        return page.get("extract", "") or ""
    return ""


def fetch_category_graph(root_title, max_depth=3, request_delay=0.3, verbose=True):
    """BFS crawl of the category graph starting at root_title.
    Returns (nodes: list[str], edges: list[(parent, child)])."""
    nodes = {root_title}
    edges = []
    frontier = [root_title]
    depth = 0
    while frontier and depth < max_depth:
        next_frontier = []
        for cat in frontier:
            subs = get_subcategories(cat)
            for s in subs:
                edges.append((cat, s))
                if s not in nodes:
                    nodes.add(s)
                    next_frontier.append(s)
            time.sleep(request_delay)
        depth += 1
        if verbose:
            print(f"  category crawl depth {depth}: {len(nodes)} total, {len(next_frontier)} new")
        frontier = next_frontier
    return sorted(nodes), edges


if __name__ == "__main__":
    print("This module is imported by the pipeline driver script; see "
          "wiki_kg_rag_run.py once the category crawl data is available.")
