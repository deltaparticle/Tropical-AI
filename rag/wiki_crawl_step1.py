"""Step 1: crawl the Wikipedia 'Artificial intelligence' category graph and
save nodes/edges to a JSON file for the next pipeline stage."""

import json
import os
from wiki_kg_rag import fetch_category_graph

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ROOT = "Artificial intelligence"

if __name__ == "__main__":
    nodes, edges = fetch_category_graph(ROOT, max_depth=3, request_delay=0.3)
    print(f"FINAL: {len(nodes)} categories, {len(edges)} edges")
    out_path = os.path.join(DATA_DIR, "wiki_category_graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"root": ROOT, "nodes": nodes, "edges": edges}, f, indent=1)
    print(f"Saved {out_path}")
