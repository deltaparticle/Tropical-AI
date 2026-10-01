"""
compute_touched_categories.py

Computes the set of Wikipedia categories actually touched (as a TF-IDF seed
match or a graph-expansion result, across all three geometries) when running
the fixed 15-question eval set through the live KG-RAG pipeline. Used to
scope content enrichment (wiki_enrich_touched.py) to only the categories
that matter for this evaluation, instead of re-crawling all 416 categories.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

from kg_rag_demo import load_everything, find_seed, expand, DIST_FNS, GEOMETRIES, TOP_K
from eval_questions import QA_PAIRS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

if __name__ == "__main__":
    nodes, node_idx, emb, content, vec, mat = load_everything()

    touched = set()
    for qa in QA_PAIRS:
        seed_idx, seed_name, sim = find_seed(qa["question"], vec, mat, nodes)
        touched.add(seed_name)
        for g in GEOMETRIES:
            idx = expand(seed_idx, emb[g], DIST_FNS[g], top_k=TOP_K)
            for i in idx:
                touched.add(nodes[i])

    print(f"Touched categories: {len(touched)}")
    out_path = os.path.join(DATA_DIR, "touched_categories.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted(touched), f, indent=1)
    print(f"Saved {out_path}")
