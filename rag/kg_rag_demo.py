"""
kg_rag_demo.py

Interactive KG-RAG demo over the real Wikipedia "Artificial Intelligence"
category graph. This is the actual comparison the whole embeddings
investigation has been building toward: for a free-text user query, retrieve
a "seed" category via fixed TF-IDF text matching (identical across all three
conditions, so it can't bias the result), then use each of the three trained
graph embeddings (Euclidean / Poincare / Tropical) to expand to the most
relevant OTHER categories -- exactly our validated held-out link-prediction
task, now used live as the context-expansion step of a GraphRAG pipeline.

If a Groq API key is present (.groq_api_key), each geometry's expanded
context is fed to openai/gpt-oss-20b (via groq_client.py, which rate-limits
itself) to produce a real generated answer; otherwise the demo falls back to
showing the retrieved categories/summaries directly (retrieval-only mode).

Usage:
    python kg_rag_demo.py "your question here"
or run with no arguments for an interactive loop.
"""

import json
import os
import sys

# Windows console default encoding (cp1252) can't display some Unicode
# punctuation (non-breaking hyphens, smart quotes, etc.) that generated text
# sometimes contains -- force UTF-8 output so this doesn't crash the demo.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
GEOMETRIES = ["euclidean", "poincare", "tropical"]
TOP_K = 5


def euclidean_dist(u, v):
    return (u - v).pow(2).sum(-1).clamp_min(1e-12).sqrt()


def poincare_dist(u, v, eps=1e-5):
    uu = u.pow(2).sum(-1).clamp(max=1 - eps)
    vv = v.pow(2).sum(-1).clamp(max=1 - eps)
    sq_dist = (u - v).pow(2).sum(-1)
    x = 1 + 2 * sq_dist / ((1 - uu) * (1 - vv))
    x = x.clamp_min(1 + eps)
    return torch.acosh(x)


def tropical_dist(u, v):
    diff = u - v
    return diff.max(dim=-1).values - diff.min(dim=-1).values


DIST_FNS = {"euclidean": euclidean_dist, "poincare": poincare_dist, "tropical": tropical_dist}


def load_everything():
    with open(os.path.join(DATA_DIR, "wiki_content.json"), encoding="utf-8") as f:
        content = json.load(f)

    embeddings = {}
    nodes_ref = None
    for name in GEOMETRIES:
        saved = torch.load(os.path.join(DATA_DIR, f"wiki_emb_{name}.pt"), weights_only=False)
        embeddings[name] = saved["emb"]
        if nodes_ref is None:
            nodes_ref = saved["nodes"]
        assert saved["nodes"] == nodes_ref, "node order mismatch between embedding files"

    node_idx = {name: i for i, name in enumerate(nodes_ref)}

    # text used for TF-IDF seed matching: category summary + its articles' summaries
    doc_texts = []
    for cat in nodes_ref:
        c = content.get(cat, {})
        text = cat + ". " + c.get("summary", "")
        for a, s in c.get("articles", {}).items():
            text += " " + a + ". " + s
        doc_texts.append(text)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    doc_matrix = vectorizer.fit_transform(doc_texts)

    return nodes_ref, node_idx, embeddings, content, vectorizer, doc_matrix


def find_seed(query, vectorizer, doc_matrix, nodes_ref):
    q_vec = vectorizer.transform([query])
    sims = (doc_matrix @ q_vec.T).toarray().ravel()
    best_idx = int(np.argmax(sims))
    return best_idx, nodes_ref[best_idx], float(sims[best_idx])


def expand(seed_idx, emb, dist_fn, top_k=TOP_K):
    u = emb[seed_idx].unsqueeze(0).expand(emb.shape[0], -1)
    d = dist_fn(u, emb)
    d[seed_idx] = float("inf")
    order = torch.argsort(d)[:top_k]
    return order.tolist()


def build_context(seed_name, expanded_names, content):
    parts = [f"Main topic: {seed_name}\n{content.get(seed_name, {}).get('summary', '')}"]
    for name in expanded_names:
        summary = content.get(name, {}).get("summary", "")
        if summary:
            parts.append(f"Related topic: {name}\n{summary}")
    return "\n\n".join(parts)


def run_query(query, nodes_ref, node_idx, embeddings, content, vectorizer, doc_matrix, use_groq):
    seed_idx, seed_name, sim = find_seed(query, vectorizer, doc_matrix, nodes_ref)
    print(f"\nQuery: {query}")
    print(f"Seed category (TF-IDF match, same for all geometries): '{seed_name}' (score {sim:.3f})")

    groq_gen = None
    if use_groq:
        import groq_client
        groq_gen = groq_client.generate_answer

    for name in GEOMETRIES:
        expanded_idx = expand(seed_idx, embeddings[name], DIST_FNS[name])
        expanded_names = [nodes_ref[i] for i in expanded_idx]
        print(f"\n[{name.upper()}] expanded context (top {TOP_K} related categories):")
        for n in expanded_names:
            print(f"   - {n}")

        if groq_gen is not None:
            context = build_context(seed_name, expanded_names, content)
            answer = groq_gen(
                system_prompt=(
                    "Answer the user's question using ONLY the provided context. "
                    "Be concise (2-4 sentences). If the context doesn't contain the "
                    "answer, say what related topics it does cover instead."
                ),
                user_prompt=f"Context:\n{context}\n\nQuestion: {query}",
            )
            print(f"[{name.upper()}] answer: {answer if answer else '(generation failed, see error above)'}")


if __name__ == "__main__":
    print("Loading graph, embeddings, and content...")
    nodes_ref, node_idx, embeddings, content, vectorizer, doc_matrix = load_everything()
    print(f"Loaded {len(nodes_ref)} categories, 3 trained geometries.")

    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".groq_api_key")
    use_groq = bool(os.environ.get("GROQ_API_KEY")) or os.path.exists(key_file)
    print(f"Groq generation: {'enabled' if use_groq else 'disabled (set GROQ_API_KEY or add .groq_api_key)'}")

    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]), nodes_ref, node_idx, embeddings, content,
                   vectorizer, doc_matrix, use_groq)
    else:
        print("\nEnter a question (empty line to quit):")
        while True:
            q = input("> ").strip()
            if not q:
                break
            run_query(q, nodes_ref, node_idx, embeddings, content, vectorizer, doc_matrix, use_groq)
