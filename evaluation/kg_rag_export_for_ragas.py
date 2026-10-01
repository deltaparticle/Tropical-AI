"""
kg_rag_export_for_ragas.py

Runs the KG-RAG pipeline (seed match + per-geometry graph expansion +
Groq gpt-oss-20b generation) on a fixed question set and exports the
structured {question, contexts, answer} records RAGAS needs, per geometry,
to a JSON file. This runs in the MAIN environment (needs torch/sklearn);
the actual RAGAS scoring runs separately in the isolated ragas_venv, which
only needs to read this JSON, not our modeling code.
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

import json
from kg_rag_demo import load_everything, find_seed, expand, build_context, DIST_FNS, GEOMETRIES, TOP_K
import groq_client
from eval_questions import QA_PAIRS


def get_contexts(seed_name, expanded_names, content):
    """Return a list of individual context strings (one per retrieved
    category's summary) -- RAGAS wants contexts as a list, not one blob."""
    ctxs = []
    seed_summary = content.get(seed_name, {}).get("summary", "")
    if seed_summary:
        ctxs.append(f"{seed_name}: {seed_summary}")
    for name in expanded_names:
        s = content.get(name, {}).get("summary", "")
        if s:
            ctxs.append(f"{name}: {s}")
    if not ctxs:
        ctxs.append(f"{seed_name} (no description text available)")
    return ctxs


if __name__ == "__main__":
    print("Loading graph, embeddings, and content...")
    nodes_ref, node_idx, embeddings, content, vectorizer, doc_matrix = load_everything()
    print(f"Loaded {len(nodes_ref)} categories.\n")

    records = {name: [] for name in GEOMETRIES}

    for qi, qa in enumerate(QA_PAIRS, 1):
        q = qa["question"]
        ground_truth = qa["ground_truth"]
        seed_idx, seed_name, sim = find_seed(q, vectorizer, doc_matrix, nodes_ref)
        print(f"[{qi}/{len(QA_PAIRS)}] '{q}' -> seed '{seed_name}' ({sim:.3f})")

        for name in GEOMETRIES:
            expanded_idx = expand(seed_idx, embeddings[name], DIST_FNS[name], top_k=TOP_K)
            expanded_names = [nodes_ref[i] for i in expanded_idx]
            contexts = get_contexts(seed_name, expanded_names, content)
            context_blob = "\n\n".join(contexts)

            answer = groq_client.generate_answer(
                system_prompt=(
                    "Answer the user's question using ONLY the provided context. "
                    "Be concise (2-4 sentences). If the context doesn't contain the "
                    "answer, say what related topics it does cover instead."
                ),
                user_prompt=f"Context:\n{context_blob}\n\nQuestion: {q}",
            )
            answer = answer or "(generation failed)"
            print(f"    [{name}] seed+{len(expanded_names)} expanded -> answer len {len(answer)}")

            records[name].append({
                "question": q,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": ground_truth,
            })

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_eval_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)
    print(f"\nSaved {out_path}")
