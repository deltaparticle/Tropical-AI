"""
webapp.py

Local, interactive comparison website for the KG-RAG demo. Type a question,
see three side-by-side answers -- one per geometry (Euclidean, Poincare,
Tropical) -- built from the same TF-IDF seed match but each geometry's own
graph-expansion step, then generated via Groq's gpt-oss-20b.

Run: python webapp.py, then open http://127.0.0.1:5000 in a browser.
Local only (no external hosting) -- matches the CPU-only/local approach
used throughout this project.
"""

import os
from flask import Flask, request, jsonify, render_template_string

from kg_rag_demo import load_everything, find_seed, expand, build_context, DIST_FNS, GEOMETRIES, TOP_K
import groq_client

app = Flask(__name__)

print("Loading graph, embeddings, and content (once, at startup)...")
NODES, NODE_IDX, EMBEDDINGS, CONTENT, VECTORIZER, DOC_MATRIX = load_everything()
_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".groq_api_key")
USE_GROQ = bool(os.environ.get("GROQ_API_KEY")) or os.path.exists(_KEY_FILE)
print(f"Loaded {len(NODES)} categories. Groq generation: {'enabled' if USE_GROQ else 'disabled'}")

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tropical vs Poincare vs Euclidean: KG-RAG Comparison</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1200px; margin: 30px auto; padding: 0 16px; background: #fafafa; color: #222; }
  h1 { font-size: 1.4em; }
  .sub { color: #666; margin-bottom: 20px; }
  #qform { display: flex; gap: 8px; margin-bottom: 24px; }
  #q { flex: 1; padding: 10px 12px; font-size: 1em; border: 1px solid #ccc; border-radius: 6px; }
  button { padding: 10px 20px; font-size: 1em; border: none; border-radius: 6px; background: #2b6cb0; color: white; cursor: pointer; }
  button:disabled { background: #999; cursor: not-allowed; }
  .cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 120px; }
  .card h3 { margin-top: 0; text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.05em; }
  .euclidean h3 { color: #888; }
  .poincare h3 { color: #4477aa; }
  .tropical h3 { color: #cc6677; }
  .expanded { font-size: 0.85em; color: #555; margin-bottom: 10px; }
  .expanded ul { margin: 4px 0; padding-left: 18px; }
  .answer { font-size: 0.95em; line-height: 1.4; }
  .seed { color: #444; margin-bottom: 14px; font-size: 0.9em; }
  .loading { color: #999; font-style: italic; }
</style>
</head>
<body>
<h1>Tropical vs Poincare vs Euclidean &mdash; live KG-RAG comparison</h1>
<div class="sub">Real Wikipedia "Artificial Intelligence" category graph ({{n_nodes}} categories).
Same TF-IDF seed match across all three; each geometry expands context differently.</div>

<form id="qform">
  <input id="q" type="text" placeholder="Ask a question about AI/ML..." autocomplete="off">
  <button id="btn" type="submit">Ask</button>
</form>

<div id="seedinfo" class="seed"></div>

<div class="cols">
  <div class="card euclidean"><h3>Euclidean</h3><div id="euclidean-body"><span class="loading">Ask a question to see results.</span></div></div>
  <div class="card poincare"><h3>Poincare</h3><div id="poincare-body"><span class="loading"></span></div></div>
  <div class="card tropical"><h3>Tropical</h3><div id="tropical-body"><span class="loading"></span></div></div>
</div>

<script>
const form = document.getElementById('qform');
const btn = document.getElementById('btn');
const seedinfo = document.getElementById('seedinfo');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  btn.disabled = true;
  btn.textContent = 'Thinking...';
  seedinfo.textContent = '';
  ['euclidean', 'poincare', 'tropical'].forEach(g => {
    document.getElementById(g + '-body').innerHTML = '<span class="loading">Retrieving + generating (this can take 10-20s per model due to free-tier rate limits)...</span>';
  });

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q}),
    });
    const data = await res.json();
    if (data.error) {
      seedinfo.textContent = 'Error: ' + data.error;
    } else {
      seedinfo.textContent = `Seed category (same for all three): "${data.seed_name}" (TF-IDF score ${data.seed_score.toFixed(3)})`;
      for (const g of ['euclidean', 'poincare', 'tropical']) {
        const r = data.results[g];
        const list = r.expanded.map(n => `<li>${n}</li>`).join('');
        document.getElementById(g + '-body').innerHTML =
          `<div class="expanded"><b>Expanded context:</b><ul>${list}</ul></div>` +
          `<div class="answer">${r.answer}</div>`;
      }
    }
  } catch (err) {
    seedinfo.textContent = 'Request failed: ' + err;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Ask';
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, n_nodes=len(NODES))


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    seed_idx, seed_name, sim = find_seed(question, VECTORIZER, DOC_MATRIX, NODES)

    results = {}
    for name in GEOMETRIES:
        expanded_idx = expand(seed_idx, EMBEDDINGS[name], DIST_FNS[name], top_k=TOP_K)
        expanded_names = [NODES[i] for i in expanded_idx]

        answer = "(Groq generation disabled -- set GROQ_API_KEY or add a .groq_api_key file)"
        if USE_GROQ:
            context = build_context(seed_name, expanded_names, CONTENT)
            gen = groq_client.generate_answer(
                system_prompt=(
                    "Answer the user's question using ONLY the provided context. "
                    "Be concise (2-4 sentences). If the context doesn't contain the "
                    "answer, say what related topics it does cover instead."
                ),
                user_prompt=f"Context:\n{context}\n\nQuestion: {question}",
            )
            answer = gen or "(generation failed -- see server console for the error)"

        results[name] = {"expanded": expanded_names, "answer": answer}

    return jsonify({
        "seed_name": seed_name,
        "seed_score": sim,
        "results": results,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
