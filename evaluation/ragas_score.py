"""
ragas_score.py

Scores ragas_eval_data.json (produced by kg_rag_export_for_ragas.py) with
RAGAS, using a judge model served via OpenRouter.

Judge: meta-llama/llama-3.3-70b-instruct -- substantially larger than the
openai/gpt-oss-20b model used to generate the answers being judged, which
reduces self/weak-evaluation bias in the scores.

Runs INSIDE the isolated ragas_venv/ (torch/numpy version pins there differ
from the main project environment).
"""

import json
import os

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness, answer_relevancy, answer_correctness, answer_similarity,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Context Precision and Context Recall are not included: both judge every
# retrieved context individually (up to 6 per record), multiplying judge-LLM
# call volume several-fold for comparatively little extra signal over the
# 4 metrics below.
RUN_CONFIG = RunConfig(timeout=180, max_retries=6, max_wait=60, max_workers=3)

# OpenRouter rejects n>1 (multiple sampled completions per call); RAGAS's
# answer_relevancy defaults to strictness=3 (n=3) -- drop to strictness=1.
answer_relevancy.strictness = 1

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"


def _load_key():
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key.strip()
    with open(os.path.join(HERE, ".openrouter_api_key"), encoding="utf-8") as f:
        return f.read().strip()


API_KEY = _load_key()
os.environ["OPENAI_API_KEY"] = API_KEY  # some ragas internals read this directly

judge_llm = ChatOpenAI(
    model=JUDGE_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=API_KEY,
    temperature=0.0,
    max_tokens=1024,
    default_headers={
        "HTTP-Referer": "https://github.com/deltaparticle/Tropical-AI",
        "X-Title": "tropical-kg-rag-eval",
    },
)

judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

METRICS = [faithfulness, answer_relevancy, answer_correctness, answer_similarity]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "answer_correctness", "answer_similarity"]


def score_geometry(name, records):
    ds = Dataset.from_list(records)
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        run_config=RUN_CONFIG,
    )
    return result.to_pandas()


if __name__ == "__main__":
    with open(os.path.join(HERE, "ragas_eval_data.json"), encoding="utf-8") as f:
        data = json.load(f)

    all_results = {}
    all_dfs = {}
    for name, records in data.items():
        print(f"\nScoring {name} ({len(records)} records) with judge={JUDGE_MODEL}...")
        df = score_geometry(name, records)
        all_dfs[name] = df
        present_metrics = [m for m in METRIC_NAMES if m in df.columns]
        print(df[["question"] + present_metrics].to_string())
        all_results[name] = {f"mean_{m}": float(df[m].mean()) for m in present_metrics}
        valid_counts = {m: int(df[m].notna().sum()) for m in present_metrics}
        all_results[name]["valid_counts"] = valid_counts

    print("\n=== RAGAS Summary (mean across all questions) ===")
    header = f"{'geometry':>12} | " + " | ".join(f"{m:>18}" for m in METRIC_NAMES)
    print(header)
    for name, r in all_results.items():
        row = f"{name:>12} | " + " | ".join(f"{r.get('mean_'+m, float('nan')):>18.3f}" for m in METRIC_NAMES)
        print(row)

    with open(os.path.join(HERE, "ragas_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=1)

    for name, df in all_dfs.items():
        df.to_csv(os.path.join(HERE, f"ragas_per_question_{name}.csv"), index=False)

    print("\nSaved ragas_results.json and per-question CSVs")
