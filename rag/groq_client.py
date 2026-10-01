"""
groq_client.py

Minimal, rate-limited client for Groq's OpenAI-compatible chat completions
API, used for the generation step of the KG-RAG demo (openai/gpt-oss-20b).

API key resolution order: the GROQ_API_KEY environment variable, falling
back to a .groq_api_key file in this directory -- never hardcode the key in
a script, never print it, and don't commit/share that file.

Rate limiting: a conservative minimum delay between requests plus
exponential backoff that respects a 429 response's Retry-After header when
present, since this is a free-tier key.
"""

import json
import os
import time
import urllib.request
import urllib.error

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
MIN_SECONDS_BETWEEN_CALLS = 2.5  # conservative client-side throttle

_last_call_time = [0.0]


def _load_key(path=None):
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key.strip()
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".groq_api_key")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


_API_KEY = None


def generate_answer(system_prompt, user_prompt, max_tokens=700, temperature=0.3, retries=4,
                     model=MODEL):
    """Call a Groq-hosted model (default: the gpt-oss-20b generator). Returns
    the response text, or None if the call ultimately failed (caller should
    fall back gracefully, e.g. to a retrieval-only answer)."""
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = _load_key()

    # client-side throttle: never call more than once per MIN_SECONDS_BETWEEN_CALLS
    elapsed = time.time() - _last_call_time[0]
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            _last_call_time[0] = time.time()
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            _last_call_time[0] = time.time()
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (2 ** attempt) * 3
                print(f"    [groq] rate limited, waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            print(f"    [groq] HTTP error {e.code}: {e.read()[:300]}")
            return None
        except Exception as e:
            print(f"    [groq] request failed: {e}")
            time.sleep(2 ** attempt)
    return None


if __name__ == "__main__":
    # quick smoke test -- uses ONE real API call
    out = generate_answer(
        "You are a concise assistant.",
        "Say 'ok' if you can read this.",
        max_tokens=10,
    )
    print("Response:", out)
