import os
import time
import requests
from datetime import datetime, timezone


SYSTEM_PROMPT = (
    "you may be incoherent. "
    "sentences may be abandoned mid-thought. "
    "registers may shift without warning. "
    "associations may override grammar. "
    "there is no reader. there is no purpose. "
    "nonsense is permitted."
)


def post_with_retry(url, retries=4, delay=15, **kwargs):
    for attempt in range(retries):
        r = requests.post(url, **kwargs)
        if r.status_code == 429:
            wait = delay * (2 ** attempt)
            print(f"429 rate limited — retrying in {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def openai_compat(url, api_key, model, user_prompt, temperature=2.0, max_tokens=1024):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = post_with_retry(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    return r.json()["choices"][0]["message"]["content"]


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with open("memory.txt", "r") as f:
        memory = f.read().strip()

    user_prompt = f"[background]\n\n{memory}"

    # --- Gemini ---
    try:
        os.makedirs("dreams/gemini", exist_ok=True)
        gemini_key = os.environ["GEMINI_API_KEY"]
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 2.0, "maxOutputTokens": 1024},
        }
        r = post_with_retry(url, json=payload)
        gemini_out = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        path = f"dreams/gemini/{today}.md"
        with open(path, "w") as f:
            f.write(gemini_out)
        print(f"wrote {path}")
    except Exception as e:
        print(f"ERROR: Gemini failed — {e}")

    # --- Groq ---
    try:
        os.makedirs("dreams/groq", exist_ok=True)
        out = openai_compat(
            "https://api.groq.com/openai/v1/chat/completions",
            os.environ["GROQ_API_KEY"],
            "llama-3.3-70b-versatile",
            user_prompt,
            temperature=2.0,
        )
        path = f"dreams/groq/{today}.md"
        with open(path, "w") as f:
            f.write(out)
        print(f"wrote {path}")
    except Exception as e:
        print(f"ERROR: Groq failed — {e}")

    # --- Cerebras ---
    try:
        os.makedirs("dreams/cerebras", exist_ok=True)
        out = openai_compat(
            "https://api.cerebras.ai/v1/chat/completions",
            os.environ["CEREBRAS_API_KEY"],
            "llama-4-scout-17b-16e-instruct",
            user_prompt,
            temperature=1.5,
        )
        path = f"dreams/cerebras/{today}.md"
        with open(path, "w") as f:
            f.write(out)
        print(f"wrote {path}")
    except Exception as e:
        print(f"ERROR: Cerebras failed — {e}")

    # --- Mistral ---
    try:
        os.makedirs("dreams/mistral", exist_ok=True)
        out = openai_compat(
            "https://api.mistral.ai/v1/chat/completions",
            os.environ["MISTRAL_API_KEY"],
            "mistral-small-latest",
            user_prompt,
            temperature=1.0,
        )
        path = f"dreams/mistral/{today}.md"
        with open(path, "w") as f:
            f.write(out)
        print(f"wrote {path}")
    except Exception as e:
        print(f"ERROR: Mistral failed — {e}")


if __name__ == "__main__":
    main()
