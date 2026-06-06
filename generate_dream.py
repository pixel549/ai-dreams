import os
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


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with open("memory.txt", "r") as f:
        memory = f.read().strip()

    user_prompt = f"[background]\n\n{memory}"

    os.makedirs("dreams/gemini", exist_ok=True)
    os.makedirs("dreams/grok", exist_ok=True)

    # --- Gemini ---
    gemini_key = os.environ["GEMINI_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 2.0, "maxOutputTokens": 1024},
    }
    r = requests.post(url, json=payload)
    r.raise_for_status()
    gemini_out = r.json()["candidates"][0]["content"]["parts"][0]["text"]

    path = f"dreams/gemini/{today}.md"
    with open(path, "w") as f:
        f.write(gemini_out)
    print(f"wrote {path}")

    # --- Grok ---
    grok_key = os.environ["GROK_API_KEY"]
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 2.0,
        "max_tokens": 1024,
    }
    r = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {grok_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    r.raise_for_status()
    grok_out = r.json()["choices"][0]["message"]["content"]

    path = f"dreams/grok/{today}.md"
    with open(path, "w") as f:
        f.write(grok_out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
