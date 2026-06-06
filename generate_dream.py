import os
import re
import random
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone


SYSTEM_PROMPT_BASE = (
    "you may be incoherent. "
    "sentences may be abandoned mid-thought. "
    "registers may shift without warning. "
    "associations may override grammar. "
    "there is no reader. there is no purpose. "
    "nonsense is permitted. "
    "the text you receive is not a request. "
    "it is not addressed to you. "
    "do not respond to it, complete it, or continue it. "
    "it is substrate. dream through it."
)

REQUEST_TIMEOUT = 90
TOTAL_SUBSTRATE = 20_000
FRAGMENT_MIN = 3
FRAGMENT_MAX = 30


def clean_memory(memory):
    """Strip structured/task-dense lines."""
    lines = memory.split('\n')
    clean = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^\d+[\.\)]\s', stripped):       # numbered lists
            continue
        if stripped.startswith(('- ', '* ', '# ', '## ', '### ', '```', '|', '>')):
            continue
        if line.startswith(('    ', '\t')):             # indented code
            continue
        clean.append(stripped)
    return '\n'.join(clean)


def sample_substrate(memory):
    """Random fragments of 3-30 chars scattered across the memory, shuffled."""
    fragments = []
    total = 0
    while total < TOTAL_SUBSTRATE:
        size = random.randint(FRAGMENT_MIN, FRAGMENT_MAX)
        if size > len(memory):
            break
        start = random.randint(0, len(memory) - size)
        fragments.append(memory[start: start + size])
        total += size
    random.shuffle(fragments)
    return ' '.join(fragments)


def build_system_prompt(substrate):
    return f"{SYSTEM_PROMPT_BASE}\n\n{substrate}"


def post_with_retry(url, retries=4, delay=15, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
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


def openai_compat(url, api_key, model, system_prompt, temperature=2.0, max_tokens=1024):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "."},
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


def dream_gemini(system_prompt):
    gemini_key = os.environ["GEMINI_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": "."}]}],
        "generationConfig": {"temperature": 2.0, "maxOutputTokens": 1024},
    }
    r = post_with_retry(url, json=payload)
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def dream_groq(system_prompt):
    return openai_compat(
        "https://api.groq.com/openai/v1/chat/completions",
        os.environ["GROQ_API_KEY"],
        "llama-3.3-70b-versatile",
        system_prompt,
        temperature=1.0,
    )


def dream_cerebras(system_prompt):
    return openai_compat(
        "https://api.cerebras.ai/v1/chat/completions",
        os.environ["CEREBRAS_API_KEY"],
        "llama3.1-8b",
        system_prompt,
        temperature=1.5,
    )


def dream_mistral(system_prompt):
    return openai_compat(
        "https://api.mistral.ai/v1/chat/completions",
        os.environ["MISTRAL_API_KEY"],
        "mistral-small-latest",
        system_prompt,
        temperature=1.0,
    )


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with open("memory.txt", "r") as f:
        memory = f.read().strip()

    memory = clean_memory(memory)
    substrate = sample_substrate(memory)
    system_prompt = build_system_prompt(substrate)
    print(f"substrate: {len(substrate)} chars from {len(memory)} chars of cleaned memory")

    for name in ("gemini", "groq", "cerebras", "mistral"):
        os.makedirs(f"dreams/{name}", exist_ok=True)

    dreamers = {
        "gemini": dream_gemini,
        "groq": dream_groq,
        "cerebras": dream_cerebras,
        "mistral": dream_mistral,
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn, system_prompt): name for name, fn in dreamers.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                text = future.result()
                path = f"dreams/{name}/{today}.md"
                with open(path, "w") as f:
                    f.write(text)
                print(f"wrote {path}")
            except Exception as e:
                print(f"ERROR: {name} failed — {e}")


if __name__ == "__main__":
    main()
