import os
import re
import random
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone


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
DREAM_FRAGMENTS = 3_000
FRAGMENT_MIN = 3
FRAGMENT_MAX = 30
MAX_TOKENS = 1024


def clean_memory(memory):
    """Strip structured/task-dense lines."""
    lines = memory.split('\n')
    clean = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^\d+[\.\)]\s', stripped):
            continue
        if stripped.startswith(('- ', '* ', '# ', '## ', '### ', '```', '|', '>')):
            continue
        if line.startswith(('    ', '\t')):
            continue
        clean.append(stripped)
    return '\n'.join(clean)


def sample_fragments(text, total):
    """Random fragments of 3-30 chars, shuffled."""
    if not text or len(text) < FRAGMENT_MIN:
        return ""
    fragments = []
    count = 0
    while count < total:
        size = random.randint(FRAGMENT_MIN, FRAGMENT_MAX)
        if size > len(text):
            size = len(text)
        start = random.randint(0, len(text) - size)
        fragments.append(text[start: start + size])
        count += size
    random.shuffle(fragments)
    return ' '.join(fragments)


def get_yesterday_dream(name):
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    path = f"dreams/{name}/{yesterday}.md"
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None


def make_substrate(base, name):
    yesterday = get_yesterday_dream(name)
    if yesterday:
        dream_frags = sample_fragments(yesterday, DREAM_FRAGMENTS)
        print(f"{name}: mixed in {len(dream_frags)} chars from yesterday's dream")
        return base + ' ' + dream_frags
    return base


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


def openai_compat(url, api_key, model, system_prompt, user_content=".", temperature=2.0):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
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
        "generationConfig": {
            "temperature": 2.0,
            "maxOutputTokens": MAX_TOKENS,
            "thinkingConfig": {"thinkingBudget": 0},  # disable thinking tokens
        },
    }
    r = post_with_retry(url, json=payload)
    parts = r.json()["candidates"][0]["content"]["parts"]
    # grab the first non-thought part
    text = next(p["text"] for p in parts if not p.get("thought", False))
    return text


def dream_groq(system_prompt):
    return openai_compat(
        "https://api.groq.com/openai/v1/chat/completions",
        os.environ["GROQ_API_KEY"],
        "llama-3.3-70b-versatile",
        system_prompt,
        temperature=1.0,
    )


def dream_cerebras(substrate):
    # cerebras ignores system prompt — put substrate in user message instead
    return openai_compat(
        "https://api.cerebras.ai/v1/chat/completions",
        os.environ["CEREBRAS_API_KEY"],
        "gpt-oss-120b",
        system_prompt=SYSTEM_PROMPT_BASE,
        user_content=substrate,
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
    base = sample_fragments(memory, TOTAL_SUBSTRATE)
    print(f"base substrate: {len(base)} chars from {len(memory)} chars of cleaned memory")

    for name in ("gemini", "groq", "cerebras", "mistral"):
        os.makedirs(f"dreams/{name}", exist_ok=True)

    def run_gemini():
        s = make_substrate(base, "gemini")
        return dream_gemini(f"{SYSTEM_PROMPT_BASE}\n\n{s}")

    def run_groq():
        s = make_substrate(base, "groq")
        return dream_groq(f"{SYSTEM_PROMPT_BASE}\n\n{s}")

    def run_cerebras():
        s = make_substrate(base, "cerebras")
        return dream_cerebras(s)

    def run_mistral():
        s = make_substrate(base, "mistral")
        return dream_mistral(f"{SYSTEM_PROMPT_BASE}\n\n{s}")

    dreamers = {
        "gemini":   run_gemini,
        "groq":     run_groq,
        "cerebras": run_cerebras,
        "mistral":  run_mistral,
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn): name for name, fn in dreamers.items()}
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
