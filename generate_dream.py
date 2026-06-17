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
DAY_PAUSE = 30  # seconds between days to avoid rate limits


def clean_memory(memory):
    """Strip structured/task-dense lines."""
    lines = memory.split('\n')
    clean = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^\d+[.)\]]\s', stripped):
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


def pick_trigger(substrate):
    """Random non-trivial fragment from substrate as trigger."""
    for _ in range(20):  # try up to 20 times to find a non-trivial fragment
        size = random.randint(FRAGMENT_MIN, FRAGMENT_MAX)
        start = random.randint(0, max(0, len(substrate) - size))
        fragment = substrate[start: start + size].strip()
        if len(fragment) >= 3 and not all(c in '.!?, ' for c in fragment):
            return fragment
    return substrate[:20]  # fallback


def get_dream(name, date_str):
    path = f"dreams/{name}/{date_str}.md"
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None


def make_substrate(base, name, yesterday_str):
    yesterday = get_dream(name, yesterday_str)
    if yesterday:
        dream_frags = sample_fragments(yesterday, DREAM_FRAGMENTS)
        print(f"{name}: mixed in {len(dream_frags)} chars from {yesterday_str}'s dream")
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


def openai_compat(url, api_key, model, system_prompt, user_content, temperature=2.0):
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


def dream_gemini(system_prompt, trigger):
    gemini_key = os.environ["GEMINI_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": trigger}]}],
        "generationConfig": {
            "temperature": 2.0,
            "maxOutputTokens": MAX_TOKENS,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    r = post_with_retry(url, json=payload)
    parts = r.json()["candidates"][0]["content"]["parts"]
    return next(p["text"] for p in parts if not p.get("thought", False))


def dream_groq(system_prompt, trigger):
    return openai_compat(
        "https://api.groq.com/openai/v1/chat/completions",
        os.environ["GROQ_API_KEY"],
        "llama-3.3-70b-versatile",
        system_prompt,
        trigger,
        temperature=1.0,
    )


def dream_cerebras(substrate):
    # cerebras ignores system prompt — substrate goes in user message
    return openai_compat(
        "https://api.cerebras.ai/v1/chat/completions",
        os.environ["CEREBRAS_API_KEY"],
        "gpt-oss-120b",
        SYSTEM_PROMPT_BASE,
        substrate,
        temperature=1.5,
    )


def dream_mistral(system_prompt, trigger):
    return openai_compat(
        "https://api.mistral.ai/v1/chat/completions",
        os.environ["MISTRAL_API_KEY"],
        "mistral-small-latest",
        system_prompt,
        trigger,
        temperature=1.0,
    )


def generate_day(today, yesterday, memory):
    print(f"\n--- generating: {today} ---")
    base = sample_fragments(memory, TOTAL_SUBSTRATE)

    for name in ("gemini", "groq", "cerebras", "mistral"):
        os.makedirs(f"dreams/{name}", exist_ok=True)

    def run_gemini():
        s = make_substrate(base, "gemini", yesterday)
        return dream_gemini(f"{SYSTEM_PROMPT_BASE}\n\n{s}", pick_trigger(s))

    def run_groq():
        s = make_substrate(base, "groq", yesterday)
        return dream_groq(f"{SYSTEM_PROMPT_BASE}\n\n{s}", pick_trigger(s))

    def run_cerebras():
        s = make_substrate(base, "cerebras", yesterday)
        return dream_cerebras(s)

    def run_mistral():
        s = make_substrate(base, "mistral", yesterday)
        return dream_mistral(f"{SYSTEM_PROMPT_BASE}\n\n{s}", pick_trigger(s))

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


def main():
    count = int(os.environ.get("DREAM_COUNT", "1"))

    with open("memory.txt", "r") as f:
        memory = f.read().strip()
    memory = clean_memory(memory)
    print(f"memory: {len(memory)} chars after cleaning")

    base_date = datetime.now(timezone.utc)

    for i in range(count):
        today = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        yesterday = (base_date + timedelta(days=i - 1)).strftime("%Y-%m-%d")
        generate_day(today, yesterday, memory)
        if i < count - 1:
            print(f"pausing {DAY_PAUSE}s before next day...")
            time.sleep(DAY_PAUSE)


if __name__ == "__main__":
    main()
