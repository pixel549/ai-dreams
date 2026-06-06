import json
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "conversations.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "memory.txt"

print(f"Loading {src}...", flush=True)
with open(src, encoding="utf-8") as f:
    data = json.load(f)

print(f"Found {len(data)} conversations", flush=True)

count = 0
with open(dst, "w", encoding="utf-8") as out:
    for convo in data:
        for msg in convo.get("chat_messages", []):
            text = msg.get("text", "").strip()
            if text:
                out.write(text)
                out.write("\n\n")
                count += 1

print(f"Done — {count} messages written to {dst}")
