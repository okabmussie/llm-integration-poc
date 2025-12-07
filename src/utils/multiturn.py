# src/utils/multiturn.py
import json

def load_jsonl_dialogues(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns = json.loads(line)  # list of {role, content}
            parts = []
            for t in turns:
                role = (t.get("role") or "user").upper()
                content = (t.get("content") or "").strip()
                if content:
                    parts.append(f"{role}: {content}")
            yield "\n".join(parts)