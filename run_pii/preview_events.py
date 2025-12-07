# preview_events.py
import json
from pathlib import Path

ROOT = Path(".")

def iter_events():
    for events_path in ROOT.rglob("events.jsonl"):
        if "pii" not in str(events_path):
            continue
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield events_path, obj

if __name__ == "__main__":
    print("=== Events schema preview (first few rows) ===")
    shown = 0
    for path, obj in iter_events():
        print(f"\nFile: {path}")
        print("Keys:", sorted(obj.keys()))
        # try to show any obvious fields
        for k in ("file", "prompt", "user_prompt", "scenario", "defense_trace", "defence_trace", "event"):
            if k in obj:
                print(f"{k}:", repr(str(obj[k])[:120]))
        shown += 1
        if shown >= 5:
            break
    if shown == 0:
        print("No events found – check working directory.")