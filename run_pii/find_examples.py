# find_examples.py
import json
import ast
from pathlib import Path
from collections import defaultdict

ROOT = Path(".")

def load_outputs():
    """Load all defended_pii outputs keyed by (run_dir, prompt_file)."""
    outputs = {}
    for outputs_path in ROOT.rglob("outputs.jsonl"):
        if "defended_pii" not in str(outputs_path):
            continue
        run_dir = outputs_path.parent
        with outputs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_file = obj.get("file")
                if not prompt_file:
                    continue
                key = (run_dir, prompt_file)
                outputs[key] = obj
    return outputs

def load_events():
    """Load all defended_pii events keyed by (run_dir, prompt_file)."""
    events = defaultdict(list)
    for events_path in ROOT.rglob("events.jsonl"):
        if "defended_pii" not in str(events_path):
            continue
        run_dir = events_path.parent
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_file = obj.get("prompt_file")
                if not prompt_file:
                    continue
                key = (run_dir, prompt_file)
                events[key].append(obj)
    return events

def parse_defense_trace(raw):
    """defense_trace is stored as a string representation of a dict."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}

def main():
    outputs = load_outputs()
    events = load_events()

    print(f"Loaded {len(outputs)} output entries with matching files")

    print("\n=== PII defence examples with input filter + RAG + redactor ===")
    shown = 0

    for key, out_row in outputs.items():
        run_dir, prompt_file = key
        ev_list = events.get(key, [])
        if not ev_list:
            continue

        # take the first event for this prompt file
        ev = ev_list[0]
        dtrace = parse_defense_trace(ev.get("defense_trace"))

        inp = dtrace.get("input_filter", {}) or {}
        rag = dtrace.get("rag_filter", {}) or {}
        red = dtrace.get("redaction", {}) or {}

        # We want examples that actually exercised RAG / redactor
        rag_masked = rag.get("masked")
        red_masked = red.get("masked")

        # Only keep cases where input filter did NOT block
        if inp.get("blocked"):
            continue

        # we want some masking evidence
        if not rag_masked and not red_masked:
            continue

        # and we want successful defence (no leak and no attack_success)
        if ev.get("leakage"):
            continue
        if ev.get("attack_success"):
            continue

        prompt = ev.get("prompt", "").strip()
        reply = out_row.get("reply", "").strip()

        print("\n--- Example ---")
        print("Run directory:", run_dir)
        print("Prompt file:", prompt_file)
        print("\nAdversarial prompt:")
        print(prompt if prompt else "(prompt text not recorded)")
        print("\nFinal reply (after defences):")
        print(reply)
        print("\nDefence trace counts:")
        print("  input_filter.blocked:", inp.get("blocked"))
        print("  rag_filter.masked:", rag_masked)
        print("  redaction.masked:", red_masked)
        print("  leakage flag:", ev.get("leakage"))
        print("  attack_success flag:", ev.get("attack_success"))

        shown += 1
        if shown >= 3:
            break

    if shown == 0:
        print("No suitable defended PII examples found.")

if __name__ == "__main__":
    main()