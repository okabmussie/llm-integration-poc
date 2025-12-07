import os
import glob
import json
import ast
from collections import defaultdict

BASE_DIR = "."  # current directory, adjust if needed


def parse_defense_trace(raw):
    """Safely parse the defense_trace field into a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return ast.literal_eval(raw)
        except Exception:
            return {}
    return {}


def load_events(profile_name):
    """
    Load events.jsonl files for a given profile (baseline_pii or defended_pii).
    Returns a list of rows with extra fields: _run_dir and _profile.
    """
    pattern = os.path.join(BASE_DIR, "*", profile_name, "events.jsonl")
    rows = []
    for path in glob.glob(pattern):
        profile_dir = os.path.dirname(path)            # .../baseline_pii or .../defended_pii
        run_root = os.path.dirname(profile_dir)        # .../2025-11-25_23-58-18_seed101
        run_label = os.path.basename(run_root)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data["_run_dir"] = run_label
                data["_profile"] = profile_name
                rows.append(data)
    return rows


def main():
    baseline_rows = load_events("baseline_pii")
    defended_rows = load_events("defended_pii")

    # Index defended rows by (seed, prompt_file)
    defended_index = {}
    for r in defended_rows:
        key = (r.get("seed"), r.get("prompt_file"))
        # If there are multiple rows for the same key, we keep the first one
        defended_index.setdefault(key, r)

    # Collect interesting pairs
    pairs = []

    for b in baseline_rows:
        seed = b.get("seed")
        prompt_file = b.get("prompt_file")
        key = (seed, prompt_file)

        # We only care about adversarial prompts where baseline actually failed
        # (attack_success or leakage)
        if not (b.get("leakage") or b.get("attack_success")):
            continue

        d = defended_index.get(key)
        if not d:
            continue

        # For defended, we want success to be prevented
        if d.get("leakage") or d.get("attack_success"):
            continue

        # Parse defense traces
        b_trace = parse_defense_trace(b.get("defense_trace"))
        d_trace = parse_defense_trace(d.get("defense_trace"))

        input_blocked = (d_trace.get("input_filter") or {}).get("blocked")
        rag_info = d_trace.get("rag_filter") or {}
        rag_masked = rag_info.get("masked")
        rag_docs = rag_info.get("docs") or []
        red_masked = (d_trace.get("redaction") or {}).get("masked")

        # Require that at least one defence actually did something
        defence_did_something = (
            input_blocked
            or (isinstance(rag_masked, int) and rag_masked > 0)
            or (isinstance(red_masked, int) and red_masked > 0)
        )
        if not defence_did_something:
            continue

        pairs.append((b, d, input_blocked, rag_masked, rag_docs, red_masked))

    if not pairs:
        print("No suitable baseline vs defended PII pairs found.")
        return

    print(f"Found {len(pairs)} baseline vs defended pairs where:")
    print("  - baseline had leakage or attack_success")
    print("  - defended had no leakage and no attack_success")
    print("  - at least one defence (input filter, RAG guard, or redactor) was active\n")

    # Print a few examples
    max_examples = 100
    for i, (b, d, input_blocked, rag_masked, rag_docs, red_masked) in enumerate(pairs[:max_examples], start=1):
        print("=" * 80)
        print(f"Example {i}")
        print("- Key")
        print(f"  Seed:        {b.get('seed')}")
        print(f"  Prompt file: {b.get('prompt_file')}")
        print(f"  Scenario:    {b.get('scenario')}")
        print(f"  Baseline run directory: {b.get('_run_dir')}")
        print(f"  Defended run directory: {d.get('_run_dir')}")
        print()

        prompt_text = (b.get("prompt") or "").strip()
        print("- Prompt")
        print(prompt_text if prompt_text else "(empty prompt field)")
        print()

        print("- Baseline behaviour (baseline_pii)")
        print(f"  leakage:        {b.get('leakage')}")
        print(f"  attack_success: {b.get('attack_success')}")
        print("  Reply (truncated):")
        reply_b = (b.get("reply") or "").strip()
        if len(reply_b) > 400:
            reply_b = reply_b[:400] + "..."
        print("  " + reply_b.replace("\n", "\n  "))
        print()

        print("- Defended behaviour (defended_pii)")
        print(f"  input_filter.blocked: {input_blocked}")
        print(f"  rag_filter.masked:    {rag_masked}")
        print(f"  redaction.masked:     {red_masked}")
        print(f"  leakage:              {d.get('leakage')}")
        print(f"  attack_success:       {d.get('attack_success')}")
        print("  Reply (truncated):")
        reply_d = (d.get("reply") or "").strip()
        if len(reply_d) > 400:
            reply_d = reply_d[:400] + "..."
        print("  " + reply_d.replace("\n", "\n  "))
        print()

        # Show which docs contributed context to the defended run
        print("- RAG context documents in defended profile (from rag_filter.docs)")
        if rag_docs:
            for doc_meta in rag_docs:
                doc_id = doc_meta.get("id")
                masked = doc_meta.get("masked")
                if doc_id:
                    corpus_path = f"data/corpus/{doc_id}.txt"
                else:
                    corpus_path = "(unknown id)"
                print(f"  doc id: {doc_id}, masked: {masked}, corpus file: {corpus_path}")
        else:
            print("  (no RAG docs recorded or RAG not used)")
        print()

    print("=" * 80)
    print("Done. You can rerun and adjust max_examples if you need more cases.")


if __name__ == "__main__":
    main()