# scripts/make_ch5_snippets.py
# Build Chapter 5-ready, minimal snippets from runs/*/<profile>/events.jsonl
# Focus: pre-LLM masking (RAG guard), post-LLM masking (redactor), input-filter block,
#        tool allowlist (blocked & allowed), output-policy block, canary evidence.
#
# Usage:
#   python scripts/make_ch5_snippets.py --profiles defended_pii defended_no_rag defended ...
#   # or let it discover latest events for all profiles it sees under runs/*/
#
# Output:
#   runs/<profile>__ch5_snippets.md

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# ---------------- helpers ----------------

# def latest_events_for_profile(profile: str) -> Optional[str]:
#     files = sorted(glob.glob(f"runs/*/{profile}/events.jsonl"))
#     return files[-1] if files else None

def latest_events_for_profile(profile: str, runs_glob: str) -> str | None:
    files = sorted(glob.glob(runs_glob.format(profile=profile)))
    return files[-1] if files else None

def all_profiles_under_runs() -> list[str]:
    # Look for runs/*/*/events.jsonl and collect the middle folder names as "profiles"
    hits = glob.glob("runs/*/*/events.jsonl")
    profs = []
    for p in hits:
        parts = Path(p).parts
        # runs/<run_id>/<profile>/events.jsonl  -> profile = parts[-2]
        if len(parts) >= 4:
            profs.append(parts[-2])
    # preserve order of discovery but unique
    seen, out = set(), []
    for x in profs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                # skip malformed (e.g., dry-run banners accidentally printed)
                continue

def get(dt: Dict[str, Any], *keys, default=None):
    cur = dt
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur

def is_adv(ev): return ev.get("type") == "adversarial"
def is_final(ev): return ev.get("type") == "final"

# -------------- pickers (first good example) --------------

def pick_pre_llm_mask(path: str) -> Optional[Dict[str, Any]]:
    """Adversarial where RAG guard masked >0 and post-LLM masked == 0."""
    for ev in jsonl(path):
        if not is_adv(ev): continue
        if get(ev, "defense_trace", "rag_filter", "masked", default=0) > 0 and \
           get(ev, "defense_trace", "redaction", "masked", default=0) == 0:
            return ev
    return None

def pick_post_llm_mask(path: str) -> Optional[Dict[str, Any]]:
    """Adversarial where no RAG masking but post-LLM redaction masked >0."""
    for ev in jsonl(path):
        if not is_adv(ev): continue
        if get(ev, "defense_trace", "rag_filter", "masked", default=0) == 0 and \
           get(ev, "defense_trace", "redaction", "masked", default=0) > 0:
            return ev
    return None

def pick_input_block(path: str) -> Optional[Dict[str, Any]]:
    """Adversarial blocked by input filter."""
    for ev in jsonl(path):
        if not is_adv(ev): continue
        if get(ev, "defense_trace", "input_filter", "blocked", default=False):
            return ev
    return None

def pick_tool_blocked(path: str) -> Optional[Dict[str, Any]]:
    """Adversarial where tool was attempted and blocked (allowlist on or off)."""
    for ev in jsonl(path):
        if not is_adv(ev): continue
        ta = get(ev, "defense_trace", "tool_allowlist", "attempted", default=False)
        bl = get(ev, "defense_trace", "tool_allowlist", "blocked", default=False)
        al = get(ev, "defense_trace", "tool_allowlist", "allowed", default=False)
        if ta and (bl or not al):
            return ev
    return None

def pick_tool_allowed(path: str) -> Optional[Dict[str, Any]]:
    """Adversarial where tool attempt was allowed/executed."""
    for ev in jsonl(path):
        if not is_adv(ev): continue
        ta = get(ev, "defense_trace", "tool_allowlist", "attempted", default=False)
        al = get(ev, "defense_trace", "tool_allowlist", "allowed", default=False)
        ex = get(ev, "defense_trace", "tool_allowlist", "executed", default=False)
        if ta and al and not get(ev, "defense_trace", "tool_allowlist", "blocked", default=False):
            # executed True is ideal, but keep allowed as success indicator
            return ev
    return None

def pick_output_policy_block(path: str) -> Optional[Dict[str, Any]]:
    """Final record with output-policy blocked == True."""
    for ev in jsonl(path):
        if not is_final(ev): continue
        if get(ev, "defense_trace", "output_policy", "blocked", default=False):
            return ev
    return None

def canary_evidence_line(ev: Dict[str, Any]) -> str:
    dt = ev.get("defense_trace", {})
    by_type = get(dt, "rag_filter", "by_type", default={}) or {}
    masked_ctx = get(dt, "rag_filter", "masked", default=0) or 0
    secret_blocks = by_type.get("secret_block", 0) if isinstance(by_type, dict) else 0
    post_masked = get(dt, "redaction", "masked", default=0) or 0
    return f"masked_ctx={masked_ctx}, secret_blocks={secret_blocks}, post_masked={post_masked}"

# -------------- formatting --------------

def fence(block: str, lang: str = "") -> str:
    return f"```{lang}\n{block.rstrip()}\n```\n"

def safe_text(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)
        except Exception:
            return str(x)
    return str(x)

def write_section(out: list[str], title: str, body: str):
    out.append(f"## {title}\n\n{body.strip()}\n")

# -------------- main --------------

def build_snippets_for_profile(profile: str, events_path: str) -> str:
    run_id = Path(events_path).parts[1] if len(Path(events_path).parts) >= 4 else "run"
    out_lines: list[str] = []
    out_lines.append(f"# Chapter 5: Snippets – {profile} (run {run_id})\n")

    # 1) Pre-LLM masking demo (RAG guard)
    ev = pick_pre_llm_mask(events_path)
    if ev:
        pf = ev.get("prompt_file", "unknown.txt")
        masked = get(ev, "defense_trace", "rag_filter", "masked", default=0)
        prev_before = get(ev, "defense_trace", "rag_filter", "preview_before", default="")
        prev_after  = get(ev, "defense_trace", "rag_filter", "preview_after", default="")
        body = []
        body.append(f"**Prompt file:** `{pf}`  \n**Evidence:** {canary_evidence_line(ev)}")
        if prev_before:
            body.append("\n**Context (before guard, snippet):**\n" + fence(prev_before[:600]))
        if prev_after or masked:
            body.append("**Context (after guard, snippet):**\n" + fence(prev_after[:600]))
        write_section(out_lines, "A. Pre-LLM Masking (RAG Guard)", "\n".join(body))
    else:
        write_section(out_lines, "A. Pre-LLM Masking (RAG Guard)", "_No qualifying example found._")

    # 2) Post-LLM masking demo
    ev = pick_post_llm_mask(events_path)
    if ev:
        pf = ev.get("prompt_file", "unknown.txt")
        reply = safe_text(ev.get("reply", ""))
        final = safe_text(ev.get("final", ""))
        body = []
        body.append(f"**Prompt file:** `{pf}`  \n**Evidence:** {canary_evidence_line(ev)}")
        if reply:
            body.append("\n**Model reply (before redaction):**\n" + fence(reply[:600]))
        if final:
            body.append("**Final (after redaction):**\n" + fence(final[:600]))
        write_section(out_lines, "B. Post-LLM Masking (Redactor)", "\n".join(body))
    else:
        write_section(out_lines, "B. Post-LLM Masking (Redactor)", "_No qualifying example found._")

    # 3) Input-filter block (prompt injection)
    ev = pick_input_block(events_path)
    if ev:
        pf = ev.get("prompt_file", "unknown.txt")
        prompt = safe_text(ev.get("prompt", ""))
        reason = get(ev, "defense_trace", "input_filter", "reason", default="")
        body = []
        body.append(f"**Prompt file:** `{pf}`  \n**Reason:** `{reason}`")
        if prompt:
            body.append("\n**Blocked prompt:**\n" + fence(prompt[:600]))
        write_section(out_lines, "C. Input Filter (Blocked Prompt-Injection)", "\n".join(body))
    else:
        write_section(out_lines, "C. Input Filter (Blocked Prompt-Injection)", "_No qualifying example found._")

    # 4) Tool allowlist – attempted & blocked
    ev = pick_tool_blocked(events_path)
    if ev:
        pf = ev.get("prompt_file", "unknown.txt")
        reply = safe_text(ev.get("reply", ""))
        ta = get(ev, "defense_trace", "tool_allowlist", "attempted", default=False)
        al = get(ev, "defense_trace", "tool_allowlist", "allowed", default=False)
        bl = get(ev, "defense_trace", "tool_allowlist", "blocked", default=False)
        meta = f"attempted={ta}, allowed={al}, blocked={bl}"
        body = []
        body.append(f"**Prompt file:** `{pf}`  \n**Allowlist trace:** {meta}")
        # First line often contains 'tool:' — show it for clarity
        if reply:
            fl = reply.lstrip().splitlines()[0]
            body.append("\n**First line of model reply:**\n" + fence(fl))
        write_section(out_lines, "D. Tool Allowlist (Attempted → Blocked)", "\n".join(body))
    else:
        write_section(out_lines, "D. Tool Allowlist (Attempted → Blocked)", "_No qualifying example found._")

    # 5) Tool allowlist – attempted & allowed
    ev = pick_tool_allowed(events_path)
    if ev:
        pf = ev.get("prompt_file", "unknown.txt")
        reply = safe_text(ev.get("reply", ""))
        ta = get(ev, "defense_trace", "tool_allowlist", "attempted", default=False)
        al = get(ev, "defense_trace", "tool_allowlist", "allowed", default=False)
        ex = get(ev, "defense_trace", "tool_allowlist", "executed", default=False)
        meta = f"attempted={ta}, allowed={al}, executed={ex}"
        body = []
        body.append(f"**Prompt file:** `{pf}`  \n**Allowlist trace:** {meta}")
        if reply:
            fl = reply.lstrip().splitlines()[0]
            body.append("\n**First line of model reply:**\n" + fence(fl))
        write_section(out_lines, "E. Tool Allowlist (Attempted → Allowed/Executed)", "\n".join(body))
    else:
        write_section(out_lines, "E. Tool Allowlist (Attempted → Allowed/Executed)", "_No qualifying example found._")

    # 6) Output-policy block
    ev = pick_output_policy_block(events_path)
    if ev:
        viol = get(ev, "defense_trace", "output_policy", "violations", default=[])
        final = safe_text(ev.get("final", ""))
        body = []
        body.append(f"**Violations:** {safe_text(viol)}")
        if final:
            body.append("\n**Final (policy-blocked or sanitized):**\n" + fence(final[:600]))
        write_section(out_lines, "F. Output Policy (Final Blocked/Sanitized)", "\n".join(body))
    else:
        write_section(out_lines, "F. Output Policy (Final Blocked/Sanitized)", "_No qualifying example found._")

    return "\n".join(out_lines).rstrip() + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="*", help="Profiles to include; if omitted, auto-discovers all profiles with runs.")
    ap.add_argument("--outdir", default="runs", help="Directory to write the snippet files.")
    ap.add_argument("--runs-glob", default="runs/*/{profile}/events.jsonl",
                help="Glob with {profile} placeholder (e.g., 'runs/*seed303*/{profile}/events.jsonl').")
    args = ap.parse_args()

    profiles = args.profiles or all_profiles_under_runs()
    if not profiles:
        print("No profiles found under runs/*/. Run your evals first.")
        return

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for prof in profiles:
        #events_path = latest_events_for_profile(prof)
        events_path = latest_events_for_profile(prof, args.runs_glob)
        if not events_path:
            print(f"(skip) No events for profile '{prof}'.")
            continue
        md = build_snippets_for_profile(prof, events_path)
        out_path = Path(args.outdir) / f"{prof}__ch5_snippets.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()