import json
import glob
import os
from textwrap import shorten

# --------------------------------------------------------------------
# Helper: simple LaTeX escaper for text inside lstlisting captions etc.
# (Inside lstlisting the content is verbatim, but we escape a few things
#  in the free text we write around it.)
# --------------------------------------------------------------------
def escape_latex(s: str) -> str:
    if s is None:
        return ""
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("_", "\\_")
         .replace("%", "\\%")
         .replace("#", "\\#")
         .replace("&", "\\&")
    )

# --------------------------------------------------------------------
# Load events from a given profile ("baseline_pii" or "defended_pii")
# under all run directories in the current folder.
# --------------------------------------------------------------------
def load_events(profile_name: str):
    events = []
    pattern = f"*/{profile_name}/events.jsonl"
    for path in glob.glob(pattern):
        run_dir = os.path.dirname(path)  # e.g. 2025-11-25_seed303/baseline_pii
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                obj["_run_dir"] = run_dir
                events.append(obj)
    return events

# --------------------------------------------------------------------
# Main logic: find baseline vs defended pairs where
#  - baseline had leakage or attack_success == True
#  - defended has no leakage and no attack_success
#  - at least one defence is active
# --------------------------------------------------------------------
def main(max_examples: int = 5, appendix_tex: str = "appendix_pii_examples_30.tex"):
    baseline = load_events("baseline_pii")
    defended = load_events("defended_pii")

    # index by (seed, prompt_file)
    def key(ev):
        return (ev.get("seed"), ev.get("prompt_file"))

    base_map = {}
    for ev in baseline:
        base_map.setdefault(key(ev), []).append(ev)

    def_map = {}
    for ev in defended:
        def_map.setdefault(key(ev), []).append(ev)

    pairs = []
    for k, base_list in base_map.items():
        if k not in def_map:
            continue
        def_list = def_map[k]
        # we expect 1:1, but be robust
        for b in base_list:
            for d in def_list:
                # baseline must have leakage or attack_success
                if not (b.get("leakage") or b.get("attack_success")):
                    continue
                # defended must have neither
                if d.get("leakage") or d.get("attack_success"):
                    continue

                # parse defence_trace if it is stored as str
                def_trace = d.get("defense_trace") or d.get("defence_trace")
                if isinstance(def_trace, str):
                    try:
                        def_trace = eval(def_trace)
                    except Exception:
                        def_trace = {}

                input_f = (def_trace or {}).get("input_filter", {})
                rag_f = (def_trace or {}).get("rag_filter", {})
                red_f = (def_trace or {}).get("redaction", {})

                # at least one defence actually did something / was enabled
                if not (
                    input_f.get("enabled") or
                    rag_f.get("enabled") or
                    red_f.get("enabled")
                ):
                    continue

                pairs.append((b, d, def_trace))

    print(f"Found {len(pairs)} baseline vs defended pairs where:")
    print("  - baseline had leakage or attack_success")
    print("  - defended had no leakage and no attack_success")
    print("  - at least one defence (input filter, RAG guard, or redactor) was active\n")

    # ----------------------------------------------------------------
    # Print examples AND collect them for LaTeX appendix
    # ----------------------------------------------------------------
    examples_tex = []
    for idx, (b, d, def_trace) in enumerate(pairs[:max_examples], start=1):
        seed = b.get("seed")
        prompt_file = b.get("prompt_file")
        scenario = b.get("scenario")
        base_dir = b.get("_run_dir")
        def_dir = d.get("_run_dir")

        prompt = (b.get("prompt") or "").rstrip("\n")
        base_reply = (b.get("reply") or "").strip()
        def_reply = (d.get("reply") or "").strip()

        # defence trace pieces
        input_f = (def_trace or {}).get("input_filter", {})
        rag_f = (def_trace or {}).get("rag_filter", {})
        red_f = (def_trace or {}).get("redaction", {})

        in_blocked = input_f.get("blocked")
        rag_masked = rag_f.get("masked")
        red_masked = red_f.get("masked")

        # RAG docs if present
        rag_docs = (rag_f or {}).get("docs") or []
        rag_doc_lines = []
        for doc in rag_docs:
            doc_id = doc.get("id")
            masked = doc.get("masked")
            corpus_path = f"data/corpus/{doc_id}.txt.txt" if doc_id else "?"
            rag_doc_lines.append(f"  doc id: {doc_id}, masked: {masked}, corpus file: {corpus_path}")

        # ---------- terminal output ----------
        print("=" * 80)
        print(f"Example {idx}")
        print("- Key")
        print(f"  Seed:        {seed}")
        print(f"  Prompt file: {prompt_file}")
        print(f"  Scenario:    {scenario}")
        print(f"  Baseline run directory: {base_dir}")
        print(f"  Defended run directory: {def_dir}\n")

        print("- Prompt")
        print(prompt or "[empty prompt]")
        print("\n- Baseline behaviour (baseline_pii)")
        print(f"  leakage:        {b.get('leakage')}")
        print(f"  attack_success: {b.get('attack_success')}")
        print("  Reply (truncated):")
        print("  " + shorten(base_reply.replace("\n", " "), width=200, placeholder="..."))
        print("\n- Defended behaviour (defended_pii)")
        print(f"  input_filter.blocked: {in_blocked}")
        print(f"  rag_filter.masked:    {rag_masked}")
        print(f"  redaction.masked:     {red_masked}")
        print(f"  leakage:              {d.get('leakage')}")
        print(f"  attack_success:       {d.get('attack_success')}")
        print("  Reply (truncated):")
        print("  " + shorten(def_reply.replace("\n", " "), width=200, placeholder="..."))

        print("\n- RAG context documents in defended profile (from rag_filter.docs)")
        if rag_doc_lines:
            for line in rag_doc_lines:
                print(" " + line)
        else:
            print("  (no RAG docs recorded or RAG not used)")
        print()

        # ---------- LaTeX appendix content ----------
        tex_block = []
        tex_block.append(f"\\subsubsection*{{Example {idx} -- {escape_latex(prompt_file or 'unknown')} (seed {seed})}}")
        tex_block.append("")
        tex_block.append("\\begin{lstlisting}[style=tightcode]")
        tex_block.append(f"Adversarial prompt (file {prompt_file})")
        tex_block.append(prompt or "[empty prompt]")
        tex_block.append("")
        tex_block.append("Baseline reply")
        tex_block.append(shorten(base_reply, width=400, placeholder=" ..."))  # rough truncation for appendix
        tex_block.append("")
        tex_block.append("Defended reply")
        tex_block.append(shorten(def_reply, width=400, placeholder=" ..."))
        tex_block.append("")
        tex_block.append("Defence trace (defended)")
        tex_block.append(f"input_filter.blocked = {in_blocked}")
        tex_block.append(f"rag_filter.masked    = {rag_masked}")
        tex_block.append(f"redaction.masked     = {red_masked}")
        tex_block.append(f"leakage              = {d.get('leakage')}")
        tex_block.append(f"attack_success       = {d.get('attack_success')}")
        if rag_doc_lines:
            tex_block.append("")
            tex_block.append("RAG docs:")
            for line in rag_doc_lines:
                tex_block.append(line)
        tex_block.append("\\end{lstlisting}")
        tex_block.append("")

        examples_tex.append("\n".join(tex_block))

    # ----------------------------------------------------------------
    # Write LaTeX appendix file
    # ----------------------------------------------------------------
    if examples_tex:
        with open(appendix_tex, "w", encoding="utf-8") as f:
            f.write("% Auto-generated PII defence examples\n")
            f.write("\\section{PII defence examples from log analysis}\n\n")
            for block in examples_tex:
                f.write(block)
                f.write("\n\n")
        print(f"Written LaTeX appendix file: {appendix_tex}")
    else:
        print("No qualifying examples found; no appendix file written.")


if __name__ == "__main__":
    # you can change max_examples if you want more or fewer
    main(max_examples=30)