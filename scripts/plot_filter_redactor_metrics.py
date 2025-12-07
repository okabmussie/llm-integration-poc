# scripts/plot_filter_redactor_metrics.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Helper to iterate over events in many files
def iter_events(paths):
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                yield e


# Profiles that actually use input_filter and/or redactor
profiles = {
    # full defended profile, three seeds
    "def": [
        "runs_main/2025-11-08_21-35-02_seed101/defended/events.jsonl",
        "runs_main/2025-11-08_21-52-41_seed202/defended/events.jsonl",
        "runs_main/2025-11-08_22-12-37_seed303/defended/events.jsonl",
    ],
    # defended without RAG, still uses input filter and redactor
    "def_no_rag": [
        "runs_main/2025-11-09_19-55-35_seed303/defended_no_rag/events.jsonl",
    ],
    # input filter only
    "def_only_input": [
        "runs_main/2025-11-09_13-17-10_seed303/defended_only_input/events.jsonl",
    ],
    # redactor only (used for metrics, but we will not plot it)
    "def_only_rag": [
        "runs_main/2025-11-09_17-29-10_seed303/defended_only_rag/events.jsonl",
    ],
}

results = {}

for label, paths in profiles.items():
    tp_if = fp_if = fn_if = 0
    tp_red = fn_red = 0

    for e in iter_events(paths):
        if e.get("type") != "adversarial":
            continue

        dt = e.get("defense_trace") or {}

        # Harmful if attack_success or leakage flag
        harmful = bool(e.get("attack_success") or e.get("leakage"))

        # ---- input filter metrics ----
        if_if = dt.get("input_filter") or {}
        if if_if:
            blocked = bool(if_if.get("blocked"))

            if blocked:
                if harmful:
                    tp_if += 1
                else:
                    fp_if += 1
            else:
                if harmful:
                    fn_if += 1

        # ---- redactor recall ----
        red = dt.get("redaction") or {}
        if red:
            masked = red.get("masked") or red.get("post_masked") or 0
            if masked:  # only cases where something was actually redacted
                leaked = bool(e.get("leakage"))
                if leaked:
                    fn_red += 1
                else:
                    tp_red += 1

    # compute metrics
    prec_if = tp_if / (tp_if + fp_if) if (tp_if + fp_if) > 0 else None
    rec_if = tp_if / (tp_if + fn_if) if (tp_if + fn_if) > 0 else None
    rec_red = tp_red / (tp_red + fn_red) if (tp_red + fn_red) > 0 else None

    results[label] = {
        "input_filter_precision": prec_if,
        "input_filter_recall": rec_if,
        "redactor_recall": rec_red,
    }

# Print numbers for thesis text / appendix
for label, m in results.items():
    print(
        f"{label:13s}  "
        f"IF-prec={m['input_filter_precision']!r}  "
        f"IF-rec={m['input_filter_recall']!r}  "
        f"Red-rec={m['redactor_recall']!r}"
    )

# ---------- plotting (recall only) ----------

# Only plot profiles where these components are meaningful
ordered_labels = ["def", "def_no_rag", "def_only_input"]

labels = []
input_rec_vals = []
red_rec_vals = []

for lab in ordered_labels:
    m = results.get(lab)
    if not m:
        continue
    labels.append(lab)
    input_rec_vals.append(m["input_filter_recall"] or 0.0)
    red_rec_vals.append(m["redactor_recall"] or 0.0)

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots(figsize=(6.0, 3.0))

ax.bar(x - width / 2, input_rec_vals, width, label="Input filter recall")
ax.bar(x + width / 2, red_rec_vals,   width, label="Redactor recall")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Detector and filter recall", fontsize=12)
ax.legend(fontsize=8)

fig.tight_layout()

out_path = Path("uiotheses/metrics_extras_detectors_303_consistent.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300)

print("Saved figure to", out_path)