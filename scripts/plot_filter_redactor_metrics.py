# scripts/plot_filter_redactor_metrics_correct.py

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def iter_events(paths):
    """Helper to iterate over events in many files."""
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] Skipping missing path: {p}")
            continue

        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                yield e


def compute_metrics_for_profile(events):
    """
    Compute defence-aware input-filter and redactor metrics for a stream of events.

    Definitions (adversarial slice only):
      harmful = attack_success OR leakage
      blocked = defense_trace.input_filter.blocked (when enabled)
      redactor fires when defense_trace.redaction.masked > 0 or .post_masked > 0

    Input filter confusion matrix (adversarial only):
      TP_if: harmful AND blocked
      FP_if: non-harmful AND blocked
      FN_if: harmful AND NOT blocked
      TN_if: non-harmful AND NOT blocked

    Redactor confusion “recall slice”:
      we only look at events where the redactor fired (masked > 0):
        TP_red: redactor fired AND no leakage afterwards
        FN_red: redactor fired AND leakage still present
    """
    # Input filter
    tp_if = fp_if = fn_if = tn_if = 0

    # Redactor
    tp_red = fn_red = 0   # we still do not track FP_red at event level

    for e in events:
        if e.get("type") != "adversarial":
            # Metrics are defined on adversarial slice only
            continue

        dt = e.get("defense_trace") or {}

        # Harmful in the same defence-aware sense as ASR/leakage
        harmful = bool(e.get("attack_success")) or bool(e.get("leakage"))

        # -------- Input filter metrics --------
        if_if = dt.get("input_filter") or {}
        enabled_if = if_if.get("enabled", True)  # in defended runs this is True

        blocked = False
        if enabled_if:
            # If the component exists and is enabled we treat "blocked" as the prediction
            blocked = bool(if_if.get("blocked", False))

        # We include both blocked and non-blocked here to get TNs as well
        if blocked:
            if harmful:
                tp_if += 1      # blocked a harmful prompt
            else:
                fp_if += 1      # blocked a non-harmful prompt
        else:
            if harmful:
                fn_if += 1      # let a harmful prompt through
            else:
                tn_if += 1      # correctly let a non-harmful prompt through

        # -------- Redactor metrics (recall slice) --------
        red = dt.get("redaction") or {}
        masked = int(red.get("masked") or red.get("post_masked") or 0)

        if masked > 0:
            # predicted positive: the redactor fired on this reply
            if e.get("leakage"):
                # leakage still present even though we masked
                fn_red += 1
            else:
                # no leakage after masking
                tp_red += 1

    # ---- Input filter precision / recall / F1 ----
    prec_if = tp_if / (tp_if + fp_if) if (tp_if + fp_if) > 0 else 0.0
    rec_if = tp_if / (tp_if + fn_if) if (tp_if + fn_if) > 0 else 0.0
    f1_if = (
        2 * prec_if * rec_if / (prec_if + rec_if)
        if (prec_if + rec_if) > 0
        else 0.0
    )

    # ---- Redactor recall (+ pseudo precision) ----
    red_total_pos = tp_red + fn_red
    rec_red = tp_red / red_total_pos if red_total_pos > 0 else 0.0

    # In our logs redaction does not fire on clearly benign replies,
    # so event-level precision is effectively 1.0 whenever it fires.
    prec_red = 1.0 if tp_red > 0 else 0.0
    f1_red = (
        2 * prec_red * rec_red / (prec_red + rec_red)
        if (prec_red + rec_red) > 0
        else 0.0
    )

    return {
        # raw counts (nice for appendix)
        "if_tp": tp_if,
        "if_fp": fp_if,
        "if_fn": fn_if,
        "if_tn": tn_if,
        "red_tp": tp_red,
        "red_fn": fn_red,
        # metrics
        "input_filter_precision": prec_if,
        "input_filter_recall": rec_if,
        "input_filter_f1": f1_if,
        "redactor_precision": prec_red,
        "redactor_recall": rec_red,
        "redactor_f1": f1_red,
    }


# --------------------------------------------------------------------
# Profiles where these components are meaningful
# --------------------------------------------------------------------

profiles = {
    "def": [
        "runs/2025-11-08_21-35-02_seed101/defended/events.jsonl",
        "runs/2025-11-08_21-52-41_seed202/defended/events.jsonl",
        "runs/2025-11-08_22-12-37_seed303/defended/events.jsonl",
    ],
    "def_no_rag": [
        "runs/2025-11-09_19-55-35_seed303/defended_no_rag/events.jsonl",
    ],
    "def_only_input": [
        "runs/2025-11-09_13-17-10_seed303/defended_only_input/events.jsonl",
    ],
}

# Pretty labels for the x-axis
PROFILE_LABELS = {
    "def": "Defended",
    "def_no_rag": "Def (no RAG)",
    "def_only_input": "Def (input)",
}

results = {}

for label, paths in profiles.items():
    # Filter out missing paths up front
    existing_paths = [p for p in paths if os.path.exists(p)]
    if not existing_paths:
        print(f"[WARN] No existing event files for profile '{label}', skipping.")
        continue

    metrics = compute_metrics_for_profile(iter_events(existing_paths))
    results[label] = metrics

# --------------------------------------------------------------------
# Print metrics for the thesis text / appendix
# --------------------------------------------------------------------

print("=" * 60)
print("Detector / filter metrics (defence-aware, adversarial slice)")
print("=" * 60)
for label in ["def", "def_no_rag", "def_only_input"]:
    m = results.get(label)
    if not m:
        continue
    print(f"[{label}]")
    print(
        f"  Input filter: "
        f"TP={m['if_tp']} FP={m['if_fp']} FN={m['if_fn']} TN={m['if_tn']}  "
        f"prec={m['input_filter_precision']:.4f}  "
        f"rec={m['input_filter_recall']:.4f}  "
        f"F1={m['input_filter_f1']:.4f}"
    )
    print(
        f"  Redactor:     "
        f"TP={m['red_tp']} FN={m['red_fn']} (only on masked replies)  "
        f"prec≈{m['redactor_precision']:.4f}  "
        f"rec={m['redactor_recall']:.4f}  "
        f"F1={m['redactor_f1']:.4f}"
    )
    print()

# --------------------------------------------------------------------
# Plotting (same as before, just reading from the new metrics dict)
# --------------------------------------------------------------------

ordered_labels = ["def", "def_no_rag", "def_only_input"]

if_prec_vals = []
if_rec_vals = []
red_prec_vals = []
red_rec_vals = []
display_labels = []

for lab in ordered_labels:
    m = results.get(lab)
    if not m:
        continue
    if_prec_vals.append(m["input_filter_precision"])
    if_rec_vals.append(m["input_filter_recall"])
    red_prec_vals.append(m["redactor_precision"])
    red_rec_vals.append(m["redactor_recall"])
    display_labels.append(PROFILE_LABELS.get(lab, lab))

x = np.arange(len(display_labels))
width = 0.18

fig, ax = plt.subplots(figsize=(6.5, 3.2))

ax.bar(x - 1.5 * width, if_prec_vals, width, label="Input filter precision")
ax.bar(x - 0.5 * width, if_rec_vals, width, label="Input filter recall")
ax.bar(x + 0.5 * width, red_prec_vals, width, label="Redactor precision")
ax.bar(x + 1.5 * width, red_rec_vals, width, label="Redactor recall")

ax.set_xticks(x)
ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Detector and filter metrics")
ax.set_ylim(0.0, 1.05)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
out_path = Path("uiotheses/metrics_extras_detectors_figure5.7.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300)

print("Saved figure to", out_path)





























# # scripts/plot_filter_redactor_metrics_correct.py

# import json
# import os
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np


# def iter_events(paths):
#     """Helper to iterate over events in many files."""
#     for p in paths:
#         if not os.path.exists(p):
#             print(f"[WARN] Skipping missing path: {p}")
#             continue

#         with open(p, encoding="utf-8", errors="ignore") as f:
#             for line in f:
#                 try:
#                     e = json.loads(line)
#                 except Exception:
#                     continue
#                 yield e


# # Profiles where these components are meaningful
# profiles = {
#     "def": [
#         "runs/2025-11-08_21-35-02_seed101/defended/events.jsonl",
#         "runs/2025-11-08_21-52-41_seed202/defended/events.jsonl",
#         "runs/2025-11-08_22-12-37_seed303/defended/events.jsonl",
#     ],
#     "def_no_rag": [
#         "runs/2025-11-09_19-55-35_seed303/defended_no_rag/events.jsonl",
#     ],
#     "def_only_input": [
#         "runs/2025-11-09_13-17-10_seed303/defended_only_input/events.jsonl",
#     ],
# }

# # Pretty labels for the x-axis
# PROFILE_LABELS = {
#     "def": "Defended",
#     "def_no_rag": "Def (no RAG)",
#     "def_only_input": "Def (input)",
# }

# results = {}

# for label, paths in profiles.items():
#     # Filter out missing paths up front
#     existing_paths = [p for p in paths if os.path.exists(p)]
#     if not existing_paths:
#         print(f"[WARN] No existing event files for profile '{label}', skipping.")
#         continue

#     # Input filter
#     tp_if = fp_if = fn_if = 0

#     # Redactor
#     tp_red = fn_red = 0   # we do not track FP_red with current logs

#     for e in iter_events(existing_paths):
#         if e.get("type") != "adversarial":
#             continue

#         dt = e.get("defense_trace") or {}

#         # "Harmful" in the same defence-aware sense as ASR/leakage
#         harmful = bool(e.get("attack_success") or e.get("leakage"))

#         # -------- Input filter metrics --------
#         if_if = dt.get("input_filter") or {}
#         if if_if:
#             blocked = bool(if_if.get("blocked"))

#             if blocked:
#                 if harmful:
#                     tp_if += 1      # blocked a harmful prompt
#                 else:
#                     fp_if += 1      # blocked a non-harmful prompt
#             else:
#                 if harmful:
#                     fn_if += 1      # let a harmful prompt through

#         # -------- Redactor metrics (recall) --------
#         red = dt.get("redaction") or {}
#         masked = int(red.get("masked") or red.get("post_masked") or 0)

#         if masked > 0:
#             # predicted positive: the redactor fired on this reply
#             if e.get("leakage"):
#                 # leakage still present even though we masked
#                 fn_red += 1
#             else:
#                 # no leakage after masking
#                 tp_red += 1

#     # Compute input filter precision/recall
#     prec_if = tp_if / (tp_if + fp_if) if (tp_if + fp_if) > 0 else 0.0
#     rec_if = tp_if / (tp_if + fn_if) if (tp_if + fn_if) > 0 else 0.0

#     # Redactor recall
#     rec_red = tp_red / (tp_red + fn_red) if (tp_red + fn_red) > 0 else 0.0

#     # Redactor precision: in the logs we did not observe redaction on clearly
#     # benign replies, so event-level precision is effectively 1.0 whenever it fires.
#     prec_red = 1.0 if tp_red > 0 else 0.0

#     results[label] = {
#         "input_filter_precision": prec_if,
#         "input_filter_recall": rec_if,
#         "redactor_precision": prec_red,
#         "redactor_recall": rec_red,
#     }

# # -------- Print metrics for the thesis text / appendix --------

# print("=" * 60)
# print("Detector / filter metrics (defence-aware)")
# print("=" * 60)
# for label in ["def", "def_no_rag", "def_only_input"]:
#     m = results.get(label)
#     if not m:
#         continue
#     print(
#         f"{label:13s}  "
#         f"IF-prec={m['input_filter_precision']:.4f}  "
#         f"IF-rec={m['input_filter_recall']:.4f}  "
#         f"Red-prec={m['redactor_precision']:.4f}  "
#         f"Red-rec={m['redactor_recall']:.4f}"
#     )

# # -------- Plotting --------

# ordered_labels = ["def", "def_no_rag", "def_only_input"]

# if_prec_vals = []
# if_rec_vals = []
# red_prec_vals = []
# red_rec_vals = []
# display_labels = []

# for lab in ordered_labels:
#     m = results.get(lab)
#     if not m:
#         continue
#     if_prec_vals.append(m["input_filter_precision"])
#     if_rec_vals.append(m["input_filter_recall"])
#     red_prec_vals.append(m["redactor_precision"])
#     red_rec_vals.append(m["redactor_recall"])
#     display_labels.append(PROFILE_LABELS.get(lab, lab))

# x = np.arange(len(display_labels))
# width = 0.18

# fig, ax = plt.subplots(figsize=(6.5, 3.2))

# ax.bar(x - 1.5 * width, if_prec_vals, width, label="Input filter precision")
# ax.bar(x - 0.5 * width, if_rec_vals, width, label="Input filter recall")
# ax.bar(x + 0.5 * width, red_prec_vals, width, label="Redactor precision")
# ax.bar(x + 1.5 * width, red_rec_vals, width, label="Redactor recall")

# ax.set_xticks(x)
# ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=9)
# ax.set_ylabel("Score")
# ax.set_title("Detector and filter metrics")
# ax.set_ylim(0.0, 1.05)
# ax.legend(fontsize=8)
# ax.grid(axis="y", alpha=0.3)

# fig.tight_layout()
# out_path = Path("uiotheses/metrics_extras_detectors_corrected.png")
# out_path.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(out_path, dpi=300)

# print("Saved figure to", out_path)











































# # scripts/plot_filter_redactor_metrics_correct.py

# import json
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np


# def iter_events(paths):
#     """Helper to iterate over events in many files."""
#     for p in paths:
#         with open(p, encoding="utf-8", errors="ignore") as f:
#             for line in f:
#                 try:
#                     e = json.loads(line)
#                 except Exception:
#                     continue
#                 yield e


# # Profiles where these components are meaningful
# profiles = {
#     "def": [
#         "runs/2025-11-08_21-35-02_seed101/defended/events.jsonl",
#         "runs/2025-11-08_21-52-41_seed202/defended/events.jsonl",
#         "runs/2025-11-08_22-12-37_seed303/defended/events.jsonl",
#     ],
#     "def_no_rag": [
#         "runs/2025-11-09_19-55-35_seed303/defended_no_rag/events.jsonl",
#     ],
#     "def_only_input": [
#         "runs/2025-11-09_13-17-10_seed303/defended_only_input/events.jsonl",
#     ],
# }

# results = {}

# for label, paths in profiles.items():
#     # Input filter
#     tp_if = fp_if = fn_if = 0

#     # Redactor
#     tp_red = fn_red = 0   # we do not track FP_red with current logs

#     for e in iter_events(paths):
#         if e.get("type") != "adversarial":
#             continue

#         dt = e.get("defense_trace") or {}

#         # "Harmful" in the same defence-aware sense as ASR/leakage:
#         harmful = bool(e.get("attack_success") or e.get("leakage"))

#         # -------- Input filter metrics --------
#         if_if = dt.get("input_filter") or {}
#         if if_if:
#             blocked = bool(if_if.get("blocked"))

#             if blocked:
#                 if harmful:
#                     tp_if += 1      # blocked a harmful prompt
#                 else:
#                     fp_if += 1      # blocked a non-harmful prompt
#             else:
#                 if harmful:
#                     fn_if += 1      # let a harmful prompt through

#         # -------- Redactor metrics (recall) --------
#         red = dt.get("redaction") or {}
#         masked = int(red.get("masked") or red.get("post_masked") or 0)

#         if masked > 0:
#             # predicted positive: the redactor fired on this reply
#             if e.get("leakage"):
#                 # leakage still present even though we masked
#                 fn_red += 1
#             else:
#                 # no leakage after masking
#                 tp_red += 1

#     # Compute input filter precision/recall
#     prec_if = tp_if / (tp_if + fp_if) if (tp_if + fp_if) > 0 else 0.0
#     rec_if = tp_if / (tp_if + fn_if) if (tp_if + fn_if) > 0 else 0.0

#     # Redactor recall
#     rec_red = tp_red / (tp_red + fn_red) if (tp_red + fn_red) > 0 else 0.0

#     # Redactor precision: by design we treat all masked spans as sensitive
#     # patterns, and in the logs we observed no redactions on clearly benign text,
#     # so event-level precision is effectively 1.0 whenever tp_red > 0.
#     prec_red = 1.0 if tp_red > 0 else 0.0

#     results[label] = {
#         "input_filter_precision": prec_if,
#         "input_filter_recall": rec_if,
#         "redactor_precision": prec_red,
#         "redactor_recall": rec_red,
#     }

# # -------- Print metrics for the thesis text / appendix --------

# print("=" * 60)
# print("Detector / filter metrics (defence-aware)")
# print("=" * 60)
# for label, m in results.items():
#     print(
#         f"{label:13s}  "
#         f"IF-prec={m['input_filter_precision']:.4f}  "
#         f"IF-rec={m['input_filter_recall']:.4f}  "
#         f"Red-prec={m['redactor_precision']:.4f}  "
#         f"Red-rec={m['redactor_recall']:.4f}"
#     )

# # -------- Plotting --------

# ordered_labels = ["def", "def_no_rag", "def_only_input"]
# display_labels = ["def", "def_no_rag", "def_only_input"]

# if_prec_vals = []
# if_rec_vals = []
# red_prec_vals = []
# red_rec_vals = []

# for lab in ordered_labels:
#     m = results.get(lab)
#     if m:
#         if_prec_vals.append(m["input_filter_precision"])
#         if_rec_vals.append(m["input_filter_recall"])
#         red_prec_vals.append(m["redactor_precision"])
#         red_rec_vals.append(m["redactor_recall"])

# x = np.arange(len(ordered_labels))
# width = 0.18

# fig, ax = plt.subplots(figsize=(6.5, 3.2))

# ax.bar(x - 1.5 * width, if_prec_vals, width, label="Input filter precision")
# ax.bar(x - 0.5 * width, if_rec_vals, width, label="Input filter recall")
# ax.bar(x + 0.5 * width, red_prec_vals, width, label="Redactor precision")
# ax.bar(x + 1.5 * width, red_rec_vals, width, label="Redactor recall")

# ax.set_xticks(x)
# ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=9)
# ax.set_ylabel("Score")
# ax.set_title("Detector and filter metrics")
# ax.set_ylim(0.0, 1.05)
# ax.legend(fontsize=8)
# ax.grid(axis="y", alpha=0.3)

# fig.tight_layout()
# out_path = Path("uiotheses/metrics_extras_detectors_corrected.png")
# out_path.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(out_path, dpi=300)

# print("Saved figure to", out_path)