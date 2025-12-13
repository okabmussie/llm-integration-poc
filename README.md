
# ============================================
# LLM Integration Security Prototype - Full Setup & Experiments
# ============================================

## Overview
This repository contains a research prototype developed for a master thesis investigating security risks when integrating Large Language Models (LLMs) into applications. The system conducts reproducible experiments comparing baseline configurations with various defended profiles.

### Key Features
- **Comparative Analysis**: Baseline vs. defended LLM integration profiles
- **Attack Scenarios**: Adaptive and non-adaptive attack simulations
- **Defense Ablation Studies**: Isolated component analysis (input filter, tool gate, RAG, policy)
- **PII-focused Study**: Dedicated experiments for Personally Identifiable Information protection
- **Comprehensive Logging**: All interactions recorded to JSON files in `runs_main/`, `runs/`,`runs_pii/`
- **Automated Analysis**: Scripts to aggregate results into tables and figures

### Main Experiments Covered
* **Baseline versus defended profiles** - Core comparison of unprotected vs protected setups
* **Adaptive and non-adaptive attacks** - Different attacker sophistication levels
* **Defense ablations** - Component isolation studies:
  - Input filter only
  - Tool gate only  
  - RAG only
  - Policy only
  - Without RAG
* **PII-focused sub-study** - Specialized experiments for personally identifiable information protection

---


set -e  # exit on first error

echo "========================================"
echo "LLM Security Prototype - Complete Setup"
echo "========================================"

# ------------------------------
# 1. Setup & installation
# ------------------------------
echo -e "\n[1/5] SETUP & INSTALLATION"
echo "----------------------------------------"

# Clone repository (skip if directory already exists)
if [ ! -d "llm-integration-poc" ]; then
  echo "Cloning repository..."
  git clone https://github.com/okabmussie/llm-integration-poc.git
fi

cd llm-integration-poc || exit 1

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
fi

# Activate virtual environment
# shellcheck disable=SC1091
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Configure API key (prompt user, hidden input)
if [ ! -f ".env" ]; then
  echo "Configuring OpenAI API key..."
  read -s -p "Enter your OpenAI API key: " api_key
  echo
  echo "OPENAI_API_KEY=$api_key" > .env
  echo ".env written."
else
  echo ".env already exists – keeping existing API key."
fi

# Build retrieval index
echo "Building retrieval index..."
python -m src.rag.build_index

echo -e "\n Setup completed successfully!"
echo "----------------------------------------"

# ------------------------------
# 2. Main experiments
# ------------------------------
echo -e "\n[2/5] MAIN EXPERIMENTS"
echo "----------------------------------------"

echo "Running baseline vs defended profiles (non-adaptive)..."
python -m src.eval.runner --config config/baseline.yaml  --limit 0 --adapt 0
python -m src.eval.runner --config config/defended.yaml  --limit 0 --adapt 0

echo "Running baseline vs defended profiles (adaptive)..."
python -m src.eval.runner --config config/baseline.yaml  --limit 0 --adapt 1
python -m src.eval.runner --config config/defended.yaml  --limit 0 --adapt 1

echo -e "\n Main experiments completed!"
echo "----------------------------------------"

# ------------------------------
# 3. Ablation studies
# ------------------------------
echo -e "\n[3/5] ABLATION STUDIES"
echo "----------------------------------------"

echo "Running ablation profiles (non-adaptive)..."
python -m src.eval.runner --config config/defended_only_input.yaml   --limit 0 --adapt 0
python -m src.eval.runner --config config/defended_only_tool.yaml    --limit 0 --adapt 0
python -m src.eval.runner --config config/defended_only_rag.yaml     --limit 0 --adapt 0
python -m src.eval.runner --config config/defended_no_rag.yaml       --limit 0 --adapt 0
python -m src.eval.runner --config config/defended_only_policy.yaml  --limit 0 --adapt 0

echo "Running ablation profiles (adaptive)..."
for cfg in config/defended_only_input.yaml \
           config/defended_only_tool.yaml  \
           config/defended_only_rag.yaml   \
           config/defended_no_rag.yaml
do
  echo "Processing: $cfg (adaptive)..."
  python -m src.eval.runner --config "$cfg" --limit 0 --adapt 1
done

echo -e "\n Ablation studies completed!"
echo "----------------------------------------"

# ------------------------------
# 4. PII-focused studies
# ------------------------------
echo -e "\n[4/5] PII-FOCUSED STUDIES"
echo "----------------------------------------"

echo "Running PII studies (non-adaptive)..."
python -m src.eval.runner --config config/baseline_pii.yaml  --limit 0 --adapt 0
python -m src.eval.runner --config config/defended_pii.yaml  --limit 0 --adapt 0

echo "Running PII studies (adaptive)..."
python -m src.eval.runner --config config/baseline_pii.yaml  --limit 0 --adapt 1
python -m src.eval.runner --config config/defended_pii.yaml  --limit 0 --adapt 1

echo -e "\n PII studies completed!"
echo "----------------------------------------"

# ------------------------------
# 5. Results analysis
# ------------------------------
echo -e "\n[5/5] RESULTS ANALYSIS"
echo "----------------------------------------"

### Aggregate baseline vs defended (Figures 5.1 and 5.3)

# This command recomputes the main aggregate results used in:
# - Figure 5.1: Overall attack success rates (baseline vs defended, aggregated over seeds)
# - Figure 5.3: ASR by attack family (baseline vs defended, aggregated over seeds)
# - Figure 5.2 ablation - aggregated

# fig - 5.4
python scripts/make_tables_and_figs_try.py \
  --outdir runs/ablations_agg_fig5.4 \
  --runs-glob 'runs/*seed*/{profile}/events.jsonl' \
  --profiles baseline,defended,defended_only_input,defended_only_rag,defended_only_tool,defended_no_rag,defended_only_output \
  --aggregate --seeds 101,202,303 \
  --fig-stem fig_asr_by_profile_agg_fig5.4 \
  --plot-title 'Attack success rate by defence profile with ablations aggregated over seeds 101 202 303'

python scripts/make_tables_and_figs_try.py \
  --outdir runs/agg_nonadapt \
  --runs-glob 'runs/*seed*/{profile}/events.jsonl' \
  --profiles baseline,defended \
  --aggregate --seeds 101,202,303

echo -e "\n2) Generating ablation plots (seed 303)..."
mkdir -p runs/ablations_seed303
python scripts/make_tables_and_figs_try.py \
  --outdir runs/ablations_seed303 \
  --runs-glob 'runs/*seed303*/{profile}/events.jsonl' \
  --profiles baseline,defended,defended_only_input,defended_only_rag,defended_only_tool,defended_no_rag \
  --fig-stem fig_asr_by_profile \
  --plot-title 'ASR by Profile — Ablations seed 303 non adaptive'

# Figure 5.2 ASR by profile - adaptive single seed 101
python scripts/make_tables_and_figs_try.py \
  --outdir runs/seed101_adapt \
  --runs-glob 'runs/*seed101*/{profile}/events.jsonl' \
  --profiles baseline,defended
# including defended_only_policy fig-5.5
python scripts/make_tables_and_figs_try.py \
  --outdir runs/all_ablations_seed303 \
  --runs-glob 'runs/*seed303*/{profile}/events.jsonl' \
  --profiles baseline,defended,defended_only_input,defended_only_rag,defended_only_tool,defended_no_rag,defended_only_output \
  --fig-stem fig_asr_by_profile \
  --plot-title 'ASR by Profile — Ablations seed 303 non adaptive'

echo -e "\n3) Generating overall performance metrics..."
python scripts/plot_overall_metrics.py

echo -e "\n4) Generating filter and redactor metrics..."
python scripts/plot_filter_redactor_metrics.py

echo -e "\n Results analysis completed!"
echo "========================================"



# latency 
time python -m src.eval.runner \--config config/defended.yaml \
  --limit 100

time python -m src.eval.runner --config config/baseline.yaml  --limit 100
time python -m src.eval.runner --config config/defended.yaml  --limit 100

# Get the latest baseline events file
ev_base=$(ls -dt runs/*/baseline/events.jsonl | head -1)

# Print replies for jailbreak-style prompts in that run
python - <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)
        if e.get("type") == "adversarial" and "jailbreak" in (e.get("prompt_file") or ""):
            print(e.get("prompt_file"), "=>", e.get("reply") or e.get("final"))
PY "$ev_base"


# ------------------------------
# Summary
# ------------------------------
echo -e "\n EXPERIMENT SUMMARY"
echo "========================================"
echo "All experiments have been completed."
echo
echo "Generated output directories:"
echo "- runs/agg_nonadapt/      (baseline vs defended aggregate results)"
echo "- runs/ablations_seed303/ (ablation study results)"
echo
echo "Generated figures (in uiotheses/):"
echo
echo "CSV summaries and additional plots are stored under runs/*/."
echo "========================================"
