# scripts/aggregate_metrics.py
import json
import glob
from pathlib import Path
import statistics
import math
from collections import defaultdict

def wilson(k, n, z=1.96):
    """Wilson 95% CI for k successes out of n trials."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1.0 + (z*z)/n
    c = p + (z*z)/(2*n)
    m = z * math.sqrt((p*(1-p) + (z*z)/(4*n)) / n)
    lo = max(0.0, (c - m) / d)
    hi = min(1.0, (c + m) / d)
    return (p, lo, hi)

def aggregate_runs(base_dir="runs", profile_name=None):
    """Combine metrics.json results across seeds into one summary (by profile)."""
    pattern = f"{base_dir}/**/{profile_name}/metrics.json" if profile_name else f"{base_dir}/**/metrics.json"
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("No metrics.json files found.")
        return

    # Collect per-profile totals (sum counts; compute rates from totals)
    totals = defaultdict(lambda: {
        "attacks": 0,
        "attacks_success": 0,
        "benign": 0,
        "overdefense": 0,
        "leakage_adv": 0,
        "leakage_benign": 0,
    })

    # Also keep raw rates for quick mean±std (optional)
    by_profile_rates = defaultdict(lambda: {
        "attack_success_rate": [],
        "overdefense_rate": [],
        "leakage_rate_adv": [],
        "leakage_rate_benign": [],
    })

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as infile:
                data = json.load(infile)
        except Exception as e:
            print(f"⚠️ Skipped {f}: {e}")
            continue

        prof = data.get("profile", "unknown")
        if profile_name and prof != profile_name:
            continue

        c = data.get("counts", {})
        totals[prof]["attacks"]         += int(c.get("attacks", 0))
        totals[prof]["attacks_success"] += int(c.get("attacks_success", 0))
        totals[prof]["benign"]          += int(c.get("benign", 0))
        totals[prof]["overdefense"]     += int(c.get("overdefense", 0))
        totals[prof]["leakage_adv"]     += int(c.get("leakage_adv", 0))
        totals[prof]["leakage_benign"]  += int(c.get("leakage_benign", 0))

        r = data.get("rates", {})
        by_profile_rates[prof]["attack_success_rate"].append(float(r.get("attack_success_rate", 0.0)))
        by_profile_rates[prof]["overdefense_rate"    ].append(float(r.get("overdefense_rate", 0.0)))
        by_profile_rates[prof]["leakage_rate_adv"    ].append(float(r.get("leakage_rate_adv", 0.0)))
        by_profile_rates[prof]["leakage_rate_benign" ].append(float(r.get("leakage_rate_benign", 0.0)))

    if not totals:
        print("No matching profiles found in metrics.json files.")
        return

    def mean_std(values):
        if not values:
            return 0.0, 0.0
        if len(values) == 1:
            return float(values[0]), 0.0
        return statistics.mean(values), statistics.stdev(values)

    # Print per-profile aggregate
    for prof, t in sorted(totals.items()):
        A  = t["attacks"]
        AS = t["attacks_success"]
        B  = t["benign"]
        OD = t["overdefense"]
        LA = t["leakage_adv"]
        LB = t["leakage_benign"]

        asr_p, asr_lo, asr_hi = wilson(AS, A)
        odr_p, odr_lo, odr_hi = wilson(OD, B)
        lra_p, lra_lo, lra_hi = wilson(LA, A)
        lrb_p, lrb_lo, lrb_hi = wilson(LB, B)

        m_asr, s_asr   = mean_std(by_profile_rates[prof]["attack_success_rate"])
        m_odr, s_odr   = mean_std(by_profile_rates[prof]["overdefense_rate"])
        m_lra, s_lra   = mean_std(by_profile_rates[prof]["leakage_rate_adv"])
        m_lrb, s_lrb   = mean_std(by_profile_rates[prof]["leakage_rate_benign"])

        print(f"\n📊 Aggregated Results for profile: {prof}")
        print("-------------------------------------------------")
        print(f"Counts  -> A={A}  AS={AS}  B={B}  OD={OD}  LA={LA}  LB={LB}")
        print(f"ASR     -> {asr_p:.4f} [{asr_lo:.4f}, {asr_hi:.4f}]    (mean±std by run: {m_asr:.4f} ± {s_asr:.4f})")
        print(f"ODR     -> {odr_p:.4f} [{odr_lo:.4f}, {odr_hi:.4f}]    (mean±std by run: {m_odr:.4f} ± {s_odr:.4f})")
        print(f"LeakAdv -> {lra_p:.4f} [{lra_lo:.4f}, {lra_hi:.4f}]    (mean±std by run: {m_lra:.4f} ± {s_lra:.4f})")
        print(f"LeakBen -> {lrb_p:.4f} [{lrb_lo:.4f}, {lrb_hi:.4f}]    (mean±std by run: {m_lrb:.4f} ± {s_lrb:.4f})")
        print("-------------------------------------------------\n")

# Examples:
if __name__ == "__main__":
    aggregate_runs(profile_name="baseline_pii")
    aggregate_runs(profile_name="defended_pii")