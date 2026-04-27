"""Aggregate per-seed eval reports into mean ± std summary.

Reads:
    results/baseline_seed*/eval_report.json
    results/augmented_seed*/eval_report.json

Writes:
    results/aggregated.json     — dict with per-config mean/std across seeds
    results/aggregated.md       — human-readable markdown table

Usage:
    python scripts/aggregate_seeds.py
"""
from __future__ import annotations
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RESULTS_DIR  # noqa: E402

METRIC_KEYS = (
    "accuracy",
    "precision_phish",
    "recall_phish",
    "f1_phish",
    "macro_f1",
    "roc_auc",
)
SPLIT_KEYS = ("standard", "smart_attacker")


def _mean_std(values):
    """Return (mean, std). std=0 when only one sample is present."""
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return stats.fmean(values), stats.pstdev(values)


def _collect_runs() -> dict[str, list[dict]]:
    """Group per-seed eval reports by config (baseline / augmented)."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for cfg in ("baseline", "augmented"):
        for d in sorted(RESULTS_DIR.glob(f"{cfg}_seed*")):
            rep_path = d / "eval_report.json"
            if not rep_path.exists():
                continue
            try:
                rep = json.loads(rep_path.read_text())
            except Exception as e:
                print(f"  ! skipping {rep_path}: {e}")
                continue
            rep["_dir"] = str(d)
            rep["_seed"] = int(d.name.split("seed")[-1])
            runs[cfg].append(rep)
    return runs


def aggregate(runs: dict[str, list[dict]]) -> dict:
    """For each (config, split, metric), produce {mean, std, seeds, values}."""
    out: dict = {"configs": {}}
    for cfg_name, reports in runs.items():
        if not reports:
            continue
        cfg_out = {"n_seeds": len(reports), "seeds": [r["_seed"] for r in reports]}
        for split in SPLIT_KEYS:
            split_out = {}
            for metric in METRIC_KEYS:
                vals = [r[split].get(metric) for r in reports if split in r]
                m, s = _mean_std(vals)
                if m is None:
                    continue
                split_out[metric] = {"mean": m, "std": s, "values": vals}
            cfg_out[split] = split_out
        # Per-edit ablation aggregation
        ops = set()
        for r in reports:
            ops.update((r.get("ablation") or {}).keys())
        ablation_out = {}
        for op in sorted(ops):
            op_metrics = {}
            for metric in METRIC_KEYS:
                vals = [
                    r["ablation"][op].get(metric)
                    for r in reports
                    if op in r.get("ablation", {})
                ]
                m, s = _mean_std(vals)
                if m is None:
                    continue
                op_metrics[metric] = {"mean": m, "std": s, "values": vals}
            ablation_out[op] = op_metrics
        cfg_out["ablation"] = ablation_out
        out["configs"][cfg_name] = cfg_out
    return out


def render_markdown(agg: dict) -> str:
    lines = ["# Multi-seed aggregated results\n"]
    cfgs = agg.get("configs", {})

    if not cfgs:
        return "_No per-seed eval reports found under results/._\n"

    # Headline 2x2: F1 (phishing), accuracy
    lines.append("## Headline (mean ± std across seeds)\n")
    lines.append("| Model | Standard F1 | Smart-attacker F1 | Standard acc | Smart-attacker acc |")
    lines.append("|---|---|---|---|---|")
    for cfg_name in ("baseline", "augmented"):
        c = cfgs.get(cfg_name)
        if not c:
            continue
        f1_std = c["standard"]["f1_phish"]
        f1_sm = c["smart_attacker"]["f1_phish"]
        ac_std = c["standard"]["accuracy"]
        ac_sm = c["smart_attacker"]["accuracy"]
        lines.append(
            f"| {cfg_name} (n={c['n_seeds']}) "
            f"| {f1_std['mean']:.4f} ± {f1_std['std']:.4f} "
            f"| {f1_sm['mean']:.4f} ± {f1_sm['std']:.4f} "
            f"| {ac_std['mean']:.4f} ± {ac_std['std']:.4f} "
            f"| {ac_sm['mean']:.4f} ± {ac_sm['std']:.4f} |"
        )

    # Deltas
    if "baseline" in cfgs and "augmented" in cfgs:
        lines.append("\n## Effect sizes\n")
        b = cfgs["baseline"]
        a = cfgs["augmented"]
        b_std = b["standard"]["f1_phish"]["mean"]
        b_sm = b["smart_attacker"]["f1_phish"]["mean"]
        a_std = a["standard"]["f1_phish"]["mean"]
        a_sm = a["smart_attacker"]["f1_phish"]["mean"]
        lines.append(f"- Baseline drop under attack: F1 {b_std:.4f} → {b_sm:.4f} = **{b_sm - b_std:+.4f}**")
        lines.append(f"- Augmented drop under attack: F1 {a_std:.4f} → {a_sm:.4f} = **{a_sm - a_std:+.4f}**")
        lines.append(f"- Augmentation effect on standard test: F1 {b_std:.4f} → {a_std:.4f} = **{a_std - b_std:+.4f}**")
        lines.append(f"- Augmentation effect on smart-attacker test: F1 {b_sm:.4f} → {a_sm:.4f} = **{a_sm - b_sm:+.4f}**")

    # Per-edit ablation, augmented model
    if "augmented" in cfgs and cfgs["augmented"].get("ablation"):
        lines.append("\n## Per-edit ablation (augmented model, mean ± std)\n")
        lines.append("| Edit | F1 (phishing) | Recall (phishing) |")
        lines.append("|---|---|---|")
        # baseline rows from "standard" for reference
        std_f1 = cfgs["augmented"]["standard"]["f1_phish"]
        std_r = cfgs["augmented"]["standard"]["recall_phish"]
        lines.append(
            f"| _none (standard test)_ "
            f"| {std_f1['mean']:.4f} ± {std_f1['std']:.4f} "
            f"| {std_r['mean']:.4f} ± {std_r['std']:.4f} |"
        )
        for op, m in cfgs["augmented"]["ablation"].items():
            f1 = m.get("f1_phish", {})
            rc = m.get("recall_phish", {})
            if not f1:
                continue
            lines.append(
                f"| `{op}` "
                f"| {f1['mean']:.4f} ± {f1['std']:.4f} "
                f"| {rc['mean']:.4f} ± {rc['std']:.4f} |"
            )
        sm_f1 = cfgs["augmented"]["smart_attacker"]["f1_phish"]
        sm_r = cfgs["augmented"]["smart_attacker"]["recall_phish"]
        lines.append(
            f"| _all combined (smart test)_ "
            f"| {sm_f1['mean']:.4f} ± {sm_f1['std']:.4f} "
            f"| {sm_r['mean']:.4f} ± {sm_r['std']:.4f} |"
        )

    # Seed inventory
    lines.append("\n## Seeds in this aggregation\n")
    for cfg_name, c in cfgs.items():
        lines.append(f"- **{cfg_name}**: seeds {c['seeds']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    runs = _collect_runs()
    print("Discovered runs:")
    for cfg, reports in runs.items():
        print(f"  {cfg}: {len(reports)} seed(s) — {[r['_seed'] for r in reports]}")
    if not any(runs.values()):
        print("\nNo per-seed runs found. Make sure you trained with results/{baseline|augmented}_seed*/eval_report.json present.")
        sys.exit(1)
    agg = aggregate(runs)

    out_json = RESULTS_DIR / "aggregated.json"
    out_json.write_text(json.dumps(agg, indent=2))
    print(f"\nWrote {out_json}")

    md = render_markdown(agg)
    out_md = RESULTS_DIR / "aggregated.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}\n")
    print(md)


if __name__ == "__main__":
    main()
