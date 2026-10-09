"""
Run the full benchmark: GPR + all enabled baselines, with GroupKFold.

Usage
-----
    python src/scripts/run_experiment.py --demo
    python src/scripts/run_experiment.py --data path/to/paired.csv --tag my_run
    python src/scripts/run_experiment.py --models gpr --max-train 400    # fast GPR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from typing import Dict, List

import pandas as pd

# scripts/run_experiment.py -> feature_engineering/  (2 levels up from src/scripts)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.models.gpr import run_gpr, load_json
from src.models.baselines import run_baseline, BASELINE_REGISTRY
from src.utils.config import config_path, RUNS_DIR


def resolve_data_path(args, data_cfg: Dict) -> str:
    if args.data:
        return args.data
    rel = data_cfg["paths"]["paired_csv"]
    if not os.path.isabs(rel):
        return os.path.join(PROJECT_ROOT, rel)
    return rel


def ensure_demo_dataset(paired_csv: str) -> str:
    if os.path.exists(paired_csv):
        return paired_csv
    print(f"[MAIN] {paired_csv} not found. Building synthetic demo dataset...")
    build_script = os.path.join(PROJECT_ROOT, "src", "data", "build_demo.py")
    subprocess.check_call([sys.executable, build_script])
    return paired_csv


def maybe_subsample(paired_csv: str, max_train: int) -> str:
    """If max_train is set, subsample the paired CSV and write to a sibling path."""
    if max_train is None or max_train <= 0:
        return paired_csv
    df = pd.read_csv(paired_csv)
    if len(df) <= max_train:
        print(f"[MAIN] max_train={max_train} >= dataset size {len(df)}; no subsampling.")
        return paired_csv
    sub_path = paired_csv.replace(".csv", f"_n{max_train}.csv")
    df.sample(n=max_train, random_state=42).reset_index(drop=True).to_csv(sub_path, index=False)
    print(f"[MAIN] subsampled to {max_train} rows -> {sub_path}")
    return sub_path


def build_comparison_table(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]["cv"]
        row = {
            "model": r["name"],
            "n_samples": r["metrics"]["n_samples"],
            "n_features": r["metrics"]["n_features"],
            "n_folds": r["metrics"]["n_folds"],
            "test_r2_mean": m["test"]["r2_mean"],
            "test_r2_std": m["test"]["r2_std"],
            "test_rmse_mean": m["test"]["rmse_mean"],
            "test_rmse_std": m["test"]["rmse_std"],
            "test_mae_mean": m["test"]["mae_mean"],
            "test_pearson_mean": m["test"]["pearson_mean"],
            "train_r2_pooled": m["train"]["r2"],
            "train_rmse_pooled": m["train"]["rmse"],
            "train_mae_pooled": m["train"]["mae"],
        }
        if "kernel_log_marginal_likelihood" in r["metrics"]:
            row["log_marginal_likelihood"] = r["metrics"]["kernel_log_marginal_likelihood"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Run GPR + baselines with GroupKFold.")
    p.add_argument("--data", default=None,
                   help="Path to a paired CSV (overrides data.json).")
    p.add_argument("--tag", default="main",
                   help="Run tag; outputs go to src/runs/<tag>/.")
    p.add_argument("--models", default="gpr,random_forest",
                   help="Comma-separated list. Available: gpr, random_forest, xgboost, svr.")
    p.add_argument("--demo", action="store_true",
                   help="Run on the synthetic 48x48 demo dataset.")
    p.add_argument("--max-train", type=int, default=None,
                   help="Optional cap on training rows. Useful when running GPR "
                        "on > 2k rows; GPR is O(n^3).")
    args = p.parse_args()

    data_cfg = load_json(config_path("data"))
    features_cfg = load_json(config_path("features"))
    model_cfg = load_json(config_path("model"))

    paired_csv = resolve_data_path(args, data_cfg)
    paired_csv = ensure_demo_dataset(paired_csv)
    paired_csv = maybe_subsample(paired_csv, args.max_train)

    if not os.path.exists(paired_csv):
        raise FileNotFoundError(f"Paired dataset not found: {paired_csv}")

    print(f"[MAIN] paired_csv = {paired_csv}")
    print(f"[MAIN] tag         = {args.tag}")
    print(f"[MAIN] models      = {args.models}")

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []

    if "gpr" in selected:
        run_name = f"{args.tag}__gpr"
        r = run_gpr(
            paired_csv=paired_csv,
            data_cfg=data_cfg,
            features_cfg=features_cfg,
            model_cfg=model_cfg,
            run_name=run_name,
        )
        results.append({"name": "gpr", "metrics": r.metrics_summary, "run_dir": r.run_dir})

    for name in selected:
        if name == "gpr":
            continue
        if name not in BASELINE_REGISTRY:
            print(f"[MAIN] WARN: unknown model {name!r}; skipping.")
            continue
        spec = model_cfg["baselines"].get(name)
        if spec is None:
            print(f"[MAIN] WARN: no spec for {name!r} in model.json; skipping.")
            continue
        if not spec.get("enabled", True):
            print(f"[MAIN] {name} is disabled in model.json; skipping.")
            continue
        try:
            r = run_baseline(
                name=name,
                spec=spec,
                paired_csv=paired_csv,
                data_cfg=data_cfg,
                features_cfg=features_cfg,
                model_cfg=model_cfg,
                run_name=f"{args.tag}__{name}",
            )
            results.append({"name": name, "metrics": r.metrics_summary, "run_dir": r.run_dir})
        except Exception as e:
            print(f"[MAIN] ERROR running {name}: {e}")
            continue

    if not results:
        print("[MAIN] No models ran successfully. Aborting.")
        sys.exit(1)

    comparison_df = build_comparison_table(results)
    print("\n" + "=" * 70)
    print("COMPARISON (test split, GroupKFold)")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    tag_dir = os.path.join(RUNS_DIR, args.tag)
    os.makedirs(tag_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(tag_dir, "comparison.csv"), index=False)
    with open(os.path.join(tag_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[MAIN] comparison saved to {tag_dir}/comparison.{{csv,json}}")


if __name__ == "__main__":
    main()
