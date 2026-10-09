"""
CLI runner for the simulated active-learning experiment.

Loads the Kosuri paired dataset, runs a fixed-seed AL trajectory for each
acquisition function (random / variance / ei_max), and writes:

    <out_dir>/learning_curves.png        <- headline plot
    <out_dir>/history_<acq>.csv         <- per-iteration metrics
    <out_dir>/comparison_summary.csv    <- final-iteration summary table
    <out_dir>/summary.json              <- same data in JSON form

Usage
-----
    python src/scripts/run_simulate_al.py
    python src/scripts/run_simulate_al.py --acquisitions random,variance \\
        --initial-fraction 0.3 --query-size 50 --n-iterations 5 \\
        --max-train 800 --tag my_al
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd

# scripts/run_simulate_al.py -> feature_engineering/  (2 levels up from src/scripts)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.simulate_al import (
    ALIteration,
    simulate_active_learning,
    plot_learning_curves,
)
from src.models.gpr import load_json, make_gpr_factory
from src.utils.config import config_path, RUNS_DIR


def main():
    p = argparse.ArgumentParser(description="Run simulated active learning with GPR.")
    p.add_argument("--data", default=None, help="Path to a paired CSV.")
    p.add_argument("--tag", default="simulate_al",
                   help="Run tag; outputs go to src/runs/<tag>/.")
    p.add_argument("--acquisitions", default="random,variance,ei_max",
                   help="Comma-separated acquisitions to compare.")
    p.add_argument("--initial-fraction", type=float, default=0.30,
                   help="Fraction of the train pool used as the initial labelled set.")
    p.add_argument("--query-size", type=int, default=50,
                   help="Number of new samples added per iteration.")
    p.add_argument("--n-iterations", type=int, default=5,
                   help="Number of AL rounds.")
    p.add_argument("--test-fraction", type=float, default=0.20,
                   help="Fraction of data held out as the *fixed* test set.")
    p.add_argument("--max-train", type=int, default=800,
                   help="Cap on rows used to fit GPR per iteration (subsample if larger).")
    p.add_argument("--ei-xi", type=float, default=0.01,
                   help="Exploration-vs-exploitation trade-off for EI.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_cfg = load_json(config_path("data"))
    features_cfg = load_json(config_path("features"))
    model_cfg = load_json(config_path("model"))

    rel = args.data or data_cfg["paths"]["paired_csv"]
    if not os.path.isabs(rel):
        paired_csv = os.path.join(PROJECT_ROOT, rel)
    else:
        paired_csv = rel
    if not os.path.exists(paired_csv):
        raise FileNotFoundError(f"Paired CSV not found: {paired_csv}")

    df = pd.read_csv(paired_csv)
    feature_cols = list(features_cfg["promoter"]) + list(features_cfg["rbs"])
    X = df[feature_cols].values.astype(np.float64) if False else __import__("numpy").array(
        df[feature_cols].values, dtype=__import__("numpy").float64)
    y = df[data_cfg["columns"]["target_log2"]].values.astype(np.float64)
    groups = df[data_cfg["columns"]["group_col"]].values

    print(f"[MAIN] paired_csv = {paired_csv}")
    print(f"[MAIN] N={len(df)}, n_features={len(feature_cols)}, "
          f"n_groups={len(set(groups))}")

    gpr_factory = make_gpr_factory(
        model_cfg["gpr"], random_state=args.seed,
    )

    selected = [s.strip() for s in args.acquisitions.split(",") if s.strip()]
    histories: Dict[str, List[ALIteration]] = {}
    for acq in selected:
        print(f"\n========== Acquisition: {acq} ==========")
        histories[acq] = simulate_active_learning(
            X=X, y=y, groups=groups,
            gpr_factory=gpr_factory,
            acquisition=acq,
            initial_fraction=args.initial_fraction,
            query_size=args.query_size,
            n_iterations=args.n_iterations,
            test_fraction=args.test_fraction,
            max_train=args.max_train,
            ei_xi=args.ei_xi,
            random_state=args.seed,
        )

    # ---- Save artefacts ----
    out_dir = os.path.join(RUNS_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    for acq, hist in histories.items():
        # Per-acq CSV
        pd.DataFrame([h.__dict__ for h in hist]).to_csv(
            os.path.join(out_dir, f"history_{acq}.csv"), index=False)
        first, last = hist[0], hist[-1]
        summary_rows.append({
            "acquisition": acq,
            "initial_test_r2": first.test_r2,
            "final_test_r2": last.test_r2,
            "r2_gain": last.test_r2 - first.test_r2,
            "final_test_rmse": last.test_rmse,
            "final_test_pearson": last.test_pearson,
            "n_labeled_final": last.n_labeled,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    print("\n" + "=" * 70)
    print("ACTIVE LEARNING SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # JSON dump
    json_payload = {
        "settings": {
            "acquisitions": selected,
            "initial_fraction": args.initial_fraction,
            "query_size": args.query_size,
            "n_iterations": args.n_iterations,
            "test_fraction": args.test_fraction,
            "max_train": args.max_train,
            "ei_xi": args.ei_xi,
            "seed": args.seed,
        },
        "histories": {a: [h.__dict__ for h in hist] for a, hist in histories.items()},
        "summary": summary_rows,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    # Plot
    plot_learning_curves(histories, os.path.join(out_dir, "learning_curves.png"))
    print(f"\n[MAIN] artefacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
