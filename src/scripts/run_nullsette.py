"""
CLI runner for the Nullsette OOD test.

Loads the most-recent trained GPR (or any explicit path), extracts features
from the GLM-Nullsette-Benchmark Nullsette files, and produces:

    <out_dir>/ood_distribution.png        (histogram of pred_mean + pred_std)
    <out_dir>/per_variant_summary.csv     (per-translocation stats)
    <out_dir>/normal_predictions.csv      (raw predictions on nonmutant.txt)
    <out_dir>/summary.json                (MW + KS test statistics)

Usage
-----
    python src/scripts/run_nullsette.py
    python src/scripts/run_nullsette.py --model path/to/final_gpr.joblib \\
        --scaler path/to/scaler.joblib --out src/runs/<tag>_nullsette
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

# Resolve project root from __file__
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.nullsette import run_nullsette_test
from src.utils.config import config_path
from src.models.gpr import load_json


def find_latest_gpr_run(runs_dir: str) -> Optional[str]:
    """Return the most-recently-modified directory under ``runs/`` whose name
    ends with ``__gpr`` (i.e. one of our GPR run outputs)."""
    candidates = sorted(glob.glob(os.path.join(runs_dir, "*__gpr")),
                           key=lambda p: os.path.getmtime(p))
    return candidates[-1] if candidates else None


def main():
    ap = argparse.ArgumentParser(description="Run Nullsette OOD test on a trained GPR.")
    ap.add_argument("--model", default=None,
                    help="Path to final_gpr.joblib (default: latest in src/runs/*__gpr/)")
    ap.add_argument("--scaler", default=None,
                    help="Path to scaler.joblib (default: same dir as --model)")
    ap.add_argument("--kosuri-text-dir", default=None,
                    help="Directory containing nonmutant.txt + mutant_translocation*.txt "
                         "(default: data/processed/processed_data/kosuri)")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: <model dir>/../<tag>_nullsette)")
    args = ap.parse_args()

    features_cfg = load_json(config_path("features"))
    feature_cols: List[str] = list(features_cfg["promoter"]) + list(features_cfg["rbs"])

    runs_dir = os.path.join(PROJECT_ROOT, "src", "runs")
    if args.model is None:
        latest = find_latest_gpr_run(runs_dir)
        if latest is None:
            raise SystemExit("No __gpr run found under src/runs/. Train a GPR first.")
        model_path = os.path.join(latest, "models", "final_gpr.joblib")
        scaler_path = os.path.join(latest, "models", "scaler.joblib")
        print(f"[MAIN] using latest GPR run: {latest}")
    else:
        model_path = args.model
        scaler_path = args.scaler or os.path.join(os.path.dirname(args.model), "scaler.joblib")

    if args.kosuri_text_dir is None:
        kosuri_text_dir = os.path.join(PROJECT_ROOT, "data", "processed_data",
                                       "processed_data", "kosuri")
    else:
        kosuri_text_dir = args.kosuri_text_dir

    if args.out is None:
        run_dir = os.path.dirname(os.path.dirname(model_path))   # .../runs/<tag>__gpr/models/.. -> <tag>__gpr
        out_dir = os.path.join(runs_dir, run_dir.rsplit("__gpr", 1)[0] + "__nullsette")
    else:
        out_dir = args.out

    print(f"[MAIN] model   = {model_path}")
    print(f"[MAIN] scaler  = {scaler_path}")
    print(f"[MAIN] textdir = {kosuri_text_dir}")
    print(f"[MAIN] outdir  = {out_dir}")
    print(f"[MAIN] n_features = {len(feature_cols)}")

    run_nullsette_test(
        model_path=model_path,
        scaler_path=scaler_path,
        kosuri_text_dir=kosuri_text_dir,
        out_dir=out_dir,
        feature_cols=feature_cols,
    )


if __name__ == "__main__":
    main()

