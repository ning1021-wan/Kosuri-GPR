"""Streamlit dashboard for the Kosuri-GPR-Seq2Expr pipeline.

Pages
-----
1. Overview           -- project summary + headline numbers
2. Predict            -- interactive expression predictor
3. Model comparison   -- RF vs GPR vs ... from comparison.csv
4. OOD (Nullsette)    -- show distribution plot + KS/MW test stats
5. About              -- data sources + caveats

Run locally
-----------
    streamlit run src/ui/streamlit_app.py

Or via compose:
    docker compose up ui
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import streamlit as st


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)


# ----------------------------------------------------------------------
# Reusable feature extraction (same as the API)
# ----------------------------------------------------------------------
def _gc(s):
    return (s.count("G") + s.count("C")) / max(1, len(s))


def _at(s):
    return (s.count("A") + s.count("T")) / max(1, len(s))


def _skew(s):
    g, c = s.count("G"), s.count("C")
    return (g - c) / max(1, g + c)


def _complexity(s):
    if len(s) < 3:
        return 0.0
    return len(set(s[i:i + 3] for i in range(len(s) - 2))) / (len(s) - 2)


def _mfe(s):
    return -_gc(s) * 15.0


def _strip(s):
    return str(s or "").strip().upper().replace(" ", "")


def _feature_dict(prom_seq, rbs_seq):
    p = _strip(prom_seq)
    r = _strip(rbs_seq)
    return {
        "prom__gc_content": _gc(p),
        "prom__at_content": _at(p),
        "prom__gc_skew": _skew(p),
        "prom__length": float(len(p)),
        "prom__complexity": _complexity(p),
        "prom__mfe": _mfe(p),
        "prom__score_minus35": (6.0 - sum(1 for a, b in zip(p[:6], "TTGACA") if a != b)) / 6.0 if len(p) >= 6 else 0.0,
        "prom__score_minus10": (6.0 - sum(1 for a, b in zip(p[10:16], "TATAAT") if a != b)) / 6.0 if len(p) >= 16 else 0.0,
        "prom__gc_motif_density": (p.count("GC") + p.count("CG")) / max(1, len(p)),
        "prom__at_motif_density": (p.count("AT") + p.count("TA")) / max(1, len(p)),
        "rbs__gc_content": _gc(r),
        "rbs__at_content": _at(r),
        "rbs__gc_skew": _skew(r),
        "rbs__length": float(len(r)),
        "rbs__complexity": _complexity(r),
        "rbs__mfe": _mfe(r),
        "rbs__sd_best_score": (6.0 - sum(1 for a, b in zip(r[:6], "AGGAGG") if a != b)) / 6.0 if len(r) >= 6 else 0.0,
        "rbs__sd_present": 1.0 if "AGGAG" in r else 0.0,
        "rbs__spacer_gc": _gc(r[-7:]) if len(r) >= 7 else _gc(r),
        "rbs__spacer_length": float(max(0, len(r) - 6)),
    }


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
RUNS_DIR = os.path.join(PROJECT_ROOT, "src", "runs")


@st.cache_resource
def load_latest_gpr():
    candidates = sorted(glob.glob(os.path.join(RUNS_DIR, "*__gpr")),
                        key=lambda p: os.path.getmtime(p))
    if not candidates:
        return None
    import joblib
    run_dir = candidates[-1]
    model = joblib.load(os.path.join(run_dir, "models", "final_gpr.joblib"))
    scaler = joblib.load(os.path.join(run_dir, "models", "scaler.joblib"))
    snap = json.load(open(os.path.join(run_dir, "config_snapshot.json")))
    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    feature_cols = list(snap["features"]["promoter"]) + list(snap["features"]["rbs"])
    return model, scaler, feature_cols, os.path.basename(run_dir).replace("__gpr", ""), metrics


def list_comparison_files():
    return sorted(glob.glob(os.path.join(RUNS_DIR, "*", "comparison.csv")))


def list_nullsette_runs():
    return sorted(glob.glob(os.path.join(RUNS_DIR, "*__nullsette")))


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
def page_overview():
    st.title("Kosuri-GPR-Seq2Expr")
    st.markdown(
        "**End-to-end promoter x RBS expression prediction** using Gaussian "
        "Process Regression trained on the Kosuri 2013 (PNAS) dataset."
    )
    bundle = load_latest_gpr()
    if bundle is None:
        st.error("No trained GPR model found. Run `python src/scripts/run_experiment.py --models gpr` first.")
        return
    model, scaler, feature_cols, run_tag, metrics = bundle
    cv = metrics.get("cv", {}).get("test", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test R^2 (mean)", f"{cv.get('r2_mean', float('nan')):.3f}")
    col2.metric("Test R^2 (std)", f"{cv.get('r2_std', float('nan')):.3f}")
    col3.metric("Test RMSE", f"{cv.get('rmse_mean', float('nan')):.2f}")
    col4.metric("N samples", f"{metrics.get('n_samples'):,}")
    st.markdown(
        f"**Run tag**: `{run_tag}`  |  **Features**: {len(feature_cols)}  |  "
        f"**CV**: GroupKFold by `promoter_id` ({metrics.get('n_folds')} folds)\n\n"
        f"**Kernel**: `{model.kernel_}`\n\n"
        "The model gives both a *mean prediction* and a *standard deviation* (GPR's "
        "built-in uncertainty quantification). This is what makes it useful for "
        "out-of-distribution detection and uncertainty-aware active learning."
    )


def page_predict():
    st.title("Predict expression")
    bundle = load_latest_gpr()
    if bundle is None:
        st.error("No model loaded.")
        return
    model, scaler, feature_cols, run_tag, metrics = bundle
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("Input")
        mode = st.radio("Input mode", ["Raw sequences", "Kosuri IDs (lookup)"])
        if mode == "Raw sequences":
            prom_seq = st.text_area("Promoter sequence (5' -> 3')",
                                    "TTGACATCAGGAAAATTTTTCTG",
                                    height=80)
            rbs_seq = st.text_area("RBS sequence (5' -> 3')",
                                   "AGGAGGCAATATTTGATTTCATATC",
                                   height=80)
        else:
            st.info("This requires the sd01/sd02 Kosuri lookup tables to be present in `data/processed_data/`.")
            prom_seq = st.text_input("Kosuri promoter name", "apFAB67")
            rbs_seq = st.text_input("Kosuri RBS name", "B0034_RBS")
    with col_r:
        st.subheader("Output")
        if st.button("Predict", type="primary"):
            try:
                fdict = _feature_dict(prom_seq, rbs_seq)
                X = np.array([[fdict[c] for c in feature_cols]], dtype=np.float64)
                Xs = scaler.transform(X)
                mean, std = model.predict(Xs, return_std=True)
                mean = float(mean[0]); std = float(std[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return
            st.metric("Predicted expression (log2)", f"{mean:.3f}")
            st.metric("Predicted expression (raw, ~a.u.)", f"{2 ** mean - 1:.0f}")
            st.metric("Predictive std (log2)", f"{std:.3f}")
            st.markdown(f"**95% band**: {mean - 1.96 * std:.2f} -> {mean + 1.96 * std:.2f}")
            with st.expander("Computed features"):
                st.json(fdict)


def page_comparison():
    st.title("Model comparison (GroupKFold)")
    files = list_comparison_files()
    if not files:
        st.info("No comparison.csv files under src/runs/.")
        return
    rows = []
    for f in files:
        run_tag = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            rows.append({"run": run_tag, **r.to_dict()})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    if not df.empty and "test_r2_mean" in df.columns:
        st.bar_chart(df, x="model", y="test_r2_mean",
                     color="run", horizontal=True)


def page_ood():
    st.title("Nullsette OOD analysis")
    runs = list_nullsette_runs()
    if not runs:
        st.info("No __nullsette runs found. Run `python src/scripts/run_nullsette.py` first.")
        return
    paths = [os.path.join(r, "ood_distribution.png") for r in runs
             if os.path.exists(os.path.join(r, "ood_distribution.png"))]
    if not paths:
        st.warning("No ood_distribution.png found in any Nullsette run.")
        return
    chosen = st.selectbox("Nullsette run",
                                [os.path.basename(os.path.dirname(p)) for p in paths])
    img_path = os.path.join(RUNS_DIR, chosen, "ood_distribution.png")
    summary_path = os.path.join(RUNS_DIR, chosen, "summary.json")
    per_var_path = os.path.join(RUNS_DIR, chosen, "per_variant_summary.csv")
    st.image(img_path, caption=f"OOD distribution: {chosen}")
    if os.path.exists(summary_path):
        s = json.load(open(summary_path))
        st.subheader("Distribution tests")
        cols = st.columns(2)
        for i, k in enumerate(["pred_mean", "pred_std"]):
            with cols[i]:
                st.markdown(f"**{k}**")
                d = s["distribution_tests"][k]
                st.markdown(
                    f"| | Normal | Mutant |\n"
                    f"|---|---|---|\n"
                    f"| mean | {d['normal_mean']:.3f} | {d['mutant_mean']:.3f} |\n"
                    f"| std  | {d['normal_std']:.3f}  | {d['mutant_std']:.3f}  |\n"
                    f"| KS p-value   | {d['ks_pvalue']:.2e} |  |\n"
                    f"| MW p-value   | {d['mw_pvalue']:.2e} |  |\n"
                )
    if os.path.exists(per_var_path):
        st.subheader("Per-variant stats")
        st.dataframe(pd.read_csv(per_var_path), hide_index=True)


def page_about():
    st.title("About")
    st.markdown(
        "## Data source\n\n"
        "Kosuri et al. 2013, PNAS 110:14024-14029. 12,563 paired promoter x RBS "
        "combinations measured in E. coli for protein, mRNA and DNA.\n\n"
        "- Promoter table: data/processed_data/sd01.xlsx (112 promoters)\n"
        "- RBS table: data/processed_data/sd02.xlsx (111 RBSs)\n"
        "- Paired measurements: data/processed_data/sd03.xlsx (12,655 rows)\n"
        "- Virtual mutants (Nullsettes): data/processed_data/processed_data/kosuri/mutant_translocation*.txt\n\n"
        "## Why GroupKFold?\n\n"
        "A random K-fold over 12k pairs puts the same promoter into both train and test sets. "
        "The model can memorise promoter-specific signal and report an unrealistically good R^2. "
        "We split by promoter_id so each fold's test set contains unseen promoters.\n\n"
        "## What is Nullsette?\n\n"
        "A virtual mutant whose promoter and RBS have been translocated within the expression "
        "cassette. These sequences never existed in evolution and have no experimental ground truth. "
        "They probe whether the model's predictive uncertainty correctly inflates on OOD inputs.\n\n"
        "## Caveats\n\n"
        "- GPR was trained on a 1,500-row subsample (full 11.5k would take >30min per fold with "
        "  scikit-learn's exact kernel). Numbers in the Overview tab reflect this subset.\n"
        "- Feature set is intentionally small (20 hand-crafted features).\n"
        "- No active-learning loop yet -- Nullsette is a post-hoc OOD probe."
    )


PAGES = {
    "Overview": page_overview,
    "Predict": page_predict,
    "Model comparison": page_comparison,
    "OOD (Nullsette)": page_ood,
    "About": page_about,
}


def main():
    st.set_page_config(page_title="Kosuri-GPR-Seq2Expr",
                       page_icon="", layout="wide")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()