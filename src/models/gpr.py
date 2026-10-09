"""
Config-driven Gaussian Process Regression for promoter x RBS expression prediction.

Why this exists
---------------
The original ``train_gpr.py`` (kept in this commit for history) hard-coded:

  * the feature list (two hand-picked columns);
  * the train/test split (random 80/20, no group awareness);
  * the kernel + hyper-parameters;
  * the target column.

This made it impossible to:

  * swap Kosuri's grouped data in without rewriting the script;
  * report grouped-CV metrics to defend against leakage;
  * A/B test different feature subsets without editing Python.

This module replaces that with a config-driven runner:

  * Reads ``configs/data.json``, ``configs/features.json``, ``configs/model.json``.
  * Builds (X, y, groups) from a paired CSV.
  * Optionally uses ``GroupKFold`` (default) or a held-out random split.
  * Persists per-fold + aggregated metrics to ``runs/<run_name>/``.
  * Saves the scaler + final (full-data) GPR model + per-fold models for later
    uncertainty-driven queries (active learning).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
    Matern,
)
from sklearn.preprocessing import StandardScaler

from src.models.group_cv import (
    CVResult,
    compute_metrics,
    run_group_cv,
    build_xyg,
)


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Kernel factory
# ----------------------------------------------------------------------
def build_kernel(spec: Dict):
    """Instantiate a kernel from a config dict.

    Supported ``spec["type"]`` values:
      * ``"rbf_white"``  -> ConstantKernel * RBF + WhiteKernel (default, small data)
      * ``"matern_white"`` -> ConstantKernel * Matern(nu=2.5) + WhiteKernel (smoother)
    """
    ktype = spec.get("type", "rbf_white")

    if ktype == "rbf_white":
        k = ConstantKernel(spec.get("constant", 1.0)) * \
            RBF(length_scale=spec.get("length_scale", 1.0)) + \
            WhiteKernel(
                noise_level=spec.get("noise_level", 0.1),
                noise_level_bounds=tuple(spec.get("noise_bounds", [1e-5, 1.0])),
            )
    elif ktype == "matern_white":
        k = ConstantKernel(spec.get("constant", 1.0)) * \
            Matern(length_scale=spec.get("length_scale", 1.0), nu=2.5) + \
            WhiteKernel(
                noise_level=spec.get("noise_level", 0.1),
                noise_level_bounds=tuple(spec.get("noise_bounds", [1e-5, 1.0])),
            )
    else:
        raise ValueError(f"Unknown kernel type: {ktype!r}")
    return k


# ----------------------------------------------------------------------
# Model factory (zero-arg, used by GroupCVRunner)
# ----------------------------------------------------------------------
def make_gpr_factory(spec: Dict, random_state: int = 42):
    """Return a callable that builds an *unfitted* GPR with the given spec."""
    kernel = build_kernel(spec.get("kernel", {}))

    def factory():
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=spec.get("n_restarts_optimizer", 10),
            alpha=spec.get("alpha", 0.01),
            normalize_y=spec.get("normalize_y", False),
            random_state=random_state,
        )
    return factory


# ----------------------------------------------------------------------
# Feature assembly: promoter features + RBS features into one design matrix
# ----------------------------------------------------------------------
def assemble_features(
    df: pd.DataFrame,
    promoter_features: List[str],
    rbs_features: List[str],
    promoter_id_col: str = "promoter_id",
    rbs_id_col: str = "rbs_id",
) -> Tuple[np.ndarray, List[str]]:
    """Return (X, feature_names) where X is the concatenation of promoter
    feature columns and RBS feature columns per row.

    Both sets must already exist in ``df`` (the suffixes have already been
    stripped during dataset construction). For the iGEM demo the columns are
    just called e.g. ``gc_content`` (no suffix), so we assume unique names.
    """
    missing = [c for c in promoter_features + rbs_features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in DataFrame: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    X = df[promoter_features + rbs_features].values.astype(np.float64)
    return X, promoter_features + rbs_features


# ----------------------------------------------------------------------
# Run entry: GPR via GroupKFold + final model on all data
# ----------------------------------------------------------------------
@dataclass
class GPRRunResult:
    cv: CVResult
    scaler: StandardScaler
    final_model: GaussianProcessRegressor
    metrics_summary: Dict
    run_dir: str
    config: Dict


def run_gpr(
    paired_csv: str,
    data_cfg: Dict,
    features_cfg: Dict,
    model_cfg: Dict,
    run_name: str = "gpr_run",
    use_group_cv: bool = True,
    save_models: bool = True,
) -> GPRRunResult:
    """Train a GPR with config-driven features + group-aware CV.

    Parameters
    ----------
    paired_csv
        Path to a paired dataset (e.g. ``src/data/paired_dataset.csv``).
    data_cfg, features_cfg, model_cfg
        Parsed JSON configs.
    run_name
        Sub-directory under ``src/runs/`` to write artefacts to.
    use_group_cv
        If True, use ``GroupKFold`` (recommended for Kosuri-style data).
        If False, fall back to a random KFold (only for A/B comparison).
    """
    from src.utils.config import RUNS_DIR, RUN_MODEL_DIR, RUN_PLOT_DIR

    # ----- 1. Load + assemble -----
    df = pd.read_csv(paired_csv)
    print(f"[GPR] loaded {paired_csv}: {df.shape}")

    promoter_features = features_cfg["promoter"]
    rbs_features = features_cfg["rbs"]
    target_col = data_cfg["columns"]["target_log2"]
    group_col = data_cfg["columns"]["group_col"]

    X, feature_names = assemble_features(
        df, promoter_features, rbs_features,
        promoter_id_col=data_cfg["columns"]["promoter_id_col"],
        rbs_id_col=data_cfg["columns"]["rbs_id_col"],
    )
    y = df[target_col].values.astype(np.float64)
    groups = df[group_col].values

    # Drop NaN rows before scaling
    keep = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(groups.astype(float))
    X, y, groups = X[keep], y[keep], groups[keep]
    print(f"[GPR] usable rows: {len(y)} (features: {X.shape[1]})")

    # ----- 2. Scale features (NOT target -- GPR handles y internally) -----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----- 3. GroupKFold CV -----
    gpr_spec = model_cfg["gpr"]
    eval_spec = model_cfg["evaluation"]
    factory = make_gpr_factory(gpr_spec, random_state=gpr_spec.get("random_state", 42))

    cv_result = run_group_cv(
        model_factory=factory,
        X=X_scaled,
        y=y,
        groups=groups,
        n_splits=eval_spec["cv_n_splits"],
        model_name="gpr",
        feature_cols=feature_names,
        target_col=target_col,
        group_col=group_col,
        return_std=True,
    )

    # ----- 4. Train on all data for deployment -----
    final_model = factory()
    final_model.fit(X_scaled, y)
    print(f"[GPR] final kernel: {final_model.kernel_}")

    # ----- 5. Aggregate + persist -----
    metrics_summary = {
        "model": "gpr",
        "n_samples": int(y.shape[0]),
        "n_features": int(X.shape[1]),
        "n_folds": cv_result.n_folds,
        "cv": cv_result.aggregated_metrics(),
        "feature_cols": feature_names,
        "kernel": str(final_model.kernel_),
        "kernel_log_marginal_likelihood": float(final_model.log_marginal_likelihood_value_),
    }

    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    # Save metrics
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    # Save per-fold table
    cv_result.to_dataframe().to_csv(os.path.join(run_dir, "per_fold_metrics.csv"), index=False)

    if save_models:
        joblib.dump(scaler, os.path.join(run_dir, "models", "scaler.joblib"))
        joblib.dump(final_model, os.path.join(run_dir, "models", "final_gpr.joblib"))
        # Persist per-fold models too (useful for uncertainty-based queries)
        for i, fold in enumerate(cv_result.folds):
            # Refit on the same train split so the model matches what the CV ran
            f = factory()
            f.fit(X_scaled[fold.train_index], y[fold.train_index])
            joblib.dump(f, os.path.join(run_dir, "models", f"fold_{i:02d}_gpr.joblib"))
        print(f"[GPR] models saved under {run_dir}/models/")

    # Config snapshot for reproducibility
    config_snapshot = {"data": data_cfg, "features": features_cfg, "model": model_cfg}
    with open(os.path.join(run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    return GPRRunResult(
        cv=cv_result,
        scaler=scaler,
        final_model=final_model,
        metrics_summary=metrics_summary,
        run_dir=run_dir,
        config=config_snapshot,
    )
