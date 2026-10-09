"""
Baseline regressors for the Kosuri-GPR-Seq2Expr pipeline.

Why this exists
---------------
The GPR alone tells us nothing about whether GP is the *right* choice for this
data. To make a defensible claim at interview time ("we compared GPR against
strong baselines"), we provide:

  * RandomForestRegressor  -- the default tree baseline; always available.
  * XGBoostRegressor       -- requires ``xgboost`` (stubs if not installed).
  * SVR (RBF kernel)       -- always available; conceptually closest to GPR.

Every baseline goes through the same ``GroupKFold`` runner as the GPR, so the
comparison is apples-to-apples.

Each baseline exposes a ``make_<name>_factory(spec) -> callable`` so the
runner can spawn a new estimator per fold.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.models.group_cv import (
    CVResult,
    run_group_cv,
)


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------
def make_rf_factory(spec: Dict, random_state: int = 42) -> Callable:
    from sklearn.ensemble import RandomForestRegressor

    def factory():
        return RandomForestRegressor(
            n_estimators=spec.get("n_estimators", 300),
            max_depth=spec.get("max_depth", None),
            min_samples_leaf=spec.get("min_samples_leaf", 2),
            n_jobs=spec.get("n_jobs", -1),
            random_state=random_state,
        )
    return factory


# ----------------------------------------------------------------------
# XGBoost (optional)
# ----------------------------------------------------------------------
def make_xgb_factory(spec: Dict, random_state: int = 42) -> Callable:
    try:
        import xgboost  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "XGBoost is not installed. Run `pip install xgboost` to enable "
            "this baseline. (Python 3.14 may not yet have a wheel -- check "
            "https://pypi.org/project/xgboost/#files)"
        ) from e

    def factory():
        return xgboost.XGBRegressor(
            n_estimators=spec.get("n_estimators", 300),
            max_depth=spec.get("max_depth", 6),
            learning_rate=spec.get("learning_rate", 0.05),
            random_state=random_state,
            n_jobs=spec.get("n_jobs", -1),
            verbosity=0,
        )
    return factory


# ----------------------------------------------------------------------
# SVR (RBF)
# ----------------------------------------------------------------------
def make_svr_factory(spec: Dict) -> Callable:
    from sklearn.svm import SVR

    def factory():
        return SVR(
            kernel=spec.get("kernel", "rbf"),
            C=spec.get("C", 1.0),
            epsilon=spec.get("epsilon", 0.05),
            gamma=spec.get("gamma", "scale"),
        )
    return factory


# ----------------------------------------------------------------------
# Factory router
# ----------------------------------------------------------------------
BASELINE_REGISTRY = {
    "random_forest": make_rf_factory,
    "xgboost": make_xgb_factory,
    "svr": make_svr_factory,
}


# ----------------------------------------------------------------------
# Generic runner that mirrors run_gpr() but for any baseline
# ----------------------------------------------------------------------
@dataclass
class BaselineRunResult:
    name: str
    cv: CVResult
    final_model: object
    metrics_summary: Dict
    run_dir: str


def run_baseline(
    name: str,
    spec: Dict,
    paired_csv: str,
    data_cfg: Dict,
    features_cfg: Dict,
    model_cfg: Dict,
    run_name: Optional[str] = None,
) -> BaselineRunResult:
    """Train a single baseline with the same GroupKFold protocol as ``run_gpr``.

    The signature mirrors ``run_gpr`` so the call site can dispatch to either
    one based on the config.
    """
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}. Available: {list(BASELINE_REGISTRY)}")
    if not spec.get("enabled", True):
        raise RuntimeError(f"Baseline {name!r} is disabled in model.json.")

    from src.utils.config import RUNS_DIR
    from src.models.gpr import (
        load_json,
        assemble_features,
    )

    df = pd.read_csv(paired_csv)
    print(f"[BASELINE:{name}] loaded {paired_csv}: {df.shape}")

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

    keep = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(groups.astype(float))
    X, y, groups = X[keep], y[keep], groups[keep]
    print(f"[BASELINE:{name}] usable rows: {len(y)}")

    eval_spec = model_cfg["evaluation"]
    random_state = spec.get("random_state", model_cfg["gpr"].get("random_state", 42))
    factory = BASELINE_REGISTRY[name](spec, random_state=random_state)

    cv_result = run_group_cv(
        model_factory=factory,
        X=X, y=y, groups=groups,
        n_splits=eval_spec["cv_n_splits"],
        model_name=name,
        feature_cols=feature_names,
        target_col=target_col,
        group_col=group_col,
        return_std=False,
    )

    # Final model on all data
    final = factory()
    final.fit(X, y)

    metrics_summary = {
        "model": name,
        "n_samples": int(y.shape[0]),
        "n_features": int(X.shape[1]),
        "n_folds": cv_result.n_folds,
        "cv": cv_result.aggregated_metrics(),
        "feature_cols": feature_names,
    }

    run_name = run_name or f"{name}_run"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    cv_result.to_dataframe().to_csv(os.path.join(run_dir, "per_fold_metrics.csv"), index=False)
    joblib.dump(final, os.path.join(run_dir, "models", f"final_{name}.joblib"))

    return BaselineRunResult(
        name=name,
        cv=cv_result,
        final_model=final,
        metrics_summary=metrics_summary,
        run_dir=run_dir,
    )
