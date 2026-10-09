"""
Group-aware cross-validation utilities.

Why this exists
---------------
The original ``train_gpr.py`` used a plain ``train_test_split`` followed by
``KFold``. For promoter x RBS data this is *unsafe*: the same promoter appears
in both the train and test partitions, so the model can memorise promoter-
specific signal and the test metrics look unrealistically good.

Kosuri 2013-style paired data carries 114 unique promoter ids (and 111 RBS ids).
A promoter-aware split must keep all RBS of a held-out promoter unseen.
``GroupKFold`` (or a manual grouped split) is the right primitive.

This module centralises:

  * Construction of a ``GroupKFold`` from a config dict.
  * A reusable ``GroupCVRunner`` that fits a model per fold and returns per-fold
    + aggregated metrics (R^2, RMSE, MAE, Pearson r).
  * Helper to build an ``(X, y, groups)`` triple from a DataFrame given the
    feature/target/group column lists.

Everything here is dependency-light (numpy, pandas, scikit-learn) so it can be
imported by the GPR, baseline, and active-learning modules without circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r, with NaN-guards for degenerate inputs."""
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metric bundle."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson": _pearson(y_true, y_pred),
        "n": int(len(y_true)),
    }


# ----------------------------------------------------------------------
# Build X, y, groups from a DataFrame
# ----------------------------------------------------------------------
def build_xyg(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    group_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y, groups) and guard against missing columns + NaN rows.

    NaN rows in *any* feature column or the target are dropped (and the same
    indices are removed from groups). This keeps the GPR happy (it can't
    handle NaN inputs) while still being explicit about it.
    """
    missing = [c for c in feature_cols + [target_col, group_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    sub = df[list(feature_cols) + [target_col, group_col]].copy()
    n_before = len(sub)
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    n_after = len(sub)
    if n_after < n_before:
        print(f"[GROUP_CV] dropped {n_before - n_after} rows with NaN/inf")

    X = sub[list(feature_cols)].values.astype(np.float64)
    y = sub[target_col].values.astype(np.float64)
    groups = sub[group_col].values
    return X, y, groups


# ----------------------------------------------------------------------
# Cross-validation runner
# ----------------------------------------------------------------------
@dataclass
class FoldResult:
    fold: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    y_train: np.ndarray
    y_train_pred: np.ndarray
    y_train_std: Optional[np.ndarray]
    y_test: np.ndarray
    y_test_pred: np.ndarray
    y_test_std: Optional[np.ndarray]
    train_index: np.ndarray
    test_index: np.ndarray


@dataclass
class CVResult:
    model_name: str
    feature_cols: List[str]
    target_col: str
    group_col: str
    folds: List[FoldResult] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        """Per-fold mean +/- std for the test split + a pooled-train metric."""
        keys = ["r2", "rmse", "mae", "pearson", "n"]
        out: Dict[str, Dict[str, float]] = {"train": {}, "test": {}}

        # Train side: concatenate across folds
        if self.folds:
            y_tr = np.concatenate([f.y_train for f in self.folds])
            y_tr_pred = np.concatenate([f.y_train_pred for f in self.folds])
            pooled_train = compute_metrics(y_tr, y_tr_pred)
            for k in keys:
                out["train"][k] = pooled_train.get(k, float("nan"))

        # Test side: report per-fold mean / std
        for k in keys:
            if k == "n":
                out["test"][k] = int(sum(f.test_metrics["n"] for f in self.folds))
            else:
                vals = [f.test_metrics[k] for f in self.folds]
                out["test"][f"{k}_mean"] = float(np.nanmean(vals))
                out["test"][f"{k}_std"] = float(np.nanstd(vals))
        return out

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            for split in ("train", "test"):
                row = {"fold": f.fold, "split": split}
                row.update({k: v for k, v in getattr(self, f"_FoldResult__{split}_metrics", {}).items()})
                rows.append(row)
        # Fallback: use the structured fields
        rows = []
        for f in self.folds:
            for split in ("train", "test"):
                row = {"fold": f.fold, "split": split}
                m = f.train_metrics if split == "train" else f.test_metrics
                row.update(m)
                rows.append(row)
        return pd.DataFrame(rows)


def make_group_kfold(n_splits: int, groups: np.ndarray) -> GroupKFold:
    """Construct a ``GroupKFold`` after a sanity check on the number of groups.

    ``GroupKFold`` requires ``n_splits <= n_unique_groups``. We clamp silently
    and warn so the user gets a sensible split even if the config is over-zealous.
    """
    n_unique = len(np.unique(groups))
    if n_splits > n_unique:
        print(f"[GROUP_CV] WARN: n_splits={n_splits} > n_unique_groups={n_unique}; clamping.")
        n_splits = max(2, n_unique)
    return GroupKFold(n_splits=n_splits)


def run_group_cv(
    model_factory: Callable[[], "object"],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    model_name: str = "model",
    feature_cols: Optional[List[str]] = None,
    target_col: str = "y",
    group_col: str = "group",
    return_std: bool = True,
) -> CVResult:
    """Run ``GroupKFold`` cross-validation.

    Parameters
    ----------
    model_factory
        A zero-arg callable that returns an *unfitted* estimator with a
        ``fit(X, y)`` method and a ``predict(X, return_std=False)`` method.
        For scikit-learn GPR this means calling ``predict(X, return_std=True)``.
    X, y, groups
        Arrays. Lengths must match.
    n_splits
        Number of group-aware folds.
    return_std
        If True, the runner calls ``predict(X, return_std=True)`` and stores the
        standard deviations. Set False for baselines that don't support std.
    """
    cv = make_group_kfold(n_splits, groups)
    result = CVResult(
        model_name=model_name,
        feature_cols=list(feature_cols or []),
        target_col=target_col,
        group_col=group_col,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        print(f"[{model_name}] fold {fold_idx + 1}/{n_splits}: fitting on {len(train_idx)} rows, predicting {len(test_idx)} ...", flush=True)
        model = model_factory()
        model.fit(X_tr, y_tr)

        if return_std and hasattr(model, "predict"):
            try:
                y_tr_pred, y_tr_std = model.predict(X_tr, return_std=True)
                y_te_pred, y_te_std = model.predict(X_te, return_std=True)
            except TypeError:
                # Baselines without return_std
                y_tr_pred = model.predict(X_tr); y_tr_std = None
                y_te_pred = model.predict(X_te); y_te_std = None
        else:
            y_tr_pred = model.predict(X_tr); y_tr_std = None
            y_te_pred = model.predict(X_te); y_te_std = None

        result.folds.append(FoldResult(
            fold=fold_idx,
            train_metrics=compute_metrics(y_tr, y_tr_pred),
            test_metrics=compute_metrics(y_te, y_te_pred),
            y_train=y_tr,
            y_train_pred=y_tr_pred,
            y_train_std=y_tr_std,
            y_test=y_te,
            y_test_pred=y_te_pred,
            y_test_std=y_te_std,
            train_index=train_idx,
            test_index=test_idx,
        ))

        print(
            f"[{model_name}] fold {fold_idx + 1}/{n_splits}: "
            f"train R^2={result.folds[-1].train_metrics['r2']:.4f}  "
            f"test R^2={result.folds[-1].test_metrics['r2']:.4f}  "
            f"test RMSE={result.folds[-1].test_metrics['rmse']:.4f}  "
            f"test n={result.folds[-1].test_metrics['n']}"
        )

    return result

