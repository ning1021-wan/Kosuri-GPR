"""
Simulated active learning on the Kosuri paired data.

Why this exists
---------------
Active learning (AL) is a core value proposition of Gaussian Processes in
biology: instead of running the full ~12k wet-lab experiments upfront, you
can use GPR's predictive uncertainty to pick the next batch of constructs
that will most reduce uncertainty or find the strongest expression.

This module simulates that loop on the Kosuri 2013 paired dataset:

    1. Hold out 20% of the data as a fixed test set (never labelled).
    2. From the remaining 80%, draw an initial labelled pool (random).
    3. Train a GPR on the labelled pool.
    4. Score every row in the unlabelled pool with an acquisition function:
         - random  : uniform random selection (baseline)
         - variance: top-K by GPR predictive std  (pure exploration)
         - ei_max  : Expected Improvement for *maximising* expression
                     (most useful when the goal is to find the strongest
                     promoter x RBS combination)
    5. Move the top-K rows into the labelled pool.
    6. Re-evaluate on the test set.
    7. Repeat for n_iterations.

The headline output is a learning-curve plot showing how test R^2 grows
as we add labelled samples -- a curve that climbs faster for the GPR-
driven strategies is the evidence that "uncertainty quantification helps
us spend fewer experiments".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Acquisition functions
# ----------------------------------------------------------------------
def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    from scipy.stats import pearsonr
    return float(pearsonr(a, b)[0])


def acquisition_variance(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Pure exploration: rank by GPR predictive std."""
    return std


def acquisition_random(mean: np.ndarray, std: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """Uniform random scores."""
    return rng.random(len(mean))


def acquisition_ei_max(mean: np.ndarray, std: np.ndarray,
                        y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement for *maximising* expression.

    Standard EI formulation flipped to a maximisation objective: positive
    when predicted mean exceeds current best by at least xi.
    """
    out = np.zeros_like(mean, dtype=np.float64)
    nonzero = std > 0
    z = np.zeros_like(mean)
    z[nonzero] = (mean[nonzero] - y_best - xi) / std[nonzero]
    out[nonzero] = std[nonzero] * (z[nonzero] * norm.cdf(z[nonzero]) + norm.pdf(z[nonzero]))
    return out


ACQ_REGISTRY = {
    "random": acquisition_random,
    "variance": acquisition_variance,
    "ei_max": acquisition_ei_max,
}


# ----------------------------------------------------------------------
# AL driver
# ----------------------------------------------------------------------
@dataclass
class ALIteration:
    iteration: int
    n_labeled: int
    n_unlabeled: int
    test_r2: float
    test_rmse: float
    test_mae: float
    test_pearson: float
    acquisition: str


def simulate_active_learning(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    gpr_factory: Callable[[], object],
    acquisition: str = "variance",
    initial_fraction: float = 0.30,
    query_size: int = 50,
    n_iterations: int = 5,
    test_fraction: float = 0.20,
    max_train: Optional[int] = None,
    ei_xi: float = 0.01,
    random_state: int = 42,
) -> List[ALIteration]:
    """Run a single simulated AL trajectory.

    The function returns a list of ``ALIteration`` records, one per iteration
    (including iteration 0 = the initial labelled pool before any queries).
    The caller is responsible for plotting / saving.
    """
    if acquisition not in ACQ_REGISTRY:
        raise ValueError(f"Unknown acquisition {acquisition!r}; "
                         f"choose from {list(ACQ_REGISTRY)}")
    rng = np.random.default_rng(random_state)
    n = len(y)
    if n < 10:
        raise ValueError("Need at least 10 rows for AL simulation.")

    # ---- 1. Hold out a stratified test set (by group) ----
    # GroupKFold(n_splits=k) does NOT mean k*test_fraction of data is held
    # out -- with uneven group sizes the first fold can swallow an extra
    # group. Instead, take ``n_test_groups`` random groups to test so that
    # the actual test fraction tracks ``test_fraction`` more reliably.
    unique_groups = np.unique(groups)
    n_test_groups = max(1, int(len(unique_groups) * test_fraction))
    perm = rng.permutation(unique_groups)
    test_groups = set(perm[:n_test_groups].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    test_idx = np.where(test_mask)[0]
    pool_idx = np.where(~test_mask)[0]

    # ---- 2. Initial labelled pool: random sample of the pool ----
    pool_idx = np.array(pool_idx)
    rng.shuffle(pool_idx)
    n_initial = max(query_size, int(len(pool_idx) * initial_fraction))
    labeled_idx = pool_idx[:n_initial].tolist()
    unlabeled_idx = pool_idx[n_initial:].tolist()

    history: List[ALIteration] = []

    for it in range(n_iterations + 1):
        # Optionally subsample the labelled set to keep GPR tractable
        train_idx = labeled_idx
        if max_train is not None and len(train_idx) > max_train:
            sub = rng.choice(train_idx, size=max_train, replace=False)
            train_idx = sub.tolist()

        X_train = X[train_idx]
        y_train = y[train_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        gp = gpr_factory()
        gp.fit(X_train_scaled, y_train)

        # ---- Evaluate on held-out test ----
        X_test_scaled = scaler.transform(X[test_idx])
        y_pred, y_std = gp.predict(X_test_scaled, return_std=True)
        history.append(ALIteration(
            iteration=it,
            n_labeled=len(labeled_idx),
            n_unlabeled=len(unlabeled_idx),
            test_r2=float(r2_score(y[test_idx], y_pred)),
            test_rmse=float(np.sqrt(mean_squared_error(y[test_idx], y_pred))),
            test_mae=float(mean_absolute_error(y[test_idx], y_pred)),
            test_pearson=_safe_pearson(y[test_idx], y_pred),
            acquisition=acquisition if it > 0 else "initial",
        ))
        print(
            f"[AL:{acquisition}] iter {it:2d} | labelled={len(labeled_idx):4d} | "
            f"test R2={history[-1].test_r2:.3f} RMSE={history[-1].test_rmse:.3f} "
            f"Pearson={history[-1].test_pearson:.3f}",
            flush=True,
        )

        if it == n_iterations or len(unlabeled_idx) == 0:
            break

        # ---- Score the unlabelled pool ----
        X_unl_scaled = scaler.transform(X[unlabeled_idx])
        y_pred_unl, y_std_unl = gp.predict(X_unl_scaled, return_std=True)
        if acquisition == "random":
            scores = ACQ_REGISTRY[acquisition](y_pred_unl, y_std_unl, rng)
        elif acquisition == "variance":
            scores = ACQ_REGISTRY[acquisition](y_pred_unl, y_std_unl)
        elif acquisition == "ei_max":
            y_best = float(y_train.max())
            scores = ACQ_REGISTRY[acquisition](y_pred_unl, y_std_unl, y_best, xi=ei_xi)
        else:
            scores = ACQ_REGISTRY[acquisition](y_pred_unl, y_std_unl)

        # ---- Top-K query ----
        k = min(query_size, len(unlabeled_idx))
        top_k_local = np.argsort(scores)[-k:][::-1]
        new_labelled_local = [unlabeled_idx[i] for i in top_k_local]
        labeled_idx = labeled_idx + new_labelled_local
        unlabeled_idx = [i for i in unlabeled_idx if i not in set(new_labelled_local)]

    return history


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_learning_curves(
    histories: Dict[str, List[ALIteration]],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for acq, hist in histories.items():
        ns = [h.n_labeled for h in hist]
        r2s = [h.test_r2 for h in hist]
        rmses = [h.test_rmse for h in hist]
        axes[0].plot(ns, r2s, marker="o", label=acq)
        axes[1].plot(ns, rmses, marker="o", label=acq)
    axes[0].set_xlabel("N labelled samples")
    axes[0].set_ylabel("Held-out test R$^2$")
    axes[0].set_title("Active learning: predictive accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("N labelled samples")
    axes[1].set_ylabel("Held-out test RMSE")
    axes[1].set_title("Active learning: predictive error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
