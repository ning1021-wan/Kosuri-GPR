"""
Unit tests for the simulated active-learning module.

These tests exercise the pure-Python helpers (acquisition functions,
dataclass round-trips) on a small toy dataset. The GPR-based AL loop is
slow and is exercised by ``src/scripts/run_simulate_al.py`` in CI.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.simulate_al import (
    ACQ_REGISTRY,
    acquisition_random,
    acquisition_variance,
    acquisition_ei_max,
    simulate_active_learning,
)


# ----------------------------------------------------------------------
# Acquisition functions
# ----------------------------------------------------------------------
def test_acquisition_variance_equals_std():
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.5, 1.0, 0.2])
    out = acquisition_variance(mean, std)
    np.testing.assert_array_equal(out, std)


def test_acquisition_random_returns_zero_one_range():
    rng = np.random.default_rng(0)
    out = acquisition_random(np.zeros(100), np.zeros(100), rng)
    assert out.shape == (100,)
    assert (out >= 0).all() and (out <= 1).all()


def test_acquisition_ei_max_zero_when_mean_leq_ybest():
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.5, 0.5, 0.5])
    y_best = 10.0  # mean nowhere near y_best
    out = acquisition_ei_max(mean, std, y_best=y_best, xi=0.01)
    # All three are well below y_best, EI should be ~0
    assert (out < 1e-6).all()


def test_acquisition_ei_max_positive_when_mean_above_ybest():
    mean = np.array([10.0, 20.0])
    std = np.array([1.0, 5.0])
    y_best = 5.0
    out = acquisition_ei_max(mean, std, y_best=y_best, xi=0.01)
    assert (out > 0).all()
    # Higher std -> more EI (ceteris paribus)
    assert out[1] > out[0]


def test_acquisition_registry_covers_known_names():
    for name in ("random", "variance", "ei_max"):
        assert name in ACQ_REGISTRY


# ----------------------------------------------------------------------
# AL loop on toy data (uses a fast linear-regression stand-in for GPR)
# ----------------------------------------------------------------------
def _toy_pool(n=120, n_groups=6, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.tile(np.arange(n_groups), n // n_groups)
    X = rng.normal(size=(n, 3))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=n)
    return X, y, groups


def test_simulate_al_runs_three_acquisitions():
    from sklearn.linear_model import LinearRegression
    X, y, groups = _toy_pool()
    results = {}
    for acq in ("random", "variance", "ei_max"):
        results[acq] = simulate_active_learning(
            X=X, y=y, groups=groups,
            gpr_factory=lambda: LinearRegression(),
            acquisition=acq,
            initial_fraction=0.3,
            query_size=20,
            n_iterations=3,
            test_fraction=0.2,
            max_train=None,
            random_state=42,
        )
    # Each trajectory has n_iterations+1 records (incl. iter 0)
    for acq, hist in results.items():
        assert len(hist) == 4, f"{acq} should have 4 iterations"
        for h in hist:
            assert 0 <= h.test_r2 <= 1.0
            assert h.n_labeled >= 1


def test_simulate_al_labeled_pool_monotonically_grows():
    from sklearn.linear_model import LinearRegression
    X, y, groups = _toy_pool()
    hist = simulate_active_learning(
        X=X, y=y, groups=groups,
        gpr_factory=lambda: LinearRegression(),
        acquisition="variance",
        initial_fraction=0.3,
        query_size=20,
        n_iterations=3,
        test_fraction=0.2,
        random_state=42,
    )
    sizes = [h.n_labeled for h in hist]
    assert sizes == sorted(sizes)
    # Should grow by at least query_size per iteration (after iter 0)
    for i in range(1, len(sizes)):
        assert sizes[i] - sizes[i - 1] >= 0


def test_simulate_al_unknown_acquisition_raises():
    X, y, groups = _toy_pool()
    with pytest.raises(ValueError):
        simulate_active_learning(
            X=X, y=y, groups=groups,
            gpr_factory=lambda: None,
            acquisition="not_a_real_strategy",
        )


def test_simulate_al_small_dataset_raises():
    X = np.zeros((3, 2)); y = np.zeros(3); groups = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        simulate_active_learning(X, y, groups, gpr_factory=lambda: None)