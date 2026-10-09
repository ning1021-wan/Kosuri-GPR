"""Tests for the GPR pipeline's reusable helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.nullsette import (
    parse_fasta,
    split_cassette_by_atg,
    HEADER_RE,
)
from src.models.group_cv import (
    build_xyg,
    compute_metrics,
    run_group_cv,
)


def _toy_df():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "y": rng.normal(size=n),
        "group": np.repeat(np.arange(n) % 6, 10),
    })


def test_build_xyg_drops_nan_rows():
    df = _toy_df().copy()
    df.loc[0, "f1"] = np.nan
    X, y, g = build_xyg(df, ["f1", "f2"], "y", "group")
    assert X.shape[0] == 59
    assert y.shape == (59,)
    assert g.shape == (59,)


def test_compute_metrics_smoke():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 2.0, 2.9, 4.2])
    m = compute_metrics(y, p)
    for k in ("r2", "rmse", "mae", "pearson", "n"):
        assert k in m
    assert m["n"] == 4
    assert m["r2"] > 0.95


def test_run_group_cv_smoke():
    from sklearn.linear_model import LinearRegression
    df = _toy_df()
    X, y, g = build_xyg(df, ["f1", "f2"], "y", "group")
    result = run_group_cv(
        model_factory=lambda: LinearRegression(),
        X=X, y=y, groups=g,
        n_splits=3,
        model_name="lin",
        return_std=False,
    )
    assert result.n_folds == 3
    assert all("r2" in f.test_metrics for f in result.folds)


SAMPLE_FASTA = (
    '>"apFAB67"_"Invitrogen_RBS"\n'
    "TTGACATCAGGAAAATTTTTCTGCATAATTATTTCATATCACAAAATTAAGAGGTATATAATGCGTAAAGG\n"
    '>"apFAB66"_"B0034_RBS"\n'
    "TTGACATCAGGAAAATTTTTCTGTATAATAGATTCATCTCAAAAAGAGGAGAAATTAATGCGTAAAGG\n"
)


def test_parse_fasta(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text(SAMPLE_FASTA, encoding="utf-8")
    records = parse_fasta(str(p))
    assert ("apFAB67", "Invitrogen_RBS") in records
    assert records[("apFAB67", "Invitrogen_RBS")].startswith("TTGACATCAGG")


def test_split_cassette_by_atg():
    seq = "TTGACATCAGGAAAATTTTTCTGCATAATTATTTCATATCACAAAATTAAGAGGTATATAATGCGTAAAGG"
    prom, rbs = split_cassette_by_atg(seq, prom_window=25, rbs_window=25)
    assert prom is not None and rbs is not None
    assert len(prom) == 25 and len(rbs) == 25


def test_split_cassette_by_atg_too_short():
    seq = "ATGAA"
    prom, rbs = split_cassette_by_atg(seq)
    assert prom is None and rbs is None


def test_split_cassette_by_atg_no_atg():
    seq = "TTTTTTTTTTTTTT"
    prom, rbs = split_cassette_by_atg(seq)
    assert prom is None and rbs is None


def test_header_re_matches():
    line = '>"apFAB67"_"Invitrogen_RBS"'
    m = HEADER_RE.match(line)
    assert m is not None
    assert m.group("prom") == "apFAB67"
    assert m.group("rbs") == "Invitrogen_RBS"