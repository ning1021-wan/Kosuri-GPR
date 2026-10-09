"""End-to-end smoke test for the FastAPI app. Skipped if fastapi/httpx missing."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False


pytestmark = pytest.mark.skipif(
    not HAVE_FASTAPI,
    reason="fastapi/httpx not installed; run inside Docker (python:3.12)",
)


def test_health_endpoint():
    from src.api.main import create_app
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body and "model_loaded" in body


def test_predict_by_sequence():
    from src.api.main import create_app
    app = create_app()
    client = TestClient(app)
    payload = {
        "promoter_sequence": "TTGACATCAGGAAAATTTTTCTG",
        "rbs_sequence": "AGGAGGCAATATTTGATTTCATATC",
    }
    r = client.post("/predict", json=payload)
    if r.status_code == 503:
        pytest.skip("no GPR model loaded; run_experiment first")
    assert r.status_code == 200
    body = r.json()
    for key in ("expression_log2", "expression_raw_estimate",
                "uncertainty_std", "uncertainty_band_95_low",
                "uncertainty_band_95_high", "model", "model_run"):
        assert key in body
    assert body["model"] == "gpr"


def test_predict_validation():
    from src.api.main import create_app
    app = create_app()
    client = TestClient(app)
    r = client.post("/predict", json={})
    assert r.status_code == 422