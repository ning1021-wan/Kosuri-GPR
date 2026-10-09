"""
FastAPI service for the Kosuri-GPR-Seq2Expr pipeline.

Endpoints
---------
GET  /                  -- service banner
GET  /health            -- liveness probe (returns 200 with model_load status)
GET  /model/info        -- details on the currently-loaded model
POST /predict           -- expression prediction from sequences or Kosuri IDs
GET  /runs              -- list available model runs in src/runs/

The service loads the *latest* GPR run at startup (controlled by the
``MODEL_RUN_TAG`` env var or auto-detection via the most recent ``__gpr`` dir).

Run locally:
    docker compose up api
or
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Make src.* importable when launched via ``uvicorn src.api.main:app``
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


# ----------------------------------------------------------------------
# 1. Feature extraction (shared with the rest of the pipeline)
# ----------------------------------------------------------------------
def _gc(s: str) -> float:
    return (s.count("G") + s.count("C")) / max(1, len(s))


def _at(s: str) -> float:
    return (s.count("A") + s.count("T")) / max(1, len(s))


def _skew(s: str) -> float:
    g, c = s.count("G"), s.count("C")
    return (g - c) / max(1, g + c)


def _complexity(s: str) -> float:
    if len(s) < 3:
        return 0.0
    return len(set(s[i:i + 3] for i in range(len(s) - 2))) / (len(s) - 2)


def _mfe(s: str) -> float:
    return -_gc(s) * 15.0


def _strip(s: str) -> str:
    return str(s or "").strip().upper().replace(" ", "")


def _feature_dict(prom_seq: str, rbs_seq: str) -> Dict[str, float]:
    """Compute the 20 features used by the GPR model."""
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
# 2. Kosuri sequence lookup (sd01/sd02)
# ----------------------------------------------------------------------
class KosuriLookup:
    """Lazily build promoter/rbs name -> sequence lookups from the sd01/sd02 tables.

    The lookup is built the first time ``.lookup()`` is called. Subsequent calls
    hit an in-memory dict.
    """

    def __init__(self, xlsx_dir: Optional[str] = None):
        self.xlsx_dir = xlsx_dir or os.path.join(PROJECT_ROOT, "data", "processed_data")
        self._promoters: Optional[Dict[str, str]] = None
        self._rbss: Optional[Dict[str, str]] = None

    @staticmethod
    def _strip_quotes(name: str) -> str:
        s = str(name).strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.strip()

    def _ensure_loaded(self) -> None:
        if self._promoters is None or self._rbss is None:
            try:
                df_p = pd.read_excel(os.path.join(self.xlsx_dir, "sd01.xlsx"),
                                     engine="openpyxl")
                df_p["promoter_name"] = df_p["Promoter"].apply(self._strip_quotes)
                df_p["prom_sequence"] = df_p["Sequence"].astype(str).str.upper().str.replace(" ", "", regex=False)
                # Strip GGCGCGCC prefix (matches kosuri_loader behaviour)
                df_p["prom_sequence"] = df_p["prom_sequence"].apply(
                    lambda s: s[8:] if s.startswith("GGCGCGCC") else s
                )
                self._promoters = dict(zip(df_p["promoter_name"], df_p["prom_sequence"]))
            except Exception as e:
                print(f"[API] WARN: promoter lookup disabled ({e})")
                self._promoters = {}
            try:
                df_r = pd.read_excel(os.path.join(self.xlsx_dir, "sd02.xlsx"),
                                     engine="openpyxl")
                df_r["rbs_name"] = df_r["RBS"].apply(self._strip_quotes)
                df_r["rbs_sequence"] = df_r["Sequence"].astype(str).str.upper().str.replace(" ", "", regex=False)
                # Strip CATATG suffix
                df_r["rbs_sequence"] = df_r["rbs_sequence"].apply(
                    lambda s: s[:-6] if s.endswith("CATATG") else s
                )
                self._rbss = dict(zip(df_r["rbs_name"], df_r["rbs_sequence"]))
            except Exception as e:
                print(f"[API] WARN: RBS lookup disabled ({e})")
                self._rbss = {}

    def lookup(self, prom_name: str, rbs_name: str) -> Tuple[str, str]:
        self._ensure_loaded()
        if prom_name not in self._promoters:
            raise HTTPException(404, f"Unknown promoter_id: {prom_name!r}")
        if rbs_name not in self._rbss:
            raise HTTPException(404, f"Unknown rbs_id: {rbs_name!r}")
        return self._promoters[prom_name], self._rbss[rbs_name]


# ----------------------------------------------------------------------
# 3. Model registry
# ----------------------------------------------------------------------
def _latest_gpr_run(runs_dir: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(runs_dir, "*__gpr")),
                        key=lambda p: os.path.getmtime(p))
    return candidates[-1] if candidates else None


def _load_model(run_dir: str) -> Tuple[object, object, List[str], Dict]:
    """Load scaler + GPR + feature_cols + metrics from a run directory."""
    scaler = joblib.load(os.path.join(run_dir, "models", "scaler.joblib"))
    model = joblib.load(os.path.join(run_dir, "models", "final_gpr.joblib"))
    snap = json.load(open(os.path.join(run_dir, "config_snapshot.json")))
    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    feature_cols = list(snap["features"]["promoter"]) + list(snap["features"]["rbs"])
    return model, scaler, feature_cols, metrics


# ----------------------------------------------------------------------
# 4. App factory
# ----------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="Kosuri-GPR-Seq2Expr",
        description=(
            "Expression prediction for promoter x RBS combinations using a "
            "Gaussian Process trained on the Kosuri 2013 (PNAS) dataset."
        ),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    runs_dir = os.path.join(PROJECT_ROOT, "src", "runs")
    tag = os.environ.get("MODEL_RUN_TAG")
    if tag:
        run_dir = os.path.join(runs_dir, tag)
    else:
        run_dir = _latest_gpr_run(runs_dir) or ""
    if not run_dir or not os.path.exists(os.path.join(run_dir, "models", "final_gpr.joblib")):
        print(f"[API] WARN: no usable GPR run found under {runs_dir}/*__gpr/")
        state = {"model": None, "scaler": None, "feature_cols": [], "run_tag": "(none)",
                 "metrics": {}, "lookup": KosuriLookup()}
    else:
        model, scaler, feature_cols, metrics = _load_model(run_dir)
        run_tag = os.path.basename(run_dir).replace("__gpr", "")
        state = {"model": model, "scaler": scaler, "feature_cols": feature_cols,
                 "run_tag": run_tag, "metrics": metrics,
                 "lookup": KosuriLookup()}
        print(f"[API] loaded model from {run_dir}  ({run_tag})")

    @app.get("/", include_in_schema=False)
    def _root():
        return {"service": "Kosuri-GPR-Seq2Expr", "endpoints": [
            "/health", "/model/info", "/predict", "/runs"
        ]}

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_loaded=state["model"] is not None,
            model_run=state["run_tag"],
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info():
        m = state["metrics"]
        return ModelInfoResponse(
            model_run=state["run_tag"],
            n_features=len(state["feature_cols"]),
            feature_cols=state["feature_cols"],
            kernel_description=str(state["model"].kernel_) if state["model"] is not None else None,
            train_test_split="GroupKFold (promoter_id)",
            n_train_samples=m.get("n_samples"),
            test_r2_mean=(m.get("cv", {}).get("test", {}).get("r2_mean")
                          if m else None),
            test_r2_std=(m.get("cv", {}).get("test", {}).get("r2_std")
                         if m else None),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if state["model"] is None:
            raise HTTPException(503, "No model loaded. Train a GPR first.")
        # Resolve sequences
        if req.promoter_sequence and req.rbs_sequence:
            prom_seq = _strip(req.promoter_sequence)
            rbs_seq = _strip(req.rbs_sequence)
        else:
            prom_seq, rbs_seq = state["lookup"].lookup(req.promoter_name, req.rbs_name)

        # Compute features in the order the model expects
        fdict = _feature_dict(prom_seq, rbs_seq)
        X = np.array([[fdict[c] for c in state["feature_cols"]]], dtype=np.float64)
        Xs = state["scaler"].transform(X)
        mean, std = state["model"].predict(Xs, return_std=True)
        mean = float(mean[0]); std = float(std[0])
        return PredictResponse(
            expression_log2=mean,
            expression_raw_estimate=float(2 ** mean - 1.0),
            uncertainty_std=std,
            uncertainty_band_95_low=mean - 1.96 * std,
            uncertainty_band_95_high=mean + 1.96 * std,
            model="gpr",
            model_run=state["run_tag"],
        )

    @app.get("/runs")
    def list_runs():
        out = []
        for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
            if not os.path.isdir(d):
                continue
            tag = os.path.basename(d)
            meta_path = os.path.join(d, "metrics.json")
            if not os.path.exists(meta_path):
                continue
            try:
                m = json.load(open(meta_path))
            except Exception:
                continue
            out.append({
                "run": tag,
                "n_samples": m.get("n_samples"),
                "test_r2_mean": m.get("cv", {}).get("test", {}).get("r2_mean"),
                "test_rmse_mean": m.get("cv", {}).get("test", {}).get("rmse_mean"),
            })
        return out

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
