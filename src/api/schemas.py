"""
Pydantic request/response models for the Kosuri-GPR-Seq2Expr REST API.

Schemas
-------
* ``PredictRequest``     -- accepts promoter / RBS by sequence OR by name lookup
* ``PredictResponse``    -- GPR prediction + uncertainty band
* ``HealthResponse``     -- simple liveness probe
* ``ModelInfoResponse``   -- which model is loaded, supported features
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    """A single expression prediction request.

    Provide *either*:
      * ``promoter_sequence`` and ``rbs_sequence``  (raw DNA strings)
      * ``promoter_name``   and ``rbs_name``        (Kosuri IDs, requires sd01/sd02)

    Raw sequence input is preferred for arbitrary / synthetic designs.
    """
    promoter_sequence: Optional[str] = Field(
        None, description="Raw promoter DNA (5'->3'). Trailing GGCGCGCC prefix is stripped."
    )
    rbs_sequence: Optional[str] = Field(
        None, description="Raw RBS DNA (5'->3'). Trailing CATATG is stripped."
    )
    promoter_name: Optional[str] = Field(
        None, description="Kosuri promoter ID (e.g. 'apFAB67'). Requires the sd01 lookup table."
    )
    rbs_name: Optional[str] = Field(
        None, description="Kosuri RBS ID (e.g. 'B0034_RBS'). Requires the sd02 lookup table."
    )

    @model_validator(mode="after")
    def _check_one_mode(self):
        seq_mode = self.promoter_sequence is not None and self.rbs_sequence is not None
        name_mode = self.promoter_name is not None and self.rbs_name is not None
        if not (seq_mode or name_mode):
            raise ValueError(
                "Provide either (promoter_sequence AND rbs_sequence) "
                "or (promoter_name AND rbs_name)."
            )
        return self


class PredictResponse(BaseModel):
    expression_log2: float = Field(..., description="Predicted protein expression, log2 scale")
    expression_raw_estimate: float = Field(
        ..., description="Back-transformed estimate in the original fluorescence units"
    )
    uncertainty_std: float = Field(
        ..., description="GPR predictive standard deviation (log2 scale)"
    )
    uncertainty_band_95_low: float = Field(..., description="95% band lower bound (log2)")
    uncertainty_band_95_high: float = Field(..., description="95% band upper bound (log2)")
    model: str = Field(..., description="Model name used for prediction")
    model_run: str = Field(..., description="Run tag the model was loaded from")


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_run: str


class ModelInfoResponse(BaseModel):
    model_run: str
    n_features: int
    feature_cols: List[str]
    kernel_description: Optional[str] = None
    train_test_split: Optional[str] = None
    n_train_samples: Optional[int] = None
    test_r2_mean: Optional[float] = None
    test_r2_std: Optional[float] = None
