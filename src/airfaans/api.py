"""Engineer-facing inference API with an explicit model boundary."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    case_id: str = Field(min_length=1)
    reynolds: float = Field(gt=0)
    angle_of_attack_deg: float = Field(ge=-30, le=30)


class PredictionSummary(BaseModel):
    case_id: str
    model_id: str
    node_count: int
    field_means: dict[str, float]
    lift_coefficient: float | None
    drag_coefficient: float | None
    mean_uncertainty: float | None
    model_latency_ms: float
    latency_ms: float
    evidence_label: str


def create_app(predictor=None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("Install AIRFAANS with the 'api' extra.") from exc
    app = FastAPI(title="AIRFAANS", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": "loaded" if predictor else "not_configured"}

    @app.post("/v1/predict", response_model=PredictionSummary)
    def predict(request: PredictionRequest):
        if predictor is None:
            raise HTTPException(status_code=503, detail="No trained checkpoint is configured.")
        started = perf_counter()
        try:
            result = predictor(request)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionSummary(
            **result,
            case_id=request.case_id,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    return app


def predictor_from_environment():
    checkpoint = os.environ.get("AIRFAANS_CHECKPOINT")
    dataset_root = os.environ.get("AIRFAANS_DATASET_ROOT")
    release_manifest = os.environ.get("AIRFAANS_RELEASE_MANIFEST")
    if not checkpoint or not dataset_root or not release_manifest:
        return None
    from airfaans.inference import CheckpointPredictor

    return CheckpointPredictor(
        Path(checkpoint),
        Path(dataset_root),
        Path(release_manifest),
        os.environ.get("AIRFAANS_DEVICE", "cpu"),
    )


app = create_app(predictor_from_environment())
