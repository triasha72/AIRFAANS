"""Engineer-facing inference API with an explicit model boundary."""

from __future__ import annotations

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
    lift_coefficient: float
    drag_coefficient: float
    mean_uncertainty: float
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
        result = predictor(request)
        return PredictionSummary(
            **result,
            case_id=request.case_id,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    return app


app = create_app()
