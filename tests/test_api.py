from fastapi.testclient import TestClient

from airfaans.api import create_app


def test_api_fails_closed_without_checkpoint():
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok", "model": "not_configured"}
    response = client.post(
        "/v1/predict",
        json={"case_id": "case", "reynolds": 1_000_000, "angle_of_attack_deg": 4},
    )
    assert response.status_code == 503


def test_api_returns_checkpoint_prediction_contract():
    def predictor(_request):
        return {
            "model_id": "test-model",
            "node_count": 10,
            "field_means": {"pressure": 0.5},
            "lift_coefficient": None,
            "drag_coefficient": None,
            "mean_uncertainty": None,
            "model_latency_ms": 1.0,
            "evidence_label": "test_fixture",
        }

    client = TestClient(create_app(predictor))
    response = client.post(
        "/v1/predict",
        json={"case_id": "case", "reynolds": 1_000_000, "angle_of_attack_deg": 4},
    )
    assert response.status_code == 200
    assert response.json()["field_means"] == {"pressure": 0.5}
    assert response.json()["lift_coefficient"] is None


def test_api_maps_predictor_input_errors_to_422():
    def predictor(_request):
        raise ValueError("metadata mismatch")

    client = TestClient(create_app(predictor))
    response = client.post(
        "/v1/predict",
        json={"case_id": "case", "reynolds": 1_000_000, "angle_of_attack_deg": 4},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "metadata mismatch"
