import pandas as pd
from fastapi.testclient import TestClient

from src.app import app
from src.features import ROOT_DIR, SAMPLE_SUB_FILE


def test_predictions_csv_format():
    pred_path = ROOT_DIR / "predictions.csv"
    assert pred_path.exists(), "predictions.csv must exist at repository root"
    sub = pd.read_csv(SAMPLE_SUB_FILE)
    pred = pd.read_csv(pred_path)
    assert list(pred.columns) == ["claim_id", "score"]
    assert len(pred) == len(sub) == 2252
    assert (pred["claim_id"] == sub["claim_id"]).all()
    assert pred["score"].notna().all()
    assert ((pred["score"] >= 0.0) & (pred["score"] <= 1.0)).all()


def test_api_health_and_predict():
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["model_loaded"] is True

        samples = client.get("/sample-claims").json()["samples"]
        assert len(samples) == 3

        res = client.post(
            "/predict",
            json={
                "claim_id": "WC799999",
                "submitted_at": "2026-07-22 00:05",
                "partner_id": "SP3129",
                "sku": "KH-MG-03",
                "product_serial": "KH395147108",
                "days_since_purchase": 90,
                "claim_amount_inr": 1995.0,
                "photo_attached": "N",
                "partner_inspected": "N",
                "claim_description": "motor not running",
                "inspector_note": None,
                "customer_prior_claims": 2,
                "source": "crm",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert 0.0 <= body["fraud_score"] <= 1.0
        assert body["risk_band"] in {"HIGH", "MEDIUM", "LOW"}
        assert len(body["reasons"]) >= 1
        assert body["partner_context"]["partner_id"] == "SP3129"

