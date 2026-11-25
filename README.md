# Kestrel Home — Warranty Claim Fraud Review

Capacity-aware warranty fraud triage service for Kestrel Home Appliances. Scores incoming claims, routes the top **40 claims/month** to the investigation desk, and explains each flag in plain English—with **zero paid API keys required**.

## Quick Start (Clean Machine)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Train model, run temporal validation, and generate predictions.csv
python -m src.train

# 2. Start the web service & UI
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser to use the interactive Claims Desk UI, or call the API directly:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "WC711887",
    "submitted_at": "2026-07-22 00:05",
    "partner_id": "SP3129",
    "sku": "KH-MG-03",
    "product_serial": "KH395147108",
    "days_since_purchase": 120,
    "claim_amount_inr": 1995.0,
    "photo_attached": "N",
    "partner_inspected": "N",
    "claim_description": "motor not running",
    "inspector_note": null,
    "customer_prior_claims": 2,
    "source": "crm"
  }' | python3 -m json.tool
```

## Endpoints
- `GET /` — Single-screen Claims Desk UI with sample high/medium/low-risk presets
- `POST /predict` — Scores a single JSON claim and returns `fraud_score`, `risk_band`, `recommended_action`, `expected_net_value_inr`, `reasons`, and `partner_context`
- `GET /metrics` — Returns temporal validation metrics
- `GET /health` — Health check

## Deliverables in Repository
- `predictions.csv` — Scored predictions for all 2,252 rows of `test_unlabelled.csv`
- `MEMO.md` — One-page non-technical memo to Ritu Deshpande
- `EVIDENCE.md` — Temporal validation, rupee economics, and failure modes
- `SUBMISSION.md` — Submission form notes (expected score, decisions, AI tools used/discarded)
- `src/` — Feature engineering (`features.py`), training/validation (`train.py`), FastAPI app (`app.py`)
- `static/index.html` — Single-screen UI
- `tests/test_service.py` — Automated test suite (`pytest`)
