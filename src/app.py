"""FastAPI service for Kestrel Home warranty claim fraud scoring."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.features import (
    CAT_COLS,
    NUM_COLS,
    ROOT_DIR,
    TEST_FILE,
    build_base_features,
    generate_reasons,
    merge_partner_stats,
)
from src.train import METRICS_PATH, MODEL_PATH, _predict_ensemble, train_and_evaluate

STATIC_DIR = ROOT_DIR / "static"
STATE: dict = {}


class ClaimInput(BaseModel):
    claim_id: str = Field(default="WC799999", description="Claim number")
    submitted_at: str = Field(default="2026-07-15 14:30", description="Submission timestamp (IST)")
    partner_id: str = Field(default="SP3129", description="Service partner ID")
    sku: str = Field(default="KH-MG-03", description="Product SKU")
    product_serial: str = Field(default="KH395147108", description="Serial as typed by partner")
    days_since_purchase: float = Field(default=120, ge=0)
    claim_amount_inr: float = Field(default=1995.0, ge=0)
    photo_attached: str = Field(default="N", description="Y or N")
    partner_inspected: str = Field(default="N", description="Y or N")
    claim_description: str = Field(default="motor not running")
    inspector_note: Optional[str] = Field(default=None)
    customer_prior_claims: int = Field(default=2, ge=0)
    source: str = Field(default="crm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        train_and_evaluate()
    STATE["artifact"] = joblib.load(MODEL_PATH)
    STATE["metrics"] = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    yield
    STATE.clear()


app = FastAPI(
    title="Kestrel Home — Warranty Claim Fraud Review",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "artifact" in STATE}


@app.get("/metrics")
def metrics():
    return STATE.get("metrics", {})


FALLBACK_SAMPLES = [
    {
        "label": "High Risk (Under-₹2k Repeat Partner)",
        "claim": {
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
            "inspector_note": None,
            "customer_prior_claims": 2,
            "source": "crm",
        },
    },
    {
        "label": "Medium Risk (Uninspected Claim)",
        "claim": {
            "claim_id": "WC711349",
            "submitted_at": "2026-07-01 01:13",
            "partner_id": "SP3209",
            "sku": "KH-MG-03",
            "product_serial": "kh662573314",
            "days_since_purchase": 419,
            "claim_amount_inr": 1234.0,
            "photo_attached": "N",
            "partner_inspected": "N",
            "claim_description": "remote not working",
            "inspector_note": None,
            "customer_prior_claims": 1,
            "source": "crm",
        },
    },
    {
        "label": "Low Risk (Inspected Normal Claim)",
        "claim": {
            "claim_id": "WC711348",
            "submitted_at": "2026-07-01 00:41",
            "partner_id": "SP3219",
            "sku": "KH-MG-03",
            "product_serial": "KH395147108",
            "days_since_purchase": 577,
            "claim_amount_inr": 3730.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "claim_description": "power button not working",
            "inspector_note": "Minor fault, part swapped",
            "customer_prior_claims": 1,
            "source": "crm",
        },
    },
]


@app.get("/sample-claims")
def sample_claims():
    """Return 3 representative claims (High, Medium, Low risk)."""
    if not TEST_FILE.exists():
        return {"samples": FALLBACK_SAMPLES}
    test_df = pd.read_csv(TEST_FILE)
    samples = []
    for label, cid in [
        ("High Risk (Under-₹2k Repeat Partner)", "WC711887"),
        ("Medium Risk (Uninspected Claim)", "WC711349"),
        ("Low Risk (Inspected Normal Claim)", "WC711348"),
    ]:
        row = test_df[test_df["claim_id"] == cid]
        if not row.empty:
            rec = row.iloc[0].where(pd.notna(row.iloc[0]), None).to_dict()
            samples.append({"label": label, "claim": rec})
    return {"samples": samples or FALLBACK_SAMPLES}


@app.post("/predict")
def predict_claim(claim: ClaimInput):
    art = STATE["artifact"]
    raw_df = pd.DataFrame([claim.model_dump()])
    base_df = build_base_features(raw_df, art["partners"], art["products"])
    feat_df = merge_partner_stats(base_df, art["partner_stats"], art["partner_defaults"])

    X = feat_df[NUM_COLS + CAT_COLS]
    score = float(_predict_ensemble(art["lr_pipe"], art["lgb_pipe"], X)[0])
    threshold = float(art["recommended_threshold"])

    if score >= threshold:
        risk_band = "HIGH"
        action = "HOLD_FOR_REVIEW"
    elif score >= 0.30:
        risk_band = "MEDIUM"
        action = "MONITOR_OR_SPOT_CHECK"
    else:
        risk_band = "LOW"
        action = "AUTO_APPROVE"

    row = feat_df.iloc[0]
    reasons = generate_reasons(row, score)
    amt = float(claim.claim_amount_inr)
    expected_net = round(score * amt - (1.0 - score) * 380.0, 2)

    return {
        "claim_id": claim.claim_id,
        "fraud_score": round(score, 4),
        "risk_band": risk_band,
        "recommended_action": action,
        "threshold_used": threshold,
        "expected_net_value_inr": expected_net,
        "reasons": reasons,
        "partner_context": {
            "partner_id": claim.partner_id,
            "city": str(row.get("city", "unknown")),
            "partner_type": str(row.get("partner_type", "unknown")),
            "historical_claims": int(row.get("p_hist_n", 0)),
            "historical_frauds": int(row.get("p_hist_fraud", 0)),
            "recent_60d_frauds": int(row.get("p_rec_fraud", 0)),
            "under_2k_share": round(float(row.get("p_under2k_share", 0.0)), 3),
        },
    }

