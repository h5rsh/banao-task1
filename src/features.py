"""Feature engineering, partner statistics, and human-readable reason generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT_DIR / "56cf31d5-987f-4cb3-8bd2-79ae2703b0e7-train.csv"
TEST_FILE = ROOT_DIR / "b5fccfca-78de-4ad7-8875-b6b3bae430d0-test_unlabelled.csv"
PARTNERS_FILE = ROOT_DIR / "bf766253-255c-47e9-9497-7da7563e1f41-partners.csv"
PRODUCTS_FILE = ROOT_DIR / "4d412d82-135b-4e15-b332-a79885e6a1db-products.csv"
SAMPLE_SUB_FILE = ROOT_DIR / "4508b45d-31cb-4bbe-8f86-8b3dfd77ddb1-sample_submission.csv"

NUM_COLS = [
    "days_since_purchase",
    "claim_amount_inr",
    "customer_prior_claims",
    "list_price_inr",
    "warranty_months",
    "partner_tenure_days",
    "is_new_partner",
    "is_under_2000",
    "near_2000",
    "amt_to_list_ratio",
    "photo_attached_flag",
    "partner_inspected_flag",
    "no_photo_no_inspect",
    "inspector_note_blank",
    "warranty_pct_used",
    "hour",
    "serial_valid_kh9",
    "serial_has_space",
    "serial_has_lower",
    "serial_len",
    "prior_claims_ge2",
    "p_hist_n",
    "p_hist_fraud",
    "p_hist_fraud_rate_smooth",
    "p_rec_n",
    "p_rec_fraud_rate_smooth",
    "p_total_claims",
    "p_under2k_share",
    "p_uninspected_share",
    "p_avg_prior_claims",
]

CAT_COLS = ["partner_type", "family", "city"]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    partners = pd.read_csv(PARTNERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)
    return train, test, partners, products


def deduplicate_train(train: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate resubmitted claim_ids (keeping earliest submission, flagging resubmission)."""
    resub_ids = set(train[train.duplicated("claim_id")]["claim_id"])
    df = train.sort_values("submitted_at").drop_duplicates("claim_id", keep="first").copy()
    df["was_resubmitted"] = df["claim_id"].isin(resub_ids).astype(int)
    return df


def build_base_features(
    df: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    out = df.copy()
    # Drop pre-existing merge columns if scoring a single dict that lacks them
    out = out.merge(partners, on="partner_id", how="left", suffixes=("", "_p"))
    out = out.merge(products, on="sku", how="left", suffixes=("", "_prod"))

    out["dt"] = pd.to_datetime(out["submitted_at"], errors="coerce").fillna(pd.Timestamp("2026-07-01"))
    out["onboarded_dt"] = pd.to_datetime(out["onboarded_date"], errors="coerce").fillna(pd.Timestamp("2025-01-01"))
    out["partner_tenure_days"] = (out["dt"] - out["onboarded_dt"]).dt.days.clip(lower=0)
    out["is_new_partner"] = (out["onboarded_dt"] >= pd.Timestamp("2025-07-01")).astype(int)

    out["claim_amount_inr"] = pd.to_numeric(out["claim_amount_inr"], errors="coerce").fillna(0.0)
    out["list_price_inr"] = pd.to_numeric(out["list_price_inr"], errors="coerce").fillna(out["claim_amount_inr"].clip(lower=1.0))
    out["warranty_months"] = pd.to_numeric(out["warranty_months"], errors="coerce").fillna(12.0)
    out["days_since_purchase"] = pd.to_numeric(out["days_since_purchase"], errors="coerce").fillna(180.0)
    out["customer_prior_claims"] = pd.to_numeric(out["customer_prior_claims"], errors="coerce").fillna(0.0)

    out["is_under_2000"] = (out["claim_amount_inr"] < 2000).astype(int)
    out["near_2000"] = ((out["claim_amount_inr"] >= 1800) & (out["claim_amount_inr"] < 2000)).astype(int)
    out["amt_to_list_ratio"] = out["claim_amount_inr"] / out["list_price_inr"].replace(0, np.nan)

    out["photo_attached_flag"] = (out["photo_attached"].astype(str).str.upper() == "Y").astype(int)
    out["partner_inspected_flag"] = (out["partner_inspected"].astype(str).str.upper() == "Y").astype(int)
    out["no_photo_no_inspect"] = (
        (out["photo_attached_flag"] == 0) & (out["partner_inspected_flag"] == 0)
    ).astype(int)
    out["inspector_note_blank"] = (
        out["inspector_note"].isna() | (out["inspector_note"].astype(str).str.strip() == "")
    ).astype(int)

    out["warranty_days"] = out["warranty_months"] * 30.5
    out["warranty_pct_used"] = out["days_since_purchase"] / out["warranty_days"].replace(0, np.nan)
    out["hour"] = out["dt"].dt.hour

    s = out["product_serial"].fillna("").astype(str)
    out["serial_valid_kh9"] = s.str.match(r"^KH\d{9}$").astype(int)
    out["serial_has_space"] = s.str.contains(r"\s", regex=True).astype(int)
    out["serial_has_lower"] = s.str.contains(r"[a-z]", regex=True).astype(int)
    out["serial_len"] = s.str.len()

    out["prior_claims_ge2"] = (out["customer_prior_claims"] >= 2).astype(int)
    out["partner_type"] = out["partner_type"].fillna("unknown")
    out["family"] = out["family"].fillna("unknown")
    out["city"] = out["city"].fillna("unknown")
    return out


def compute_partner_stats_table(
    history_df: pd.DataFrame, behavior_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build partner lookup table from labelled history and unlabelled operational behavior."""
    global_rate = float(history_df["is_fraud"].mean()) if len(history_df) else 0.012
    max_dt = history_df["dt"].max()
    rec = history_df[history_df["dt"] >= (max_dt - pd.Timedelta(days=60))]
    rec_rate = float(rec["is_fraud"].mean()) if len(rec) else global_rate

    p_hist = history_df.groupby("partner_id").agg(
        p_hist_n=("is_fraud", "count"),
        p_hist_fraud=("is_fraud", "sum"),
    )
    p_hist["p_hist_fraud_rate_smooth"] = (
        (p_hist["p_hist_fraud"] + 8.0 * global_rate) / (p_hist["p_hist_n"] + 8.0)
    )

    p_rec = rec.groupby("partner_id").agg(
        p_rec_n=("is_fraud", "count"),
        p_rec_fraud=("is_fraud", "sum"),
    )
    p_rec["p_rec_fraud_rate_smooth"] = (
        (p_rec["p_rec_fraud"] + 3.0 * rec_rate) / (p_rec["p_rec_n"] + 3.0)
    )

    p_beh = behavior_df.groupby("partner_id").agg(
        p_total_claims=("claim_id", "count"),
        p_under2k_share=("is_under_2000", "mean"),
        p_uninspected_share=("partner_inspected_flag", lambda s: 1.0 - s.mean()),
        p_avg_prior_claims=("customer_prior_claims", "mean"),
    )

    stats = p_beh.join(p_hist, how="outer").join(p_rec[["p_rec_n", "p_rec_fraud", "p_rec_fraud_rate_smooth"]], how="outer")
    stats = stats.reset_index()
    defaults = {"global_rate": global_rate, "rec_rate": rec_rate}
    return stats, defaults


def merge_partner_stats(
    target_df: pd.DataFrame, stats: pd.DataFrame, defaults: dict[str, float]
) -> pd.DataFrame:
    out = target_df.merge(stats, on="partner_id", how="left")
    out["p_hist_n"] = out["p_hist_n"].fillna(0)
    out["p_hist_fraud"] = out["p_hist_fraud"].fillna(0)
    out["p_hist_fraud_rate_smooth"] = out["p_hist_fraud_rate_smooth"].fillna(defaults["global_rate"])
    out["p_rec_n"] = out["p_rec_n"].fillna(0)
    out["p_rec_fraud"] = out.get("p_rec_fraud", pd.Series(0, index=out.index)).fillna(0)
    out["p_rec_fraud_rate_smooth"] = out["p_rec_fraud_rate_smooth"].fillna(defaults["rec_rate"])
    out["p_total_claims"] = out["p_total_claims"].fillna(1)
    out["p_under2k_share"] = out["p_under2k_share"].fillna(out["is_under_2000"])
    out["p_uninspected_share"] = out["p_uninspected_share"].fillna(1.0 - out["partner_inspected_flag"])
    out["p_avg_prior_claims"] = out["p_avg_prior_claims"].fillna(out["customer_prior_claims"])
    return out


def generate_reasons(row: pd.Series, score: float) -> list[str]:
    """Generate plain-English reasons a Kestrel operations specialist would read."""
    reasons: list[str] = []

    p_rec_fraud = float(row.get("p_rec_fraud", 0) or 0)
    p_rec_n = float(row.get("p_rec_n", 0) or 0)
    p_hist_fraud = float(row.get("p_hist_fraud", 0) or 0)
    p_hist_n = float(row.get("p_hist_n", 0) or 0)

    if p_rec_fraud >= 2 and p_rec_n > 0:
        reasons.append(
            f"Partner {row['partner_id']} ({row.get('city', 'unknown')}) had {int(p_rec_fraud)} confirmed frauds "
            f"in {int(p_rec_n)} recent claims (last 60 days)"
        )
    elif p_hist_fraud >= 2 and p_hist_n > 0:
        rate_pct = 100.0 * p_hist_fraud / p_hist_n
        reasons.append(
            f"Partner {row['partner_id']} has {int(p_hist_fraud)}/{int(p_hist_n)} historical frauds ({rate_pct:.0f}%)"
        )

    amt = float(row.get("claim_amount_inr", 0) or 0)
    if 1800 <= amt < 2000:
        reasons.append(
            f"Claim amount ₹{amt:,.0f} sits just under the ₹2,000 auto-approval threshold"
        )
    elif amt < 2000 and int(row.get("partner_inspected_flag", 1)) == 0:
        reasons.append(
            f"Auto-approved under-₹2,000 claim (₹{amt:,.0f}) submitted without partner inspection"
        )

    if int(row.get("partner_inspected_flag", 1)) == 0 and int(row.get("photo_attached_flag", 1)) == 0:
        reasons.append("No partner inspection sign-off and no photo attached")
    elif int(row.get("photo_attached_flag", 1)) == 0:
        reasons.append("No supporting photo attached to the claim")

    prior = int(row.get("customer_prior_claims", 0) or 0)
    if prior >= 2:
        reasons.append(f"Customer has {prior} prior warranty claims (elevated repeat-claim risk)")
    elif prior == 1:
        reasons.append("Customer has 1 prior warranty claim")

    under2k_share = float(row.get("p_under2k_share", 0) or 0)
    p_total = float(row.get("p_total_claims", 0) or 0)
    if p_total >= 5 and under2k_share >= 0.80:
        reasons.append(
            f"Partner submits {under2k_share * 100:.0f}% of its claims below the ₹2,000 inspection cutoff"
        )

    days = float(row.get("days_since_purchase", 0) or 0)
    if days <= 60:
        reasons.append(f"Early claim filed {int(days)} days after purchase")

    if int(row.get("serial_valid_kh9", 1)) == 0:
        reasons.append(f"Non-standard product serial format ('{row.get('product_serial', '')}')")

    if not reasons:
        if score >= 0.50:
            reasons.append("Elevated combined risk across partner profile, claim amount, and product history")
        else:
            reasons.append("Low risk: partner history, claim amount, and documentation align with normal claims")

    return reasons[:4]

