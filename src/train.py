"""Train the Kestrel warranty fraud model, run temporal validation, and generate predictions.csv."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CAT_COLS,
    NUM_COLS,
    ROOT_DIR,
    SAMPLE_SUB_FILE,
    build_base_features,
    compute_partner_stats_table,
    deduplicate_train,
    load_raw_data,
    merge_partner_stats,
)

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "validation_metrics.json"
PREDICTIONS_PATH = ROOT_DIR / "predictions.csv"


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )


def _fit_ensemble(
    X: pd.DataFrame, y: pd.Series, weights: np.ndarray
) -> tuple[Pipeline, Pipeline]:
    lr_pipe = Pipeline([
        ("pre", _make_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.3, random_state=42)),
    ])
    lgb_pipe = Pipeline([
        ("pre", _make_preprocessor()),
        (
            "clf",
            lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=15,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            ),
        ),
    ])
    lr_pipe.fit(X, y, clf__sample_weight=weights)
    lgb_pipe.fit(X, y, clf__sample_weight=weights)
    return lr_pipe, lgb_pipe


def _predict_ensemble(lr_pipe: Pipeline, lgb_pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return 0.65 * lr_pipe.predict_proba(X)[:, 1] + 0.35 * lgb_pipe.predict_proba(X)[:, 1]


def evaluate_slice(
    y_true: np.ndarray, amt: np.ndarray, probs: np.ndarray, n_months: int, cap_per_month: int = 40
) -> dict:
    k = min(len(y_true), n_months * cap_per_month)
    df_e = pd.DataFrame({"y": y_true, "amt": amt, "p": probs})
    top = df_e.sort_values("p", ascending=False).head(k)
    tp = int((top["y"] == 1).sum())
    fp = int((top["y"] == 0).sum())
    total_fraud = int((df_e["y"] == 1).sum())
    fraud_stopped = float(top.loc[top["y"] == 1, "amt"].sum())
    total_fraud_amt = float(df_e.loc[df_e["y"] == 1, "amt"].sum())
    goodwill_cost = fp * 380.0
    net_saved = fraud_stopped - goodwill_cost
    return {
        "n_claims": int(len(y_true)),
        "n_fraud": total_fraud,
        "dummy_zero_accuracy": float((y_true == 0).mean()),
        "roc_auc": float(roc_auc_score(y_true, probs)) if total_fraud > 0 else None,
        "pr_auc": float(average_precision_score(y_true, probs)) if total_fraud > 0 else None,
        "accuracy_at_0_5": float(accuracy_score(y_true, (probs >= 0.5).astype(int))),
        "capacity_k": int(k),
        "tp_at_k": tp,
        "fp_at_k": fp,
        "precision_at_k": float(tp / k) if k else 0.0,
        "recall_at_k": float(tp / total_fraud) if total_fraud else 0.0,
        "fraud_stopped_inr": round(fraud_stopped, 2),
        "total_fraud_inr": round(total_fraud_amt, 2),
        "goodwill_cost_inr": round(goodwill_cost, 2),
        "net_saved_inr": round(net_saved, 2),
        "net_per_review_inr": round(net_saved / k, 2) if k else 0.0,
    }


def _oof_partner_features(
    known_df: pd.DataFrame, behavior_df: pd.DataFrame
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    parts = []
    for fit_idx, oof_idx in skf.split(known_df, known_df["is_fraud"]):
        stats, defaults = compute_partner_stats_table(known_df.iloc[fit_idx], behavior_df)
        parts.append(merge_partner_stats(known_df.iloc[oof_idx], stats, defaults))
    return pd.concat(parts, axis=0).sort_index()


def train_and_evaluate() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    train_raw, test_raw, partners, products = load_raw_data()
    train_dedup = deduplicate_train(train_raw)
    train_feat = build_base_features(train_dedup, partners, products)
    test_feat = build_base_features(test_raw, partners, products)
    known = train_feat[train_feat["is_fraud"].notna()].copy()

    # 1. Temporal Validation:
    # (a) Pre-May stable regime: Train <= 2026-02-28, Val = Mar-Apr 2026 (2 months)
    # (b) Post-May policy shift (after seeing May 2026): Train <= 2026-05-31, Val = Jun 2026 (1 month)
    val_results = {}
    for label, cutoff_str, n_months, boost in [
        ("pre_may_mar_apr_2026", "2026-03-01", 2, 1.0),
        ("post_may_jun_2026", "2026-06-01", 1, 8.0),
    ]:
        cutoff = pd.Timestamp(cutoff_str)
        if label == "pre_may_mar_apr_2026":
            tr_k = known[known["dt"] < cutoff].copy()
            va_k = known[(known["dt"] >= cutoff) & (known["dt"] < pd.Timestamp("2026-05-01"))].copy()
            beh_tr = train_feat[train_feat["dt"] < cutoff]
            beh_va = train_feat[train_feat["dt"] < pd.Timestamp("2026-05-01")]
        else:
            tr_k = known[known["dt"] < cutoff].copy()
            va_k = known[known["dt"] >= cutoff].copy()
            beh_tr = train_feat[train_feat["dt"] < cutoff]
            beh_va = train_feat

        tr_full = _oof_partner_features(tr_k, beh_tr)
        stats_va, def_va = compute_partner_stats_table(tr_k, beh_va)
        va_full = merge_partner_stats(va_k, stats_va, def_va)

        weights = np.where(tr_full["dt"] >= pd.Timestamp("2026-05-01"), boost, 1.0)
        lr_p, lgb_p = _fit_ensemble(tr_full[NUM_COLS + CAT_COLS], tr_full["is_fraud"].astype(int), weights)
        probs = _predict_ensemble(lr_p, lgb_p, va_full[NUM_COLS + CAT_COLS])
        val_results[label] = evaluate_slice(
            va_full["is_fraud"].astype(int).values,
            va_full["claim_amount_inr"].values,
            probs,
            n_months=n_months,
            cap_per_month=40,
        )

    # 2. Final Model on ALL known labels (with 8x recency weight on post-May 2026 regime)
    all_beh = pd.concat([train_feat, test_feat], ignore_index=True)
    full_tr = _oof_partner_features(known, train_feat)
    full_weights = np.where(full_tr["dt"] >= pd.Timestamp("2026-05-01"), 8.0, 1.0)
    lr_final, lgb_final = _fit_ensemble(
        full_tr[NUM_COLS + CAT_COLS], full_tr["is_fraud"].astype(int), full_weights
    )

    partner_stats, partner_defaults = compute_partner_stats_table(known, all_beh)
    te_full = merge_partner_stats(test_feat, partner_stats, partner_defaults)
    test_scores = _predict_ensemble(lr_final, lgb_final, te_full[NUM_COLS + CAT_COLS])

    # Save predictions.csv matching sample_submission.csv order
    sub = pd.read_csv(SAMPLE_SUB_FILE)
    pred_df = pd.DataFrame({"claim_id": te_full["claim_id"], "score": np.round(test_scores, 6)})
    pred_out = sub[["claim_id"]].merge(pred_df, on="claim_id", how="left")
    assert len(pred_out) == len(sub), "Prediction row count mismatch"
    assert pred_out["score"].isna().sum() == 0, "Missing prediction scores"
    pred_out.to_csv(PREDICTIONS_PATH, index=False)

    # Recommended operational threshold: top 40/month (~top 120 of 2252 = 5.3% -> ~0.44 score)
    top120_threshold = float(np.sort(test_scores)[-120])

    artifact = {
        "lr_pipe": lr_final,
        "lgb_pipe": lgb_final,
        "partner_stats": partner_stats,
        "partner_defaults": partner_defaults,
        "partners": partners,
        "products": products,
        "recommended_threshold": round(top120_threshold, 4),
    }
    joblib.dump(artifact, MODEL_PATH)

    summary = {
        "train_dedup_known_rows": int(len(known)),
        "train_fraud_count": int(known["is_fraud"].sum()),
        "test_rows": int(len(pred_out)),
        "recommended_threshold": round(top120_threshold, 4),
        "test_score_p95": round(float(np.percentile(test_scores, 95)), 4),
        "test_score_max": round(float(np.max(test_scores)), 4),
        "validation": val_results,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_and_evaluate()

