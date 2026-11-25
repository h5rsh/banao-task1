# Model Evidence, Validation & Failure Analysis

## 1. Data Integrity & Cleaning Decisions
- **Resubmitted Claims (`claim_id` duplicates):** `train.csv` has 12,029 rows but 11,348 unique `claim_id`s (681 claims were resubmitted once with a later `submitted_at` and identical `is_fraud` label). We deduplicated `train.csv` by `claim_id` (keeping earliest submission) to prevent duplicate-row leakage. `test_unlabelled.csv` has 2,252 unique `claim_id`s.
- **Open Investigations (`is_fraud` blanks):** 215 CRM rows have blank `is_fraud` (still under investigation at export). We excluded them from supervised training labels rather than imputing `0`. Legacy Zoho rows (`source == legacy_zoho`, before 1 Oct 2025) had no blanks.
- **Serial Messiness:** ~30% of `product_serial` values deviate from `KH#########` (spaces, lowercase, length). We engineered format flags rather than dropping rows.

## 2. Why "97% Accuracy" Is Misleading
In the deduplicated labelled dataset (`11,146` rows), only `141` (`1.27%`) are fraudulent (`3.09%` in May–Jun 2026 for `< ₹2,000`). A trivial all-zero classifier scores **98.76% accuracy** in Mar–Apr 2026 and **96.91% accuracy** in Jun 2026 while catching **₹0** of fraud. Therefore, we evaluate on **ranking quality (ROC-AUC, PR-AUC)** and **net rupees saved under Farhan's 40-reviews/month cap** (`Fraud ₹ Stopped − ₹380 × False Positives`).

## 3. Temporal Validation Results (Strict Out-of-Time Splits)
We deliberately avoided random K-fold splits, which leak future partner fraud rates and ignore the 1 May 2026 policy shift. Partner historical fraud rates were computed out-of-fold inside training and strictly from pre-cutoff data for validation.

| Validation Window | Capacity | Fraud Cases | Dummy All-0 Acc | ROC-AUC | PR-AUC | TP@K (Recall) | Fraud Stopped | Goodwill Cost (₹380×FP) | **Net ₹ Saved** | **Net ₹ / Review** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Mar–Apr 2026** (Pre-May regime, 2 mo) | 80 | 18 | 98.76% | **0.7408** | **0.4362** | 10 (55.6%) | ₹63,824 | ₹26,600 | **₹37,224** | **₹465** |
| **Jun 2026** (Post-May regime, 1 mo) | 40 | 22 | 96.91% | **0.8667** | **0.3872** | 14 (63.6%) | ₹20,263 | ₹9,880 | **₹10,383** | **₹260** |

## 4. Model Architecture
- **Ensemble:** `0.65 × LogisticRegression (L2, C=0.3, balanced)` + `0.35 × LightGBM (15 leaves, balanced)` with **8× recency sample weight** on post-May 2026 claims so the model reflects the active `< ₹2,000` uninspected fraud regime.
- **Top Signals:** Recent 60-day & all-time smoothed partner fraud rate, partner's `< ₹2,000` & uninspected claim share, `claim_amount_inr` & proximity to ₹2,000 (`1800–1999`), `customer_prior_claims`, `partner_inspected`, `photo_attached`, `days_since_purchase`, and `amt_to_list_ratio`.

## 5. How Often It Does Not Work (Failure Modes & Limits)
1. **Unseen Policy Shifts (Zero-Shot Regime Change):** When we trained *only* on pre-May 2026 data and tested on May–Jun 2026 without any post-May examples, ROC-AUC collapsed (`< 0.45`) because fraud flipped from high-value old partners (`>= ₹2,000`) to low-value uninspected claims (`< ₹2,000`). **Mitigation:** Recency weighting + retraining monthly as new investigation outcomes close.
2. **False Positives at 40/Month:** In June 2026, `26` of the `40` flagged claims were genuine (`65%` false-alarm rate within the review queue), costing ₹9,880 in customer goodwill—still net-positive because fraud claims average ₹1,447+ vs ₹380 goodwill.
3. **Cold-Start Partners:** `14` partners (`81` rows) in `test_unlabelled.csv` have no prior training history. For them, the model falls back on global priors plus real-time behavioral signals (`customer_prior_claims`, `< ₹2,000` share, missing photo/inspection), which has lower precision than established repeat-offender partners.
4. **Undetected Novel Fraudsters (8/22 missed in Jun 2026):** ~36% of June frauds came from partners with 1–2 isolated claims and clean documentation; catching them requires random spot-checks rather than model ranking alone.

