# Submission Form Notes — Kestrel Home (Variant C)

## 1. Expected Score of `predictions.csv` & Why
- **Expected Performance on `test_unlabelled.csv` (Jul–Sep 2026, 2,252 claims):**
  - **ROC-AUC:** `0.82 – 0.88`
  - **PR-AUC:** `0.35 – 0.45` (11×–15× lift over the ~3% post-May base rate)
  - **Top-40/month (`top 120` over 3 months):** Expected **35%–45% precision** (~42–54 frauds caught), **55%–65% recall**, stopping **~₹60,000–₹80,000** in fraud vs **~₹25,000–₹30,000** in goodwill (`₹380 × FP`), yielding **~₹30,000–₹50,000 net savings** (**₹250–₹420 net/review**).
- **Why:**
  - The post-May 1 policy (`< ₹2,000` auto-approved without inspection) remains active in Jul–Sep 2026 (`77%` of test claims are `< ₹2,000`, `~21%` inspected).
  - The 5 primary post-May repeat-offender partners (`SP3160`, `SP3232`, `SP3318`, `SP3129`, `SP3319`) continue submitting high volumes of `< ₹2,000` uninspected claims in the test window (`63` claims total, `52` under ₹2,000).
  - Our June 2026 out-of-time validation (`ROC-AUC = 0.8667`, `PR-AUC = 0.3872`, `14/22` frauds caught in top 40) directly mirrors this Jul–Sep 2026 regime.
  - **Caveat:** `81` test claims (`3.6%`) come from `14` partners unseen in `train.csv`; if a brand-new fraud ring emerges among those 14, recall will dip slightly until their first investigation outcomes are labelled.

## 2. Decisions Made on Ambiguities
1. **KPI Choice:** Rejected the 97% accuracy target (a dummy all-zero model gets 96.9%–98.8% accuracy while stopping ₹0) and optimized for **ROC-AUC / PR-AUC + Net Rupees Saved at 40 reviews/month**.
2. **Duplicate `claim_id`s:** Deduplicated `train.csv` (681 resubmitted pairs) keeping earliest submission so partner/claim counts are not double-counted.
3. **Blank `is_fraud` in CRM:** Excluded the 215 open/undecided CRM rows from training labels instead of treating them as `0`.
4. **Recommended Threshold:** Set `recommended_threshold = 0.5794` so the flagged volume matches the investigation desk's **40 claims/month** capacity (~5.3% of claims).

## 3. AI Tools Used, Cost, and What Was Discarded
- **Tools Used:** Antigravity coding assistant (Gemini) for EDA scripting, pipeline structuring, FastAPI/HTML scaffolding, and documentation.
- **Cost:** \$0 in external/paid APIs inside the product (the deployed service runs 100% locally with `scikit-learn` + `lightgbm` and requires no paid API keys).
- **What Was Discarded:**
  - **Pre-May unweighted model:** Discarded after temporal validation showed `ROC-AUC = 0.09–0.42` on May–Jun 2026 due to the May 1 policy regime shift.
  - **Expected-Value (`p × amount`) ranking on uncalibrated probabilities:** Discarded because `class_weight="balanced"` inflated probabilities on high-ticket ₹20k+ legitimate claims, causing false positives that wiped out net savings.
  - **Complex NLP/LLM scoring on `claim_description`:** Discarded because fault descriptions are short, standardized templates (with 4 noisy boilerplate/injected strings in train) that add latency and API-key fragility with zero lift over structured + partner features.

