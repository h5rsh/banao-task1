# Memo: Warranty Claim Fraud Triage — Decision & Next Week's Plan

**To:** Ritu Deshpande, Head of D2C Operations  
**Cc:** Farhan Sheikh (Finance), Meenal Joshi (Service Desk), Tanmay Kulkarni (Data & IT)  
**Date:** 26 November 2026  

---

### 1. The Decision
Deploy a **ranked risk-triage model** that scores every incoming claim and routes the **top 40 highest-risk claims per month** to the investigation desk before payout. Do **not** use a 97% accuracy target or block all new partners.

### 2. Why "97% Accuracy" Fails Finance — and the Real Numbers
Only **1.3% to 3.1%** of Kestrel's claims are fraudulent. A useless rule that approves *every* claim achieves **98.8% accuracy** (Mar–Apr) and **96.9% accuracy** (Jun) while losing **100% of fraud rupees**.

Instead, measured against Farhan's **40-claim/month review cap** and **₹380 goodwill cost** per genuine claim held:
- **Post-May 1 Regime (June 2026 test, 40 reviews):** Catches **14 of 22 frauds (64% recall)**, stops **₹20,263** in fraudulent payouts, incurs **₹9,880** in goodwill on 26 false positives, and delivers **₹10,383 net savings (₹260 net saved per claim reviewed)**.
- **Pre-May 1 Regime (Mar–Apr 2026 test, 80 reviews):** Catches **10 of 18 frauds (56% recall)**, stops **₹63,824**, incurs **₹26,600** in goodwill, and delivers **₹37,224 net savings (₹465 net saved per claim reviewed)**.
- **Projected Q3 (Jul–Sep 2026, 120 reviews):** Expected **₹30,000–₹50,000 net savings** after goodwill costs.

### 3. What the Data Actually Shows (New Partners & the May 1 Loophole)
1. **Do not penalise all new partners.** While newer partners (onboarded since Jul 2025) show a higher *average* fraud rate (7.0% vs 1.0%), **fraud is concentrated in ~10 specific outlets out of 380**—both old franchises (`SP3207`, `SP3376`, `SP3095`) and 5 newer outlets (`SP3160`, `SP3232`, `SP3318`, `SP3129`, `SP3319`). Most new tier-2/tier-3 partners have **0% fraud**.
2. **The 1 May 2026 auto-approval rule shifted fraud under ₹2,000.** Before May 1, 69% of fraud was ≥ ₹2,000. After May 1, **35 of 36 frauds were uninspected claims under ₹2,000** (fraud rate jumped 5.7× from 0.54% to 3.09%), with repeat offenders submitting claims just below the cutoff (e.g., ₹1,995) and customers with 2+ prior claims showing 5–50% fraud rates.

### 4. What You Should Do Next Week
1. **Monday — Put the triage queue live:** Route the top 40 scored claims/month (score ≥ `0.58`) to the investigation desk with the model's plain-English reason codes.
2. **Tuesday — Audit the 5 active under-₹2,000 repeat outlets:** Have Meenal's team require photo + inspection sign-off for **all** claims (even under ₹2,000) from `SP3160`, `SP3232`, `SP3318`, `SP3129`, and `SP3319` for 60 days—without touching clean new partners.
3. **Wednesday — Close the ₹1,800–₹1,999 structuring gap:** Add random 5% spot-checks on uninspected ₹1,800–₹1,999 claims and any customer with `customer_prior_claims >= 2`.
4. **Friday — Track Finance's metric weekly:** Review **Net ₹ Saved per Claim Reviewed** (target ≥ ₹250/review) rather than overall accuracy.

