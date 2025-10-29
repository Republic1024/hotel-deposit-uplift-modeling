# Hotel Deposit Uplift Modeling

**Causal Inference · Customer Behavior Analytics · Observational A/B**

This project uses **uplift modeling (heterogeneous treatment effect / ITE)** to estimate how **deposit policy** (`deposit_type`) affects **booking cancellations** (`is_canceled`) on a per-user basis. Rather than predicting “who cancels,” we quantify **the causal *difference***:
[
\text{uplift}(x) ;=; P(\text{cancel}\mid \text{deposit},x);-;P(\text{cancel}\mid \text{no-deposit},x)
]

## ✨ Key Findings

* **Counter-intuitive pattern.** Users predicted to be **most sensitive to deposits** (high uplift) actually show **lower baseline cancel probability** (≈ **0.13**) than low-sensitivity users (≈ **0.51**).
  → **Interpretation:** deposit works more like a **selection filter** than a deterrent.
* **Policy:**

  * **Require deposits** from **low-sensitivity / high-risk** users.
  * **Waive deposits** for **high-sensitivity / low-risk** users and **suppress post-booking offers** to avoid regret/overthinking-induced churn.

## 📦 Dataset

* **File:** `./archive_8/hotel_bookings.csv`
* **Size:** 119,390 bookings, 32 columns
* **Notable fields:** `deposit_type`, `is_canceled`, `lead_time`, `customer_type`, `market_segment`, `previous_cancellations`, `booking_changes`, `total_of_special_requests`, `days_in_waiting_list`, etc.

> **Note:** The dataset is excluded via `.gitignore` (`archive_8/`, `archive_8.zip`). Place the CSV at the path above before running.

## 🧪 Methodology

* **Design:** Observational causal study with **Propensity Scores** and **Inverse Probability Weighting (IPW)** to reduce selection bias.
* **Learner:** **T-Learner** with **Gradient Boosting**:

  * Train **f₁(x)** on treated (deposit=1) with IPW.
  * Train **f₀(x)** on controls (deposit=0) with IPW.
  * Compute **uplift = f₁(x) − f₀(x)** for each user.
* **Features (examples):** `lead_time`, `market_segment`, `previous_cancellations`, `booking_changes`, `total_of_special_requests`, `days_in_waiting_list`, one-hot hotel/segment dummies.

**Why uplift (vs. plain classification)?**
It answers *“what changes if we add/remove the deposit?”* instead of *“who cancels?”*, enabling **policy targeting** and **ROI-aware interventions**.

## 📓 Notebook Outline

* **Cell 1 — Data loading & project scaffold**: shape, columns, sanity checks.
* **Cell 2 — Naïve A/B**: raw difference in cancel rates.
* **Cell 3 — Feature prep**: one-hot, targets, treatment.
* **Cell 3.5 — Propensity score**: logistic regression (P(T=1\mid X)).
* **Cell 3.6 — IPW**: stabilized weights for treated/controls.
* **Cell 4 — Weighted T-Learner**: train **f₁**, **f₀** with sample weights.
* **Cell 4.1 — Fit sanity**: group-specific train accuracy (focus remains on effect estimation).
* **Cell 5 — Per-user uplift**: compute and describe distribution; deciles.
* **Cell 6 — Decile analysis**: cancellation vs. uplift deciles; baseline risk check.


## 📈 Sanity Checks (illustrative)

* **Group fits (train)**: f₁ (treated) ≈ **0.998**, f₀ (control) ≈ **0.783**.
  *Note:* these are **not** the objective; the goal is **credible effect estimation**, not max classification accuracy.

## 🧠 Business Playbook

* **Low-sensitivity users** (high baseline risk): require **deposit / non-refundable** or use **deposit-as-credit** mechanisms to lock commitment.
* **High-sensitivity users** (low baseline risk): **waive deposit** and **avoid post-booking promotions** to prevent cognitive dissonance and cancellations.

## 🛡️ Caveats

* Observational data → relies on **unconfoundedness** given features; hidden confounders may remain.
* Always A/B validate policy before full rollout (e.g., staggered or geo experiments).

## 📜 License

MIT (code). Dataset license follows its original source.

---

**Contact:** [GitHub @republic1024](https://github.com/Republic1024) · For academic/industry collaboration on decision intelligence & causal uplift modeling.
