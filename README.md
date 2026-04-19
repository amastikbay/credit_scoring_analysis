# Classical Credit Scoring — BankCred LLC

**Portfolio role:** Classical credit scoring project (logistic regression, WoE/IV, scorecard design).  
Interpretability-focused counterpart to the `Risk-assessment` ML project.

---

## Business Context

**Client:** BankCred LLC (fictional), Moscow  
**Product:** Unsecured consumer loans  
**Default definition:** Any missed scheduled payment within 24 months of disbursement (`debt = 1`)  
**Dataset period:** 2021–2022 | 21,454 borrowers after cleaning  
**Base default rate:** ~8.1%

**Goal:** Build a classical credit scorecard to rank-order borrowers by default probability, enabling threshold-based accept/decline decisions and risk-based pricing.

---

## What This Project Covers

| Section | Content |
|---|---|
| Data Understanding | EDA, missing value analysis, anomaly detection |
| Data Preprocessing | Imputation, deduplication, lemmatization, categorization |
| EDA | Target distribution, univariate plots, categorical default rates, correlation heatmap |
| Feature Engineering | WoE/IV analysis, encoding, WoE transformation |
| Modeling | Logistic regression with L2 regularization, stratified CV |
| Evaluation | ROC/AUC, Gini, KS statistic, PR curve, score decile table, confusion matrices |
| Scorecard Design | Siddiqi points-based scorecard, score bands, `score_borrower()` function |
| Business Implications | Accept rate analysis, cost asymmetry, portfolio monitoring |

---

## Data Dictionary

| Feature | Type | Description | Notes |
|---|---|---|---|
| `children` | int | Number of children | -1 corrected to 0 (47 rows); values >10 treated as data entry errors but kept as "large family" |
| `days_employed` | int | Work experience in days | Sign error: non-retirees have negative values (corrected via abs); pensioner values ~365k days (artifact); capped at 10,950 for modeling |
| `dob_years` | int | Client age in years | 101 rows with age=0 replaced with group mean by income type |
| `education` | str | Education level (Russian) | Mixed case normalized to lowercase |
| `education_id` | int | Ordinal education level | 0=higher, 1=secondary, 2=incomplete higher, 3=primary, 4=doctorate |
| `family_status` | str | Marital status (Russian) | 5 categories |
| `family_status_id` | int | Ordinal marital status | 0=married, 1=civil union, 2=widowed, 3=divorced, 4=single |
| `gender` | str | Gender | M / F / XNA (1 row, treated as F) |
| `income_type` | str | Employment type (Russian) | 8 categories; 4 rare ones (n<5) merged into "other" for modeling |
| `debt` | int | **Target variable** | 1 = defaulted, 0 = repaid on time |
| `total_income` | int | Monthly income (RUB) | 2,174 NaN filled with group mean by income type |
| `purpose` | str | Loan purpose (Russian free text) | 38 unique phrases → 4 categories via pymystem3 lemmatization |

---

## Key Findings

| Feature | Default Rate (Best) | Default Rate (Worst) |
|---|---|---|
| Age | 5.5% (>65 yrs) | 11.0% (<30 yrs) |
| Marital status | 6.6% (widowed) | 9.8% (single) |
| Loan purpose | 7.2% (real estate) | 9.4% (car) |
| Children | 7.5% (no children) | 9.3% (1–2 children) |

**Model performance (expected):** AUC ~0.60–0.65, Gini ~0.20–0.30, KS ~0.18–0.22.  
All IVs < 0.10 — typical for demographic-only, thin-file datasets without bureau behavioral data.

---

## How to Run

### Option A — conda (recommended)

```bash
git clone <repo-url>
cd credit_scoring_analysis
conda env create -f environment.yml
conda activate credit_scoring
jupyter notebook scoring.ipynb
```

### Option B — pip

```bash
pip install -r requirements.txt
jupyter notebook scoring.ipynb
```

Once Jupyter opens, run **Kernel → Restart & Run All**.

**Expected runtime:** 3–5 minutes (pymystem3 downloads mystem binary on first run).  
**Expected AUC:** 0.58–0.65 (if ≥ 0.70, check for data leakage).

---

## Tools

**Language:** Python 3.12  
**Libraries:** pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, nltk, pymystem3, snowballstemmer, jupyter
