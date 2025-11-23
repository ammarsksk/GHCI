# TXCAT – Smart transaction categorisation console

On‑prem, explainable transaction categorisation for bank, card, and UPI statements – no third‑party APIs, everything runs locally on your machine.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-5.5+-67b045?logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7.x-DC382D?logo=redis&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e?logo=scikitlearn&logoColor=white)
![Vanilla JS](https://img.shields.io/badge/Frontend-Vanilla%20JS%20%2B%20HTML%2FCSS-646cff?logo=javascript&logoColor=white)

## Overview

TXCAT is an end‑to‑end stack for automatically classifying raw transaction narratives (e.g. `AMEX SWIGGY JAIPUR REF CARD614755`) into rich financial categories such as `DINING`, `GROCERIES`, `UTILITIES_TELECOM`, `DEBT_LOAN_HOME`, `INVEST_MF_SIP`, `PETS`, and more.

The core goals:

- **On‑prem & privacy‑preserving** – all data and models run locally; nothing leaves your laptop or cluster.
- **Explainable** – the model exposes confidences, routing decisions, and feature‑based “top signals”.
- **Human‑in‑the‑loop** – low‑confidence predictions are flagged for review, and the UI lets humans give feedback.
- **Fast** – an optimised “faststack” of word+char TF‑IDF + scikit‑learn models, with vectorised threshold search and batched inference.

High‑level pipeline:

- Normalise and featurise text with a shared **word (1–2gram) + char (3–5gram)** TF‑IDF featuriser.
- Run a **coarse router** (e.g. `FOOD`, `SHOPPING`, `TRAVEL`, `UTILITIES`, `HEALTH`, `OTHER_MISC`) with brand lexicon boosts.
- For each coarse bucket, run a **fine‑level ensemble** (LogisticRegression + ComplementNB).
- Apply **per‑class F1‑optimised thresholds** to decide final labels and “needs review”.
- Expose everything via a **FastAPI** backend, **Celery** batch worker, and a **vanilla JS web console**.

---

## Features

- **On‑prem inference** – no external ML APIs; models are local `.joblib` artifacts.
- **Rich taxonomy** – detailed, India‑centric financial categories with metadata (`config/taxonomy.yaml`).
- **Hierarchical model**
  - Coarse router over a small set of semantic buckets.
  - Fine ensembles per bucket (VotingClassifier: LR + ComplementNB).
  - Per‑bucket thresholds calibrated for F1.
- **Lexicon & overrides**
  - Brand lexicon boosts coarse probabilities (e.g. `ZOMATO` → `FOOD`, `JIO` → `UTILITIES_TELECOM`).
  - Fine‑label hard overrides to fix worst confusions (e.g. fuel vs groceries).
- **Fast evaluation**
  - Batched vectorisation & prediction.
  - Multi‑core parallelism for bucket training/prediction.
  - Example v7 metrics on 100k test rows:
    - Accuracy: **0.966**
    - Macro‑F1: **0.969**
    - Throughput (hierarchy only): ~**4.9k tx/s**
    - Latency: ~**0.20 ms/tx**
- **Web console**
  - Single‑transaction playground with confidence ring and explanation snippets.
  - CSV batch scoring with progress, metrics, execution log, and download.
  - Session prediction log with feedback (“Correct” / “Incorrect: label X”).
  - Dark, dashboard‑style UI with a separate evaluation report page.

---

## Architecture

### Components

- **Taxonomy & synthetic data**
  - `config/taxonomy.yaml` – rich label schema for Indian retail banking.
  - `src/synthetic_data.py` – generates realistic synthetic `train.csv` and `test.csv` using the taxonomy.

- **Featuriser**
  - `src/featurizer_patch.py`:
    - `normalize_text` removes banking boilerplate, long numbers, cities, etc.
    - `build_text_union()` (mirrored in v7 as `build_text_union_fast`) returns a `FeatureUnion` of:
      - Word TF‑IDF (1–2 grams).
      - Char‑WB TF‑IDF (3–5 grams).
    - Tuned for short financial strings.

- **Lexicon & overrides**
  - `src/lexicon_patch.py`:
    - `COARSE_KEYS`: `['FOOD','HEALTH','SHOPPING','TRAVEL','UTILITIES','OTHER_MISC']`.
    - `LEX`: regex patterns for brands and hints (`zomato`, `jio`, `amazon`, `hpcl`, etc.).
    - `coarse_boost(proba, text)`: bumps router probabilities for matching buckets.
    - `fine_hard_override(label, text)`: fixes obvious mis‑routes (e.g. `JIO` → `UTILITIES_TELECOM`).

- **Training (v7 faststack)**
  - `src/train_eval_v7_faststack.py`:
    - Step `[1/6]` – Fit featuriser, vectorise train/test.
    - Step `[2/6]` – Fit coarse router (LogReg or SGD).
    - Step `[3/6]` – Calibrate coarse thresholds via vectorised F1 search.
    - Step `[4/6]` – Train fine bucket models (VotingClassifier: LR + ComplementNB) and per‑class thresholds.
    - Step `[5/6]` – Predict on test set (coarse routing + fine per bucket, optional explanations).
    - Step `[6/6]` – Report metrics, confusion matrix, and (optionally) save model to `models/txcat_fastcpu_v7.joblib`.

- **Runtime**
  - `src/model_runtime.py`:
    - Loads `models/txcat_fastcpu_v7.joblib` once per process.
    - Applies lexicon boosts and hard overrides at inference.
    - Reconstructs `fine_solo` buckets (single‑label buckets with no trained classifier).
    - Exposes `predict_batch(texts: List[str], low_conf: float)` returning:

      ```python
      {
        "text": "...",
        "label": "DINING",
        "confidence": 0.987,
        "coarse_label": "FOOD",
        "coarse_confidence": 0.995,
        "needs_review": False,
        "explanation": "zomato:1.23; swiggy:0.98; ..."
      }
      ```

- **API & worker**
  - `src/app_api.py`:
    - `POST /predict/text` – synchronous scoring for small batches of narratives.
    - `POST /jobs/csv` – create a Celery job for CSV scoring.
    - `GET /jobs/{job_id}` – job status + logs.
    - `POST /jobs/{job_id}/cancel` – cooperative cancellation via marker file.
    - `GET /jobs/{job_id}/result` – download scored CSV.
    - Serves static frontend at `/app` from `frontend/`.
  - `src/celery_app.py`:
    - Celery app bound to `redis://localhost:6379/0` (by default).
    - Uses `worker_pool="solo"` for Windows compatibility; full multicore best on Linux/WSL/Docker.
  - `src/tasks.py`:
    - `txcat.score_csv` Celery task:
      - Streams input CSV from `data/uploads/` in chunks.
      - Calls `predict_batch` on the configured text column.
      - Writes `*_scored.csv` to `data/results/` with new columns:
        - `label`, `confidence`, `coarse_label`, `coarse_confidence`, `needs_review`.
      - Regularly updates Celery task state with progress & logs.
      - Honours cancellation markers from `/jobs/{job_id}/cancel`.

- **Frontend**
  - `frontend/index.html`, `frontend/styles.css`, `frontend/main.js`:
    - Hero card with current v7 metrics.
    - **Single transaction card**:
      - Input textbox, “Score transaction” button.
      - Shows label pill, coarse routing line, status pill (“No review needed” / “Needs review”), top‑signal explanations, and a confidence ring.
      - Feedback UI: “Correct” / “Incorrect” + dropdown for corrected label; updates session log.
    - **Batch scoring card**:
      - Upload CSV, choose text column, set review threshold via input + slider.
      - “Start batch job” / “Abort job”.
      - Job status pill (Idle / Running / Completed / Cancelled / Failed), progress bar, metric grid (rows processed, duration, throughput, flagged %), execution log panel, and download link.
    - **Prediction log**:
      - Table of all single‑transaction predictions this session, with feedback status.

  - `frontend/metrics.html`:
    - Summary card with accuracy/macro‑F1/throughput/latency.
    - Full v7 classification report in a table.
    - Button to go back to the main playground.

---

## Taxonomy & labels

TXCAT’s taxonomy is defined in `config/taxonomy.yaml` and is designed for Indian retail banking feeds.

Each label includes fields like:

- `id` – stable machine label (e.g. `DINING`, `GROCERIES`, `INCOME_REFUNDS`).
- `display_name` – human‑facing label.
- `description` – short explanation of the category.
- `group` – `EXPENSE | INCOME | TRANSFER | INTERNAL`.
- `primary` – `true` for top‑level categories.
- `parent_id` – parent label (for hierarchical grouping).
- `budget_bucket` – `ESSENTIAL | DISCRETIONARY | DEBT | INVESTMENT | INTERNAL`.
- `default_direction` – `DEBIT | CREDIT`.
- `default_recurring` – whether transactions are typically recurring.
- `channel_tags` / `instrument_tags` – typical channels/instruments (UPI, CARD, NETBANKING, WALLET, CASH, etc.).
- `mcc_hints`, `upi_merchant_bucket`, `risk_tags`, `tax_relevant`.
- `keywords`, `example_merchants`, `example_descriptions`.

Examples (non‑exhaustive):

- **Everyday spend**
  - `DINING`, `GROCERIES`, `FUEL`, `MOBILITY`, `ENTERTAINMENT`, `SHOPPING_ECOM`, `SHOPPING_ELECTRONICS`.
- **Utilities**
  - `UTILITIES_POWER`, `UTILITIES_TELECOM`, `UTILITIES_WATER_GAS`.
- **Housing / services**
  - `HOUSING_RENT`, `HOUSING_MAINTENANCE`, `HOME_SERVICES`.
- **Debt / loans**
  - `DEBT_CREDIT_CARD_BILL`, `DEBT_LOAN_HOME`, `DEBT_LOAN_PERSONAL`, `DEBT_LOAN_VEHICLE`, `DEBT_LOAN_BNPL`, `DEBT_COLLECTION_AGENCY`.
- **Investments & insurance**
  - `INVEST_MF_SIP`, `INVEST_STOCK_BROKERAGE`, `INVEST_GOLD_SILVER`, `INVEST_RETIREMENT`,
  - `INSURANCE_PREMIUM_LIFE`, `INSURANCE_PREMIUM_HEALTH`, `INSURANCE_PREMIUM_MOTOR`.
- **Income**
  - `INCOME_SALARY`, `INCOME_BUSINESS_SELF_EMPLOYED`, `INCOME_INVESTMENT`, `INCOME_REFUNDS`, `INCOME_GOVT_BENEFIT`.
- **Transfers**
  - `TRANSFER_P2P_FAMILY_FRIENDS`, `TRANSFER_SELF_INTERNAL`, `TRANSFER_CASH_WITHDRAWAL`, `TRANSFER_FOREX`.

To extend the taxonomy, edit `config/taxonomy.yaml`, regenerate synthetic data, and retrain the model.

---

## Data format

### Synthetic training/test data

Generated by `src/synthetic_data.py` into `data/train.csv` and `data/test.csv`:

- Columns:
  - `transaction_id` – unique identifier.
  - `value_date`, `posted_at` – dates.
  - `amount` – numeric, as string.
  - `currency` – e.g. `INR`.
  - `dr_cr` – `DR` or `CR`.
  - `merchant_name` – merchant or counterparty.
  - `reference` – reference IDs (UPI/NEFT/IMPS/etc.).
  - `narrative` – raw statement line used as text input.
  - `channel` – `UPI`, `CARD`, `NETBANKING`, `NEFT`, `IMPS`, `EMI`, `SUBSCRIPTION`, etc.
  - `account_type` – `SAVINGS`, `CURRENT`, `CREDIT_CARD`, etc.
  - `city` – location string.
  - `keyword_hit` – primary keyword used for generation.
  - `category_id` – ground‑truth label ID.
  - `category_display_name` – human label.

`train_eval_v7_faststack.py` automatically detects which columns to use:

- Text: prefers `raw_text` if present; otherwise `narrative`.
- Label: prefers `label`; otherwise `category_id` or `category_display_name`.

### Review & reports

During training:

- `reports/metrics.json` – summary metrics (accuracy, macro‑F1, throughput, latency).
- `reports/confusion_matrix.csv` – full confusion matrix per label.
- `reports/preds_test.csv` – test rows with predictions & explanation strings.
- `data/review.csv` – (optional) subset of low‑confidence rows for human review.

During batch scoring:

- Input CSVs are stored temporarily under `data/uploads/` (ignored by git).
- Scored outputs go to `data/results/{job_id}_scored.csv` (ignored by git).

---

## Model training & evaluation

You **don’t have to retrain** to use TXCAT if `models/txcat_fastcpu_v7.joblib` is present (ideally tracked via Git LFS). Training is reproducible if you want to tweak the taxonomy or data.

### 1. Create virtualenv & install dependencies

From the repo root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install fastapi uvicorn[standard] celery redis \
            numpy pandas scikit-learn scikit-learn-intelex \
            joblib tqdm python-multipart
```

> `scikit-learn-intelex` is optional, but the scripts are ready to use it for faster CPU performance.

### 2. Generate synthetic data (optional)

If you want fresh data based on the current taxonomy:

```bash
python -m src.synthetic_data \
  --train-size 100000 \
  --test-size 100000 \
  --noise-ratio 0.18
```

This writes `data/train.csv` and `data/test.csv`.

### 3. Train v7 faststack model

Example high‑parallelism run:

```bash
python -m src.train_eval_v7_faststack \
  --coarse-trainer sgd \
  --n-jobs -1 \
  --bucket-workers -1 \
  --predict-workers -1 \
  --threshold-workers -1 \
  --chunk-size 25000 \
  --disable-review \
  --save-model
```

What this does:

- Fits the featuriser, coarse router, and fine bucket models.
- Calibrates thresholds for both coarse and fine layers.
- Evaluates on the hold‑out test set.
- Writes:
  - `reports/metrics.json`
  - `reports/confusion_matrix.csv`
  - `reports/preds_test.csv`
- Saves the model artifact to:
  - `models/txcat_fastcpu_v7.joblib`

Example metrics (from `reports/metrics.json`):

```json
{
  "macro_f1": 0.969,
  "accuracy": 0.966,
  "n_train": 100000,
  "n_test": 100000,
  "throughput_tx_per_s": 4913.09,
  "latency_ms_per_tx": 0.2035
}
```

---

## Running the API & web console

TXCAT uses Redis + Celery for batch jobs, and FastAPI + Uvicorn for the API/UI.

### 1. Start Redis (Docker)

First time:

```bash
docker run -d --name redis -p 6379:6379 redis:7
```

Subsequent runs:

```bash
docker start redis
```

### 2. Start the FastAPI app

From the repo root, in a venv:

```bash
uvicorn src.app_api:app --host 127.0.0.1 --port 8000 --reload
```

- Root health check: `http://localhost:8000/`
- Web console: `http://localhost:8000/app`

### 3. Start the Celery worker

In another terminal (same venv):

```bash
celery -A src.celery_app:celery_app worker -l info
```

On Windows the worker uses `worker_pool="solo"` to avoid semaphore issues. For true multicore Celery execution, consider running under WSL or Docker/Linux.

### 4. Use the web console

Open `http://localhost:8000/app`:

- **Hero card**
  - Shows current v7 metrics (accuracy, macro‑F1, throughput, latency).
  - Button to open the detailed evaluation page (`metrics.html`).

- **Single transaction playground**
  - Paste a narrative like `AMEX SWIGGY JAIPUR REF CARD614755`.
  - Click **“Score transaction”**.
  - See:
    - Fine label pill (`MOBILITY`, `DINING`, etc.).
    - Coarse routing line (`Coarse: TRAVEL | 98.7% routed`).
    - Status pill: `No review needed` or `Needs review`.
    - Top signals explanation (when available).
    - Confidence ring (percentage).
  - Give feedback:
    - “Correct” (logs as `Accepted`).
    - “Incorrect” + choose correct label (logs as `Corrected: LABEL`).
  - All events go into the **Prediction log** table.

- **Batch scoring (CSV)**
  - Upload a `.csv` file.
  - Choose the text column (e.g. `narrative`).
  - Set review threshold (e.g. `0.60`) via textbox or slider.
  - Click **“Start batch job”**:
    - Status pill changes to `Running`.
    - Progress bar advances as chunks are processed.
    - Metrics update live: rows processed, duration, throughput, flagged %.
    - Execution log shows per‑chunk messages.
  - Click **“Abort job”** to cancel:
    - Uses marker file in `data/cancelled/` + Celery revoke.
    - UI status pill switches to `Cancelled`.
  - On completion, the UI exposes a **download link** to the scored CSV.

- **Prediction log**
  - Shows a session‑local audit log of single predictions:
    - Source (`single`), text, label, conf, coarse & conf, review flag, feedback.

---

## Using the model programmatically

### Direct Python usage

```python
from src.model_runtime import predict_batch, DEFAULT_LOW_CONF

texts = [
    "UPI/UPI54321987 UBERINDIA@icici DELHI",
    "VISA HPCL FUEL PUMP KOLKATA REF CARD965225",
]

results = predict_batch(texts, low_conf=DEFAULT_LOW_CONF)
for r in results:
    print(r["text"], "→", r["label"], f"({r['confidence']:.3f})")
```

### REST API

#### Single prediction

```bash
curl -X POST "http://localhost:8000/predict/text" \
  -H "Content-Type: application/json" \
  -d '{
        "texts": ["AMEX SWIGGY JAIPUR REF CARD614755"],
        "low_conf": 0.6
      }'
```

Response (shape):

```json
[
  {
    "text": "AMEX SWIGGY JAIPUR REF CARD614755",
    "label": "DINING",
    "confidence": 0.987,
    "coarse_label": "FOOD",
    "coarse_confidence": 0.995,
    "needs_review": false,
    "explanation": "swiggy:1.23; jaipur:0.87; ..."
  }
]
```

#### Batch CSV job

Create a job:

```bash
curl -X POST "http://localhost:8000/jobs/csv" \
  -F "file=@data/train.csv" \
  -F "text_column=narrative" \
  -F "low_conf=0.6"
```

You’ll receive:

```json
{ "job_id": "10295f82-0b4e-4a5b-b572-91e74ce93bf6" }
```

Poll status:

```bash
curl "http://localhost:8000/jobs/10295f82-0b4e-4a5b-b572-91e74ce93bf6"
```

Download result when `state` is `SUCCESS`:

```bash
curl -o scored.csv "http://localhost:8000/jobs/10295f82-0b4e-4a5b-b572-91e74ce93bf6/result"
```

Cancel a job:

```bash
curl -X POST "http://localhost:8000/jobs/10295f82-0b4e-4a5b-b572-91e74ce93bf6/cancel"
```

---

## Project layout

```text
txcat/
├─ config/
│  └─ taxonomy.yaml         # Rich label taxonomy (v3)
├─ data/
│  ├─ train.csv             # Synthetic training data (generated)
│  ├─ test.csv              # Synthetic test data (generated)
│  ├─ uploads/              # CSV uploads for batch jobs (ignored by git)
│  ├─ results/              # Scored CSVs from batch jobs (ignored by git)
│  └─ cancelled/            # Cancel markers for jobs (ignored by git)
├─ frontend/
│  ├─ index.html            # Main console UI
│  ├─ metrics.html          # Evaluation report page
│  ├─ styles.css            # Dark dashboard styling
│  ├─ main.js               # Vanilla JS behaviour
│  └─ favicon.png           # Tab icon
├─ models/
│  └─ txcat_fastcpu_v7.joblib  # Trained model artifact (track via Git LFS)
├─ reports/
│  ├─ metrics.json          # Training metrics
│  ├─ confusion_matrix.csv  # Full confusion matrix
│  └─ preds_test.csv        # Test predictions + explanations
├─ src/
│  ├─ app_api.py            # FastAPI app & endpoints
│  ├─ celery_app.py         # Celery config (Redis, solo pool)
│  ├─ tasks.py              # Celery task: txcat.score_csv
│  ├─ model_runtime.py      # v7 runtime: predict_batch()
│  ├─ lexicon_patch.py      # Brand lexicon & overrides
│  ├─ featurizer_patch.py   # Text normalisation + featuriser
│  ├─ synthetic_data.py     # Taxonomy-driven synthetic data generator
│  ├─ train_eval_v7_faststack.py  # Main trainer/evaluator
│  └─ train_eval_v6_faststack.py  # Legacy v6 trainer (kept for reference)
├─ .gitignore
└─ featurizer_patch.py      # Top-level shim for older pickles
```

---

## Extending & customization

TXCAT is designed to be hackable:

- **Add or refine categories**
  - Edit `config/taxonomy.yaml` to add new `labels` or adjust metadata.
  - Regenerate synthetic data (`src/synthetic_data.py`) and retrain v7.

- **Improve brand handling**
  - Add regex patterns in `src/lexicon_patch.py`:
    - `LEX` for new brands or merchants.
    - Adjust `BUCKET_TO_COARSE` if you add new coarse buckets.
  - Extend `fine_hard_override` with targeted rules.

- **Tune thresholds and routing**
  - Adjust `COARSE_MIN` and `FINE_MIN` in `train_eval_v7_faststack.py`.
  - Use CLI flags to change:
    - `--coarse-trainer` (`lr` vs `sgd`).
    - `--coarse-max-iter`, `--coarse-epochs`, `--coarse-batch`, `--coarse-cal-size`.
    - `--fine-cal-frac`, `--low-conf`.

- **Performance tuning**
  - Use `--n-jobs -1`, `--bucket-workers -1`, `--predict-workers -1`, `--threshold-workers -1` to fully utilise CPU.
  - Adjust `--chunk-size` if you hit memory limits.

- **Explainability**
  - Turn explanations on/off with `--enable-explanations` when training v7.
  - Runtime uses LR feature weights for low‑confidence explanations only.

---

## Development & contributions

This is a Python + FastAPI + Celery + vanilla JS project.

Suggested workflow:

- Use a virtualenv and pin library versions once you’re happy (`requirements.txt`).
- Keep code style roughly in line with what’s already in `src/`:
  - Small, focused modules.
  - Prefer explicit, clear names over abbreviations.
- When adding new functionality:
  - Update this README and, if relevant, the web console (`frontend/`).
  - Regenerate metrics & include a brief note in `reports/` if you change the model.

Simple manual checks:

- `python -m src.synthetic_data --train-size 10000 --test-size 5000` to verify data generation.
- `python -m src.train_eval_v7_faststack --coarse-trainer sgd --n-jobs -1 --bucket-workers -1 --predict-workers -1 --threshold-workers -1 --chunk-size 25000 --disable-review --save-model` to verify training.
- `uvicorn src.app_api:app --reload` + `celery -A src.celery_app:celery_app worker -l info` to run the full stack.
---
