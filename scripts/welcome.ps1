function Write-Section {
    param([string]$Title)
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Write-Host "Welcome to TXCAT" -ForegroundColor Cyan
Write-Host "On-prem, explainable transaction categorisation for bank/card/UPI narratives." -ForegroundColor Gray
Write-Host "Repo: $root" -ForegroundColor DarkGray

Write-Section "Key files"
Write-Host " - README.md                 : Overview, setup, commands" -ForegroundColor Gray
Write-Host " - config/taxonomy.yaml      : Financial label schema" -ForegroundColor Gray
Write-Host " - src/model_runtime.py      : Loads models/txcat_fastcpu_v7.joblib; predict_batch()" -ForegroundColor Gray
Write-Host " - src/app_api.py            : FastAPI app; /predict/text, /jobs/csv, serves frontend" -ForegroundColor Gray
Write-Host " - src/tasks.py              : Celery task for batch CSV scoring" -ForegroundColor Gray
Write-Host " - frontend/index.html/js/css: Web console UI" -ForegroundColor Gray
Write-Host " - reports/metrics.json      : Latest metrics (acc 0.966, macro-F1 0.969, thr ~4.9k tx/s, ~0.204 ms/tx)" -ForegroundColor Gray
Write-Host " - models/txcat_fastcpu_v7.joblib: Trained model artifact (keep in place)" -ForegroundColor Gray
Write-Host " - data/train.csv, data/test.csv: Synthetic datasets (ignored uploads/results dirs)" -ForegroundColor Gray

Write-Section "Setup (PowerShell)"
Write-Host " python -m venv .venv" -ForegroundColor Yellow
Write-Host " .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Yellow
Write-Host " pip install fastapi uvicorn[standard] celery redis numpy pandas scikit-learn scikit-learn-intelex joblib tqdm python-multipart" -ForegroundColor Yellow

Write-Section "Run API + web console"
Write-Host " uvicorn src.app_api:app --host 0.0.0.0 --port 8000 --reload" -ForegroundColor Yellow
Write-Host " open http://localhost:8000/app" -ForegroundColor Yellow

Write-Section "Start batch worker"
Write-Host " docker run -d --name redis -p 6379:6379 redis:7   # first time" -ForegroundColor Yellow
Write-Host " docker start redis                                # subsequent runs" -ForegroundColor Yellow
Write-Host " celery -A src.celery_app:celery_app worker -l info" -ForegroundColor Yellow

Write-Section "CLI predictions"
Write-Host " python -m src.run_predictions                      # interactive REPL" -ForegroundColor Yellow
Write-Host " python -m src.run_predictions -t \"AMEX SWIGGY JAIPUR REF CARD614755\"" -ForegroundColor Yellow
Write-Host " python -m src.run_predictions --csv data/test.csv --column narrative --json" -ForegroundColor Yellow

Write-Section "Data & training"
Write-Host " python -m src.synthetic_data --train-size 100000 --test-size 100000 --noise-ratio 0.18" -ForegroundColor Yellow
Write-Host " python -m src.train_eval_v7_faststack --coarse-trainer sgd --n-jobs -1 --bucket-workers -1 --predict-workers -1 --threshold-workers -1 --chunk-size 25000 --disable-review --save-model" -ForegroundColor Yellow

Write-Host "`nHappy Learning!" -ForegroundColor Green
