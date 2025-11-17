from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery.result import AsyncResult
from celery import states
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .celery_app import celery_app
from .model_runtime import DEFAULT_LOW_CONF, predict_batch
from .tasks import RESULT_DIR, UPLOAD_DIR, CANCEL_DIR, score_csv


DATA_DIR = Path("data")


class TextPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="Raw transaction narratives.")
    low_conf: float = Field(
        DEFAULT_LOW_CONF,
        ge=0.0,
        le=1.0,
        description="Threshold below which predictions are flagged for review.",
    )


class TextPrediction(BaseModel):
    text: str
    label: str
    confidence: float
    coarse_label: str
    coarse_confidence: float
    needs_review: bool
    explanation: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    state: str
    progress: int
    current: int
    total: int
    stage: str
    logs: List[str] = []


app = FastAPI(title="txcat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.post("/predict/text", response_model=List[TextPrediction])
def predict_texts(payload: TextPredictRequest) -> List[TextPrediction]:
    """
    Synchronous prediction for small batches of transaction text.
    """
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    preds = predict_batch(payload.texts, low_conf=payload.low_conf)
    out: List[TextPrediction] = []
    for res in preds:
        out.append(
            TextPrediction(
                text=res["text"],
                label=res["label"],
                confidence=float(res["confidence"]),
                coarse_label=res["coarse_label"],
                coarse_confidence=float(res["coarse_confidence"]),
                needs_review=bool(res["needs_review"]),
                explanation=res.get("explanation") or None,
            )
        )
    return out


@app.post("/jobs/csv", response_model=JobCreateResponse)
async def create_csv_job(
    file: UploadFile = File(...),
    text_column: str = Form(..., description="Name of the text column in the CSV."),
    low_conf: float = Form(
        DEFAULT_LOW_CONF,
        ge=0.0,
        le=1.0,
        description="Threshold below which predictions are flagged for review.",
    ),
) -> JobCreateResponse:
    """
    Create a background job to score a CSV file.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}.csv"
    output_path = RESULT_DIR / f"{job_id}_scored.csv"

    # Save uploaded file to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with input_path.open("wb") as fh:
        content = await file.read()
        fh.write(content)

    # Enqueue Celery task
    task = score_csv.apply_async(
        args=[str(input_path), str(output_path), text_column, float(low_conf)]
    )
    return JobCreateResponse(job_id=task.id)


def _build_status(job_id: str, res: AsyncResult) -> JobStatusResponse:
    meta: Dict[str, Any] = {}
    logs: List[str] = []

    if isinstance(res.info, dict):
        meta = res.info
        raw_logs = meta.get("logs", [])
        if isinstance(raw_logs, list):
            logs = [str(x) for x in raw_logs]
    elif res.info:
        # Failure or other non-dict info; surface it as a single log line.
        logs = [str(res.info)]

    progress = int(meta.get("progress", 0))
    current = int(meta.get("current", 0))
    total = int(meta.get("total", 0))
    stage = str(meta.get("stage", res.state))
    return JobStatusResponse(
        job_id=job_id,
        state=res.state,
        progress=progress,
        current=current,
        total=total,
        stage=stage,
        logs=logs,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get progress and status information for a CSV scoring job.
    """
    res = AsyncResult(job_id, app=celery_app)
    status = _build_status(job_id, res)

    # If a cancel marker is present and the task has not completed
    # successfully, surface the job as cancelled regardless of the
    # underlying Celery state so the UI does not appear to "resume".
    cancel_marker = CANCEL_DIR / f"{job_id}.cancel"
    if cancel_marker.exists() and status.state not in (states.SUCCESS, "SUCCESS"):
        status.state = "CANCELLED"
        status.stage = "cancelled"
    return status


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, str]:
    """
    Request cancellation of a running job.
    """
    # Mark this job as cancelled so the worker can cooperatively stop
    # after the current chunk, and also send a best-effort revoke signal.
    CANCEL_DIR.mkdir(parents=True, exist_ok=True)
    cancel_marker = CANCEL_DIR / f"{job_id}.cancel"
    cancel_marker.touch(exist_ok=True)
    celery_app.control.revoke(job_id, terminate=False)
    return {"job_id": job_id, "status": "cancel_requested"}


@app.get("/jobs/{job_id}/result")
def download_result(job_id: str):
    """
    Download the scored CSV for a completed job.
    """
    res = AsyncResult(job_id, app=celery_app)
    if res.state not in (states.SUCCESS, "SUCCESS"):
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current state: {res.state}",
        )

    # Reconstruct output path from task result or naming convention
    meta: Dict[str, Any] = {}
    if isinstance(res.info, dict):
        meta = res.info
    output_path_str = meta.get("output_path")
    if output_path_str:
        out_path = Path(output_path_str)
    else:
        out_path = RESULT_DIR / f"{job_id}_scored.csv"

    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found.")

    return FileResponse(
        path=str(out_path),
        filename=out_path.name,
        media_type="text/csv",
    )


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "txcat API is running. Open /app for the web UI if configured."}
