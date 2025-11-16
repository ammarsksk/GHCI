from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from celery import states

from .celery_app import celery_app
from .model_runtime import DEFAULT_LOW_CONF, predict_batch


DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@celery_app.task(bind=True, name="txcat.score_csv")
def score_csv(
    self,
    input_path: str,
    output_path: str,
    text_column: str,
    low_conf: float = DEFAULT_LOW_CONF,
    chunk_size: int = 5000,
) -> Dict[str, Any]:
    """
    Score a CSV file in chunks and write a new CSV with prediction columns appended.

    The task periodically updates its state with progress information and a short
    rolling log of recent events.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {in_path}")

    # Estimate total rows (minus header)
    with in_path.open("r", encoding="utf-8") as fh:
        total_rows = sum(1 for _ in fh) - 1
    if total_rows < 0:
        total_rows = 0

    processed = 0
    logs = []

    try:
        first_chunk = True
        for chunk_idx, df_chunk in enumerate(
            pd.read_csv(in_path, chunksize=chunk_size, encoding="utf-8")
        ):
            if text_column not in df_chunk.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")

            texts = df_chunk[text_column].astype(str).fillna("").tolist()
            preds = predict_batch(texts, low_conf=low_conf)

            df_preds = pd.DataFrame(preds)[
                [
                    "label",
                    "confidence",
                    "coarse_label",
                    "coarse_confidence",
                    "needs_review",
                ]
            ]
            out_chunk = pd.concat(
                [df_chunk.reset_index(drop=True), df_preds], axis=1
            )

            mode = "w" if first_chunk else "a"
            header = first_chunk
            out_chunk.to_csv(
                out_path, index=False, mode=mode, header=header, encoding="utf-8"
            )
            first_chunk = False

            processed += len(df_chunk)
            logs.append(
                f"Processed chunk {chunk_idx + 1}, rows={len(df_chunk)}, total_processed={processed}"
            )
            progress = int(100 * processed / max(total_rows, 1)) if total_rows else 100

            self.update_state(
                state=states.STARTED,
                meta={
                    "current": processed,
                    "total": total_rows,
                    "progress": progress,
                    "stage": "scoring",
                    "logs": logs[-20:],
                    "output_path": str(out_path),
                },
            )
    except Exception as exc:  # pragma: no cover - defensive
        # Surface a helpful message in the logs, but let Celery handle
        # the failure bookkeeping so the backend metadata stays valid.
        logs.append(f"ERROR: {exc}")
        self.update_state(
            state=states.STARTED,
            meta={
                "current": processed,
                "total": total_rows,
                "progress": int(
                    100 * processed / max(total_rows, 1)
                )
                if total_rows
                else 0,
                "stage": "failure",
                "logs": logs[-20:],
                "output_path": str(out_path),
            },
        )
        raise

    # Final state
    final = {
        "current": processed,
        "total": total_rows,
        "progress": 100,
        "stage": "completed",
        "logs": logs,
        "output_path": str(out_path),
    }
    return final
