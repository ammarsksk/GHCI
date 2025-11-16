from __future__ import annotations

import os

from celery import Celery


CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)


celery_app = Celery(
    "txcat_jobs",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["src.tasks"],
)

# Windows has limited support for the prefork pool used by default in Celery.
# Force the "solo" pool to avoid semaphore-related PermissionError issues.
# For true multi-core execution, prefer running this stack under Linux/WSL/Docker.
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600 * 4,
    worker_pool="solo",
)
