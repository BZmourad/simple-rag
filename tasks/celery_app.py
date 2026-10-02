"""Celery application for async ingestion."""

from __future__ import annotations

import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery(
    "simple_rag",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "3600")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "3300")),
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)