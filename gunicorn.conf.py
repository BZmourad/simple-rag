"""Gunicorn configuration."""

import multiprocessing
import os
import pathlib

bind = "unix:/run/simple-rag/gunicorn.sock"
workers = int(os.getenv("WORKERS", multiprocessing.cpu_count()))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 300
graceful_timeout = 30
keepalive = 5
max_requests = 2000
max_requests_jitter = 200
preload_app = True
accesslog = "-"
errorlog = "-"
loglevel = "info"


def on_starting(server):
    sock = pathlib.Path("/run/simple-rag/gunicorn.sock")
    if sock.exists():
        sock.unlink()