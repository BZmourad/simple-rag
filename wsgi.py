"""Gunicorn entry point."""

from src.api.server import create_app
from src.config import load_config

cfg = load_config()
app = create_app(cfg)