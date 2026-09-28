"""MRCD review + playground application (API, worker, static web client)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return Path(os.environ.get("MRCD_DATA", ROOT / "data" / "app"))


def open_store():
    from app.store import Store

    data_dir().mkdir(parents=True, exist_ok=True)
    return Store(os.environ.get("DATABASE_URL", f"sqlite:///{(data_dir() / 'app.db').as_posix()}"))


def media_dir(recording: dict) -> Path:
    return data_dir() / "media" / recording["data"]["sha256"]
