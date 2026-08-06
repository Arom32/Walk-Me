"""Walk-Me 강원 관광 RAG 모듈 (ai/rag)."""

from __future__ import annotations

import os
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
AI_DIR = RAG_DIR.parent
REPO_ROOT = AI_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"

# 원본 CSV: backend/data (기존 위치 유지)
RAW_DIR = BACKEND_DIR / "data" / "raw"
EXTRACTED_DIR = BACKEND_DIR / "data" / "extracted"

# RAG 산출물: ai/rag/data
PROCESSED_DIR = RAG_DIR / "data" / "processed"
CHROMA_DIR = RAG_DIR / "data" / "chroma"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def resolve_raw_dir() -> Path:
    return _env_path("RAW_DIR") or RAW_DIR


def resolve_extracted_dir() -> Path:
    return _env_path("EXTRACTED_DIR") or EXTRACTED_DIR


def resolve_chroma_dir() -> Path:
    return _env_path("CHROMA_DIR") or CHROMA_DIR


def resolve_processed_dir() -> Path:
    return _env_path("PROCESSED_DIR") or PROCESSED_DIR
