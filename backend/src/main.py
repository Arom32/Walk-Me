"""Walk-Me FastAPI — RAG + 사투리 + (선택) TTS.

실행:
  cd backend
  set PYTHONPATH=%CD%;%CD%\\..\\ai
  uvicorn src.main:app --host 0.0.0.0 --port 8000

또는 repo 루트에서:
  uvicorn backend.src.main:app --app-dir . --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
AI_DIR = REPO_ROOT / "ai"
for p in (str(BACKEND_DIR), str(AI_DIR), str(AI_DIR / "tts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import settings  # noqa: E402
#추가----------------
from src.database import Base, engine  # noqa: E402
from src import models  # noqa: E402 
from src.routers import messages, sessions, users  # noqa: E402



app = FastAPI(title="Walk-Me", version="0.2.0")
Base.metadata.create_all(bind=engine)
app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(messages.router)

#-------------------------



class GuideRequest(BaseModel):
    question: str = Field(..., examples=["속초 가볼 만한 곳"])
    region: Optional[str] = None
    k: int = 5
    places_only: bool = True
    with_llm: bool = True
    no_lora: bool = False
    tts: bool = False


@app.get("/")
def read_root():
    return {"message": "Walk-Me 서버 정상", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "database_host": settings.DB_HOST,
        "database_name": settings.DB_NAME,
        "lora_path": str(settings.lora_path()),
        "chroma_dir": str(settings.chroma_dir()),
    }


@app.post("/guide")
def guide(req: GuideRequest):
    """관광 RAG → 사투리 텍스트 → (선택) CosyVoice wav."""
    try:
        from rag.ask import answer_question
    except ImportError as e:
        raise HTTPException(500, f"rag 모듈 로드 실패: {e}") from e

    result = answer_question(
        req.question,
        k=req.k,
        region=req.region,
        places_only=req.places_only,
        with_llm=req.with_llm,
        no_lora=req.no_lora,
    )

    payload = {
        "question": result["question"],
        "region": result["region"],
        "places": [
            {"name": p.get("name"), "address": p.get("address"), "visit_type": p.get("visit_type")}
            for p in result["places"]
        ],
        "standard": result["standard"],
        "answer": result["answer"],
        "dialect_ok": result["dialect_ok"],
        "audio_url": None,
    }

    if req.tts:
        try:
            # VRAM: LLM은 지우지 않고 CPU로만 내려서 TTS에 GPU를 내준다
            # (지우면 다음 요청 때 디스크에서 다시 로딩해야 함 — release_llm_gpu_memory 참고)
            from rag.ask import release_llm_gpu_memory

            release_llm_gpu_memory()

            from tts.subprocess_client import synthesize_via_cosyvoice_env

            speak = result["answer"]
            wav_path = synthesize_via_cosyvoice_env(speak)
            payload["audio_url"] = f"/guide/audio/{wav_path.name}"
            payload["audio_path"] = str(wav_path)
        except Exception as e:
            payload["tts_error"] = str(e)

    return payload


@app.get("/guide/audio/{filename}")
def guide_audio(filename: str):
    out_dir = AI_DIR / "tts" / "outputs"
    path = (out_dir / filename).resolve()
    if not str(path).startswith(str(out_dir.resolve())):
        raise HTTPException(400, "invalid path")
    if not path.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
