import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src import models, schemas
from src.database import get_db

router = APIRouter(prefix="/sessions", tags=["messages"])


@router.post(
    "/{session_id}/messages",
    response_model=schemas.MessageResponse,
    status_code=201,
)
def create_message(
    session_id: uuid.UUID,
    message_data: schemas.MessageCreate,
    db: Session = Depends(get_db),
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=404,
            detail="대화 세션을 찾을 수 없습니다.",
        )

    if not session.is_active:
        raise HTTPException(
            status_code=400,
            detail="종료된 대화 세션에는 메시지를 추가할 수 없습니다.",
        )

    new_message = models.ChatMessage(
        session_id=session_id,
        role=message_data.role,
        input_kind=message_data.input_kind,
        processed_text=message_data.processed_text,
        raw_media_url=message_data.raw_media_url,
        tts_audio_url=message_data.tts_audio_url,
        used_chunks=message_data.used_chunks,
    )

    db.add(new_message)
    db.commit()
    db.refresh(new_message)

    return new_message


@router.get(
    "/{session_id}/messages",
    response_model=list[schemas.MessageResponse],
)
def get_messages(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=404,
            detail="대화 세션을 찾을 수 없습니다.",
        )

    return (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.created_at.asc())
        .all()
    )