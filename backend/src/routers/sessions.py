import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src import models, schemas
from src.database import get_db

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/", response_model=schemas.SessionResponse, status_code=201)
def create_session(
    session_data: schemas.SessionCreate,
    db: Session = Depends(get_db),
):
    user = db.query(models.User).filter(
        models.User.id == session_data.user_id
    ).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="사용자를 찾을 수 없습니다.",
        )

    new_session = models.ChatSession(
        user_id=session_data.user_id,
        dialect_zone=session_data.dialect_zone,
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return new_session


@router.get("/user/{user_id}", response_model=list[schemas.SessionResponse])
def get_user_sessions(
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    user = db.query(models.User).filter(
        models.User.id == user_id
    ).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="사용자를 찾을 수 없습니다.",
        )

    return (
        db.query(models.ChatSession)
        .filter(models.ChatSession.user_id == user_id)
        .order_by(models.ChatSession.created_at.desc())
        .all()
    )


@router.get("/{session_id}", response_model=schemas.SessionResponse)
def get_session(
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

    return session


@router.patch("/{session_id}/end", response_model=schemas.SessionResponse)
def end_session(
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

    session.is_active = False
    session.ended_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(session)

    return session