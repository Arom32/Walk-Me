import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    nickname: str
    password: str


class UserUpdate(BaseModel):
    nickname: str | None = None


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    nickname: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

from src.models import DialectType, InputType, MessageRole


class SessionCreate(BaseModel):
    user_id: uuid.UUID
    dialect_zone: DialectType = DialectType.GANGWON


class SessionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    dialect_zone: DialectType
    is_active: bool
    context_summary: str | None
    created_at: datetime
    ended_at: datetime | None

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    role: MessageRole
    input_kind: InputType = InputType.TEXT
    processed_text: str | None = None
    raw_media_url: str | None = None
    tts_audio_url: str | None = None
    used_chunks: list[dict] | None = None


class MessageResponse(BaseModel):
    id: int
    session_id: uuid.UUID
    role: MessageRole
    input_kind: InputType
    processed_text: str | None
    raw_media_url: str | None
    tts_audio_url: str | None
    used_chunks: list[dict] | None
    created_at: datetime

    class Config:
        from_attributes = True