import enum
import uuid

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True) 
    email = Column(String(255), unique=True, nullable=False, index=True)
    nickname = Column(String(50), nullable=False)  #사용자 이름
    password_hash = Column(String, nullable=False)  #해쉬된 비밀번호
    is_active = Column(Boolean, default=True, nullable=False) #활동, 비활동
    created_at = Column(DateTime(timezone=True), server_default=func.now()) #가입일
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now()) #수정일
    sessions = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )
class DialectType(str, enum.Enum):
    GANGWON = "gangwon"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class InputType(str, enum.Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dialect_zone = Column(
        Enum(DialectType, name="dialect_type"),
        nullable=False,
        default=DialectType.GANGWON,
    )
    is_active = Column(Boolean, default=True, nullable=False)
    context_summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(
        Enum(MessageRole, name="message_role"),
        nullable=False,
    )
    input_kind = Column(
        Enum(InputType, name="input_type"),
        nullable=False,
        default=InputType.TEXT,
    )
    processed_text = Column(Text, nullable=True)
    raw_media_url = Column(Text, nullable=True)
    tts_audio_url = Column(Text, nullable=True)
    used_chunks = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")