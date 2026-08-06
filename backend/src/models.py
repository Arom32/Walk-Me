from sqlalchemy import Column, Integer, String
from src.database import Base
import uuid
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True) 
    email = Column(String(255), unique=True, nullable=False, index=True)
    nickname = Column(String(50), nullable=False)  #사용자 이름
    password_hash = Column(String, nullable=False)  #해쉬된 비밀번호
    is_active = Column(Boolean, default=True, nullable=False) #활동, 비활동
    created_at = Column(DateTime(timezone=True), server_default=func.now()) #가입일
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now()) #수정일