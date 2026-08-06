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
        
#추가