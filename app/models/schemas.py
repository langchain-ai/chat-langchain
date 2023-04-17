"""Schemas for the chat app."""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, validator
from sqlmodel import Field, SQLModel


class ChatLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    time_stamp: datetime = Field(default_factory=datetime.now())
    question: str
    answer: str
    chat_trace: str


class DataSource(BaseModel):
    page_content: str
    meta_data: Dict


class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: str
    message: str
    type: str
    sources: Optional[List[DataSource]]

    @validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["bot", "you"]:
            raise ValueError("sender must be bot or you")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in ["start", "stream", "end", "error", "info"]:
            raise ValueError("type must be start, stream or end")
        return v
