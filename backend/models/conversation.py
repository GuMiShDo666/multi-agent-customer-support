"""对话、消息及Agent执行元数据模型。"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .ticket import Base


class ConversationDB(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=True)
    customer_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship(
        "MessageDB", back_populates="conversation", cascade="all, delete-orphan"
    )


class MessageDB(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # "user" / "assistant"
    content = Column(Text)
    agent_name = Column(String, nullable=True)
    # 用于追踪路由结果和回复质量
    intent = Column(String, nullable=True)       # 意图分类结果
    sentiment = Column(String, nullable=True)    # 情感分析结果
    qa_score = Column(Float, nullable=True)      # QA质检得分
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("ConversationDB", back_populates="messages")


class Message(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    conversation_id: Optional[int] = None
    role: str
    content: str
    agent_name: Optional[str] = None
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    qa_score: Optional[float] = None
    created_at: Optional[datetime] = None

class Conversation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    ticket_id: Optional[int] = None
    customer_id: str
    messages: List[Message] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class MessageCreate(BaseModel):
    conversation_id: int
    role: str
    content: str
    agent_name: Optional[str] = None
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    qa_score: Optional[float] = None
