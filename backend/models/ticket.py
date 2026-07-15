"""工单状态、优先级和持久化模型。"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketDB(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    subject = Column(String)
    description = Column(Text)
    status = Column(SQLEnum(TicketStatus), default=TicketStatus.OPEN)
    priority = Column(SQLEnum(TicketPriority), default=TicketPriority.MEDIUM)
    assigned_agent = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    resolution = Column(Text, nullable=True)


class Ticket(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    customer_id: str
    subject: str
    description: str
    status: TicketStatus = TicketStatus.OPEN
    priority: TicketPriority = TicketPriority.MEDIUM
    assigned_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None

class TicketCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(min_length=1, max_length=128)
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=10000)
    priority: TicketPriority = TicketPriority.MEDIUM


class TicketUpdate(BaseModel):
    status: Optional[TicketStatus] = None
    priority: Optional[TicketPriority] = None
    assigned_agent: Optional[str] = None
    resolution: Optional[str] = None
