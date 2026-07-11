"""
FastAPI 主应用 — LangGraph 多Agent智能客服系统。

相比原项目的改进：
- 移除了原项目对 FastAPI build_middleware_stack 的 monkey-patch（不再需要）
- 使用 lifespan 替代已弃用的 on_event
- /api/support/message 返回完整的执行轨迹（trace）和元数据，前端可视化Agent协作过程
"""

import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from ..database.database import get_db, init_db
from ..models.conversation import Conversation
from ..models.ticket import Ticket, TicketCreate, TicketPriority, TicketUpdate
from ..services.conversation_service import ConversationService
from ..services.support_service import (
    SupportAccessDenied,
    SupportResourceNotFound,
    SupportService,
)
from ..services.ticket_service import TicketService


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Multi-Agent Customer Support (LangGraph)",
    description="基于 LangGraph 状态图的多Agent智能客服系统",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 懒加载支持服务（首次请求时才编译图/初始化LLM客户端）
_support_service: Optional[SupportService] = None
_support_service_lock = Lock()


def get_support_service() -> SupportService:
    global _support_service
    if _support_service is None:
        with _support_service_lock:
            if _support_service is None:
                _support_service = SupportService()
    return _support_service


def _support_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, SupportResourceNotFound):
        return HTTPException(status_code=404, detail="Requested resource was not found")
    if isinstance(exc, SupportAccessDenied):
        return HTTPException(status_code=403, detail="Resource access denied")
    return HTTPException(
        status_code=503, detail="Support service is temporarily unavailable"
    )


# ===== 基础端点 =====


@app.get("/")
async def root():
    return {"message": "Multi-Agent Customer Support API (LangGraph)", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ===== 工单端点 =====


@app.post("/api/tickets", response_model=Ticket)
def create_ticket(ticket: TicketCreate, db: Session = Depends(get_db)):
    return TicketService.create_ticket(db, ticket)


@app.get("/api/tickets", response_model=List[Ticket])
def get_tickets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return TicketService.get_all_tickets(db, skip=skip, limit=limit)


@app.get("/api/tickets/{ticket_id}", response_model=Ticket)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = TicketService.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.get("/api/tickets/customer/{customer_id}", response_model=List[Ticket])
def get_customer_tickets(customer_id: str, db: Session = Depends(get_db)):
    return TicketService.get_tickets_by_customer(db, customer_id)


@app.patch("/api/tickets/{ticket_id}", response_model=Ticket)
def update_ticket(
    ticket_id: int, ticket_update: TicketUpdate, db: Session = Depends(get_db)
):
    ticket = TicketService.update_ticket(db, ticket_id, ticket_update)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.delete("/api/tickets/{ticket_id}")
def delete_ticket(ticket_id: int, db: Session = Depends(get_db)):
    if not TicketService.delete_ticket(db, ticket_id):
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {"message": "Ticket deleted successfully"}


# ===== 对话端点 =====


@app.post("/api/conversations", response_model=Conversation)
def create_conversation(
    customer_id: str = Query(min_length=1, max_length=128),
    ticket_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    if ticket_id:
        ticket = TicketService.get_ticket(db, ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if ticket.customer_id != customer_id:
            raise HTTPException(status_code=403, detail="Resource access denied")
    return ConversationService.create_conversation(db, customer_id, ticket_id)


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    conversation = ConversationService.get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/api/conversations/ticket/{ticket_id}", response_model=Conversation)
def get_conversation_by_ticket(ticket_id: int, db: Session = Depends(get_db)):
    conversation = ConversationService.get_conversation_by_ticket(db, ticket_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


# ===== 支持端点（核心） =====


class MessageRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: Optional[int] = None
    ticket_id: Optional[int] = None
    priority: TicketPriority = TicketPriority.MEDIUM


@app.post("/api/support/message")
def handle_message(request: MessageRequest, db: Session = Depends(get_db)):
    """
    处理客户消息 — 执行 LangGraph 多Agent工作流。

    返回内容包含：
    - response: 最终回复
    - agents_used: 参与的Agent列表
    - metadata: 意图/情感/优先级/QA得分/检索文档等
    - trace: 完整执行轨迹（可用于前端可视化工作流）
    """
    try:
        service = get_support_service()
        return service.handle_customer_message(
            db=db,
            customer_id=request.customer_id,
            message=request.message,
            conversation_id=request.conversation_id,
            ticket_id=request.ticket_id,
            priority=request.priority.value,
        )
    except Exception as exc:
        db.rollback()
        raise _support_http_exception(exc) from exc


class TicketMessageRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(min_length=1, max_length=128)
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=10000)
    priority: TicketPriority = TicketPriority.MEDIUM


@app.post("/api/support/ticket")
def create_ticket_and_start_conversation(
    request: TicketMessageRequest, db: Session = Depends(get_db)
):
    """创建工单并开始对话。"""
    try:
        ticket = TicketService.create_ticket(
            db,
            TicketCreate(
                customer_id=request.customer_id,
                subject=request.subject,
                description=request.description,
                priority=request.priority,
            ),
            commit=False,
        )

        service = get_support_service()
        result = service.handle_customer_message(
            db=db,
            customer_id=request.customer_id,
            message=request.description,
            ticket_id=ticket.id,
            priority=request.priority.value,
        )
        updated_ticket = TicketService.get_ticket(db, ticket.id)
        return {"ticket": updated_ticket, "conversation": result}
    except Exception as exc:
        db.rollback()
        raise _support_http_exception(exc) from exc


# ===== 前端页面 =====


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    """聊天界面（含Agent执行轨迹可视化）。"""
    html_path = os.path.join(
        os.path.dirname(__file__), "../../frontend/templates/chat.html"
    )
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return "<h1>Chat interface not found</h1>"
