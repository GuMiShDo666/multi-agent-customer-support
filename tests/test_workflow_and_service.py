import inspect

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api import main as api
from backend.graph import nodes
from backend.graph.builder import build_support_graph
from backend.graph.routing import route_after_qa, route_to_agent
from backend.models.conversation import ConversationDB, MessageDB
from backend.models.ticket import (
    Base,
    TicketCreate,
    TicketDB,
    TicketPriority,
    TicketStatus,
)
from backend.services.conversation_service import ConversationService
from backend.services.support_service import (
    SupportAccessDenied,
    SupportResourceNotFound,
    SupportService,
)
from backend.services.ticket_service import TicketService


class SuccessfulGraph:
    def invoke(self, state):
        return {
            "final_response": "ok",
            "handled_by": "support_agent",
            "intent": "general",
            "intent_confidence": 1.0,
            "sentiment": "neutral",
            "predicted_priority": state["priority"],
            "qa_score": 9.0,
            "retry_count": 0,
            "retrieved_docs": [],
            "agents_used": ["support_agent", "qa_agent"],
            "trace": [],
        }


class FailingGraph:
    def invoke(self, state):
        raise RuntimeError("simulated upstream failure")


class StaticLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, messages):
        return AIMessage(content=self.content)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def make_service(graph) -> SupportService:
    service = SupportService.__new__(SupportService)
    service.graph = graph
    return service


def test_graph_compiles_without_api_key():
    assert build_support_graph() is not None


def test_missing_conversation_is_reported(db):
    service = make_service(SuccessfulGraph())

    with pytest.raises(SupportResourceNotFound, match="Conversation not found"):
        service.handle_customer_message(
            db, "customer-a", "hello", conversation_id=999
        )

    assert db.query(MessageDB).count() == 0


def test_customer_cannot_write_to_another_conversation(db):
    conversation = ConversationService.create_conversation(db, "customer-a")
    service = make_service(SuccessfulGraph())

    with pytest.raises(SupportAccessDenied):
        service.handle_customer_message(
            db, "customer-b", "cross-customer message", conversation_id=conversation.id
        )

    assert db.query(MessageDB).count() == 0


def test_customer_cannot_use_another_customers_ticket(db):
    ticket = TicketService.create_ticket(
        db,
        TicketCreate(
            customer_id="customer-a",
            subject="subject",
            description="description",
            priority=TicketPriority.MEDIUM,
        ),
    )
    service = make_service(SuccessfulGraph())

    with pytest.raises(SupportAccessDenied):
        service.handle_customer_message(
            db, "customer-b", "message", ticket_id=ticket.id
        )

    assert db.query(ConversationDB).count() == 0


def test_graph_failure_rolls_back_conversation_and_message(db):
    service = make_service(FailingGraph())

    with pytest.raises(RuntimeError, match="simulated upstream failure"):
        service.handle_customer_message(db, "customer-a", "message")

    assert db.query(ConversationDB).count() == 0
    assert db.query(MessageDB).count() == 0


def test_successful_workflow_commits_both_messages(db):
    service = make_service(SuccessfulGraph())

    result = service.handle_customer_message(db, "customer-a", "message")

    messages = ConversationService.get_messages(db, result["conversation_id"])
    assert [message.role for message in messages] == ["user", "assistant"]


def test_invalid_qa_output_fails_closed(monkeypatch):
    monkeypatch.setattr(nodes, "create_llm", lambda **kwargs: StaticLLM("not-json"))

    result = nodes.qa_check(
        {
            "message": "question",
            "draft_response": "answer",
            "retrieved_docs": [],
            "retry_count": 0,
            "agents_used": [],
        }
    )

    assert result["qa_score"] == 0.0
    assert result["retry_count"] == 1
    assert route_after_qa({**result, "handled_by": "support_agent"}) == "support_agent"


def test_invalid_classification_output_routes_to_escalation(monkeypatch):
    monkeypatch.setattr(nodes, "create_llm", lambda **kwargs: StaticLLM("not-json"))

    result = nodes.classify_intent(
        {
            "message": "unknown request",
            "priority": "medium",
            "conversation_history": [],
        }
    )

    assert result["intent_confidence"] == 0.0
    assert result["predicted_priority"] == "high"
    assert route_to_agent(result) == "escalation_agent"


@pytest.mark.parametrize(
    "payload",
    [
        {"customer_id": "", "message": "hello"},
        {"customer_id": "customer-a", "message": ""},
        {
            "customer_id": "customer-a",
            "message": "hello",
            "priority": "not-a-priority",
        },
    ],
)
def test_message_request_rejects_invalid_input(payload):
    with pytest.raises(ValidationError):
        api.MessageRequest(**payload)


def test_support_endpoint_is_sync_and_hides_internal_errors(monkeypatch, db):
    class ErrorService:
        def handle_customer_message(self, **kwargs):
            raise KeyError("missing graph result field")

    monkeypatch.setattr(api, "get_support_service", lambda: ErrorService())

    assert not inspect.iscoroutinefunction(api.handle_message)
    with pytest.raises(HTTPException) as exc_info:
        api.handle_message(
            api.MessageRequest(customer_id="customer-a", message="hello"), db
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Support service is temporarily unavailable"


def test_ticket_workflow_commits_ticket_conversation_and_messages(monkeypatch, db):
    monkeypatch.setattr(
        api, "get_support_service", lambda: make_service(SuccessfulGraph())
    )

    result = api.create_ticket_and_start_conversation(
        api.TicketMessageRequest(
            customer_id="customer-a",
            subject="subject",
            description="description",
        ),
        db,
    )

    assert result["ticket"].status == TicketStatus.IN_PROGRESS
    assert db.query(TicketDB).count() == 1
    assert db.query(ConversationDB).count() == 1
    assert db.query(MessageDB).count() == 2


def test_ticket_workflow_failure_rolls_back_ticket(monkeypatch, db):
    class ErrorService:
        def handle_customer_message(self, **kwargs):
            raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr(api, "get_support_service", lambda: ErrorService())

    with pytest.raises(HTTPException) as exc_info:
        api.create_ticket_and_start_conversation(
            api.TicketMessageRequest(
                customer_id="customer-a",
                subject="subject",
                description="description",
            ),
            db,
        )

    assert exc_info.value.status_code == 503
    assert db.query(TicketDB).count() == 0
    assert db.query(ConversationDB).count() == 0
