from backend.graph.routing import route_after_qa, route_to_agent
from backend.rag.retriever import BM25Retriever

def test_route_technical_intent():
    state = {"intent": "technical", "predicted_priority": "medium", "sentiment": "neutral"}
    assert route_to_agent(state) == "technical_agent"


def test_route_high_priority_to_escalation():
    state = {"intent": "general", "predicted_priority": "high", "sentiment": "neutral"}
    assert route_to_agent(state) == "escalation_agent"


def test_route_angry_customer_to_escalation():
    state = {"intent": "general", "predicted_priority": "low", "sentiment": "angry"}
    assert route_to_agent(state) == "escalation_agent"


def test_route_complaint_to_escalation():
    state = {"intent": "complaint", "predicted_priority": "medium", "sentiment": "negative"}
    assert route_to_agent(state) == "escalation_agent"


def test_route_general_to_support():
    state = {"intent": "general", "predicted_priority": "medium", "sentiment": "neutral"}
    assert route_to_agent(state) == "support_agent"


def test_qa_pass_goes_to_finalize():
    state = {"qa_score": 8.5, "retry_count": 0, "handled_by": "support_agent"}
    assert route_after_qa(state) == "finalize"


def test_qa_fail_retries_same_agent():
    state = {"qa_score": 5.0, "retry_count": 1, "handled_by": "technical_agent"}
    assert route_after_qa(state) == "technical_agent"


def test_qa_fail_but_max_retries_exceeded():
    state = {"qa_score": 5.0, "retry_count": 99, "handled_by": "support_agent"}
    assert route_after_qa(state) == "finalize"


DOCS = [
    {"title": "API 401错误", "content": "API Key无效或过期导致401 Unauthorized"},
    {"title": "退款政策", "content": "订阅后7天内可无理由全额退款"},
    {"title": "密码重置", "content": "点击忘记密码，通过邮箱重置"},
]


def test_bm25_retrieves_relevant_doc():
    retriever = BM25Retriever(DOCS)
    results = retriever.search("API返回401错误怎么办", top_k=1)
    assert len(results) == 1
    assert "401" in results[0]["title"]


def test_bm25_scores_positive():
    retriever = BM25Retriever(DOCS)
    results = retriever.search("退款", top_k=3)
    assert all(float(r["score"]) > 0 for r in results)


def test_bm25_top_k_limit():
    retriever = BM25Retriever(DOCS)
    results = retriever.search("退款 密码 API", top_k=2)
    assert len(results) <= 2
