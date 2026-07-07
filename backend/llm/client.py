"""
LLM 客户端 — OpenAI 兼容接口，支持 DeepSeek / 通义千问 / Kimi / OpenAI 等。

国产大模型基本都提供 OpenAI 兼容的 API 格式，因此统一用 langchain-openai 的
ChatOpenAI，通过 base_url 切换提供商，无需改代码。

配置方式（.env）:
    LLM_API_KEY=sk-xxx
    LLM_BASE_URL=https://api.deepseek.com
    LLM_MODEL=deepseek-chat
"""

from langchain_openai import ChatOpenAI

from ..config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def create_llm(temperature: float = 0.7, model: str | None = None) -> ChatOpenAI:
    """
    创建 LLM 实例。

    Args:
        temperature: 采样温度。生成回复用较高值(0.7)，
                     意图分类/QA打分等结构化任务用低值(0.0)。
        model: 覆盖默认模型名（可选）。

    Returns:
        ChatOpenAI 实例（兼容任何 OpenAI 格式的 API）
    """
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=model or LLM_MODEL,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )
