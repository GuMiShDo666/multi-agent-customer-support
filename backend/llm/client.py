"""
使用 ChatOpenAI 连接 OpenAI 兼容接口，通过 base_url 切换提供商。

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
