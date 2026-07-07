"""
全局配置 — 通过环境变量/.env 文件配置。
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv未安装时直接读环境变量
    pass

# ===== LLM 配置（OpenAI兼容接口） =====
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ===== 数据库 =====
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./customer_support.db")

# ===== 工作流参数 =====
# QA质检阈值（0-10分），低于该分触发重新生成
QA_THRESHOLD = float(os.getenv("QA_THRESHOLD", "7.0"))
# QA不通过时的最大重试次数
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
# RAG检索返回文档数
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
