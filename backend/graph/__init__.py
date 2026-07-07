"""LangGraph 状态图模块.

注意：build_support_graph 采用懒加载，避免在仅使用 routing/state
（如单元测试）时强制依赖 langgraph。
"""

from .state import SupportState

__all__ = ["build_support_graph", "SupportState"]


def __getattr__(name):
    if name == "build_support_graph":
        from .builder import build_support_graph

        return build_support_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
