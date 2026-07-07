"""
命令行演示脚本 — 无需启动Web服务即可体验多Agent工作流。

用法：
    cp .env.example .env  # 填入你的 LLM_API_KEY
    python demo.py "我的API一直返回401错误怎么办"
    python demo.py        # 交互模式
"""

import sys

from backend.graph import build_support_graph


def run_once(graph, message: str, priority: str = "medium"):
    print(f"\n{'='*60}")
    print(f"客户消息: {message}  (优先级: {priority})")
    print(f"{'='*60}")

    result = graph.invoke(
        {
            "customer_id": "demo_user",
            "message": message,
            "priority": priority,
            "conversation_history": [],
            "ticket_info": None,
        }
    )

    print("\n--- 执行轨迹 ---")
    for line in result.get("trace", []):
        print(" ", line)

    print("\n--- 参与Agent ---")
    print(" ", " → ".join(result.get("agents_used", [])))

    print("\n--- 最终回复 ---")
    print(result.get("final_response", ""))
    print()


def main():
    graph = build_support_graph()

    if len(sys.argv) > 1:
        run_once(graph, " ".join(sys.argv[1:]))
        return

    print("多Agent客服演示（输入 exit 退出）")
    while True:
        try:
            msg = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() in ("exit", "quit"):
            break
        run_once(graph, msg)


if __name__ == "__main__":
    main()
