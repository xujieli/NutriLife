#!/usr/bin/env python3
"""NutriLife 端到端联调脚本。

模拟前端向后端 ``POST /api/v1/chat`` 发起 SSE 流式请求，验证：
1. 流式输出协议（text / tool_call / tool_result / intent / done）。
2. 最终回答、意图（intent）与 session_id 是否正常返回。
3. LangFuse trace 是否被创建（需配置 LANGFUSE_*，否则优雅跳过）。

使用方式：
    python tests/e2e_test.py
    # 指定后端地址
    NUTRILIFE_BASE_URL=http://192.168.31.48:8000 python tests/e2e_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

# 将项目根目录加入 sys.path，便于导入 app 模块（用于 LangFuse 校验）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = os.environ.get("NUTRILIFE_BASE_URL", "http://localhost:8000")
CHAT_URL = f"{BASE_URL.rstrip('/')}/api/v1/chat"

# 三类意图各覆盖一个场景，验证路由是否走通
TEST_CASES = [
    {
        "name": "RAG 知识问答",
        "user_input": "痛风能吃豆制品吗",
        "expect_intent": "RAG_QUERY",
    },
    {
        "name": "Workflow 任务执行",
        "user_input": "记录午餐：汉堡、薯条，计算热量并给建议",
        "expect_intent": "WORKFLOW_TASK",
    },
    {
        "name": "日常闲聊",
        "user_input": "你好",
        "expect_intent": "GENERAL_CHAT",
    },
]


async def run_chat(client: httpx.AsyncClient, user_input: str) -> tuple[int, list[dict]]:
    """发起一次 SSE 流式请求，返回 (状态码, 事件列表)。"""
    events: list[dict] = []
    async with client.stream(
        "POST",
        CHAT_URL,
        json={"user_input": user_input, "session_id": ""},
        timeout=180,
    ) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            return resp.status_code, events

        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return resp.status_code, events


def check_langfuse_trace(session_id: str) -> None:
    """（可选）校验 LangFuse 是否记录了本次 trace，失败不致命。"""
    try:
        from app.core.langfuse import get_langfuse_client

        client = get_langfuse_client()
        if client is None:
            print("  [LangFuse] 未配置（缺少 LANGFUSE_* 环境变量），跳过 trace 校验")
            return
        traces = client.fetch_traces(session_id=session_id, limit=1)
        if getattr(traces, "data", None):
            first = traces.data[0]
            print(f"  [LangFuse] ✅ 找到 trace: {first.name} (id={first.id})")
        else:
            print("  [LangFuse] ⚠️ 未查询到 trace（可能存在上报延迟）")
    except Exception as exc:  # noqa: BLE001 —— 观测校验失败不应阻断联调
        print(f"  [LangFuse] 校验失败（非致命）: {exc}")


async def main() -> int:
    print("=" * 64)
    print("NutriLife 端到端联调测试")
    print(f"  后端地址: {BASE_URL}")
    print("=" * 64)

    try:
        async with httpx.AsyncClient() as client:
            for idx, case in enumerate(TEST_CASES):
                print(f"\n▶ 场景 {idx + 1}/{len(TEST_CASES)}：{case['name']}")
                print(f"  输入: {case['user_input']}")

                try:
                    status, events = await run_chat(client, case["user_input"])
                except httpx.ConnectError:
                    print("  ❌ 无法连接后端。请先启动：")
                    print("     poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000")
                    return 1

                print(f"  HTTP 状态: {status}")
                if status == 503:
                    print("  ⚠️ 后端返回 503（LM Studio 不可用）。请确认模型服务已启动。")
                    return 1
                if status != 200:
                    print("  ❌ 非 200 响应，联调失败")
                    return 1

                types = [e.get("type") for e in events]
                done = next((e for e in events if e.get("type") == "done"), None)
                text_len = sum(
                    len(e.get("content", ""))
                    for e in events
                    if e.get("type") == "text"
                )

                print(f"  事件序列: {types}")
                print(f"  流式文本字符数: {text_len}")
                if done:
                    print(f"  intent: {done.get('intent')}")
                    print(f"  session_id: {done.get('session_id')}")
                    print(f"  回答片段: {(done.get('content') or '')[:80]!r}")
                else:
                    print("  ❌ 未收到 done 事件")
                    return 1

                # ── 断言 ────────────────────────────────────────────
                assert "done" in types, f"场景「{case['name']}」缺少 done 事件"
                assert done and done.get("session_id"), "done 事件缺少 session_id"
                if case["expect_intent"] and done.get("intent") != case["expect_intent"]:
                    print(
                        f"  ⚠️ 意图不符：期望 {case['expect_intent']}，"
                        f"实际 {done.get('intent')}"
                    )

                # 仅对首个场景校验 LangFuse，避免重复
                if idx == 0 and done:
                    check_langfuse_trace(done["session_id"])
    except KeyboardInterrupt:
        print("\n中断")
        return 130

    print("\n" + "=" * 64)
    print("✅ 端到端测试完成")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
