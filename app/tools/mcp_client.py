"""基于 langchain-mcp-adapters 的 Open Food Facts MCP 集成。

通过 ``langchain_mcp_adapters.client.MultiServerMCPClient`` 连接远程
Open Food Facts MCP 服务，并把服务暴露的工具转换为 LangChain 工具后调用。

适配器在每次工具调用时都会自动建立并释放一个 ``ClientSession``，因此本模块
不需要维护持久连接，也没有需要显式关闭的资源。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger

from app.core.config import get_settings


class OpenFoodFactsMCPError(RuntimeError):
    """Open Food Facts MCP 服务调用失败。"""


_SERVER_NAME = "openfoodfacts"

_client: MultiServerMCPClient | None = None
_tools_by_name: dict[str, BaseTool] | None = None


def _build_client() -> MultiServerMCPClient:
    """根据配置构造 Open Food Facts 的 ``MultiServerMCPClient``。"""
    settings = get_settings().openfoodfacts_mcp
    return MultiServerMCPClient(
        {
            _SERVER_NAME: {
                "transport": "streamable_http",
                "url": f"{settings.url.rstrip('/')}/mcp",
                "timeout": settings.timeout_seconds,
            }
        },
        # 与旧实现的 ``call_tool`` 语义保持一致：MCP 返回 isError 时抛出异常，
        # 而不是被适配器转换成 error ToolMessage。
        handle_tool_errors=False,
    )


def _get_client() -> MultiServerMCPClient:
    """获取进程内共享的 MCP 客户端。"""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


async def _get_tools() -> dict[str, BaseTool]:
    """懒加载并缓存 Open Food Facts MCP 暴露的 LangChain 工具。"""
    global _tools_by_name
    if _tools_by_name is None:
        tools = await _get_client().get_tools()
        _tools_by_name = {tool.name: tool for tool in tools}
        logger.debug("Open Food Facts MCP 工具已加载: {}", list(_tools_by_name))
    return _tools_by_name


def _extract_json(result: Any, tool_name: str) -> Any:
    """把 MCP 工具调用结果解析为 JSON 对象。

    适配器在 ``tool_call_id`` 缺省时直接返回内容块列表（而非 ``ToolMessage``），
    因此这里从文本内容块中拼接出文本，再做 JSON 解析。
    """
    if isinstance(result, str):
        text = result
    elif isinstance(result, list):
        text = "\n".join(
            item["text"]
            for item in result
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    else:
        raise OpenFoodFactsMCPError(
            f"MCP 工具 {tool_name} 返回了无法解析的结果类型: {type(result).__name__}"
        )

    text = text.strip()
    if not text:
        raise OpenFoodFactsMCPError(f"MCP 工具 {tool_name} 未返回文本内容")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenFoodFactsMCPError(
            f"MCP 工具 {tool_name} 返回内容不是有效 JSON: {text[:300]}"
        ) from exc


async def call_tool_json(
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """调用 MCP 工具，并将返回的文本内容解析为 JSON。"""
    tools = await _get_tools()
    tool = tools.get(name)
    if tool is None:
        raise OpenFoodFactsMCPError(f"未找到 MCP 工具: {name}")

    try:
        result = await tool.ainvoke(arguments or {})
    except Exception as exc:
        raise OpenFoodFactsMCPError(f"MCP 工具 {name} 调用失败: {exc}") from exc

    return _extract_json(result, name)
