"""PARALLEL_ANALYSIS 子图：并行子 Agent 协作的「综合膳食评估」场景。

编排拓扑：

    coordinator ──(asyncio.gather 并行 fan-out)──▶ 3 个维度子 Agent ──▶ reviewer ──▶ END

核心设计（消息 / 产物 / 事件日志 三层职责分离）：

    1. **消息只传引用，不传产物**：子 Agent 把报告作为 ``Artifact`` 写入
       ``ArtifactStore``，仅返回一条 ``CoordinationMessage``，其中只携带
       ``ArtifactRef``（artifact_id / version / content_hash / 可选 pointer），
       绝不把完整报告塞进消息。
    2. **Reviewer 需要细节时按引用局部读取**：先看消息摘要定位，再通过
       ``store.read_ref`` / ``store.read_pointer`` 读取原文或指定段落；
       读取时校验版本与哈希，防止「验收 v3 时底层已悄悄变成 v4」。
    3. **事件日志负责审计**：分配任务、写入产物、验收产物等关键判断全部写入
       ``EventLog``，记录「谁在何时基于哪个版本做了什么判断」。

三个并行维度子 Agent（各自独立、互不依赖）：
    - ``calorie``  热量与宏量营养素分析
    - ``risk``     疾病饮食禁忌审查（如痛风）
    - ``balance``  膳食结构均衡评分
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from loguru import logger

from app.core.artifact_store import (
    ArtifactStore,
    EventLog,
    get_artifact_store,
    get_event_log,
)
from app.core.llm import get_chat_llm
from app.schemas.parallel import (
    Artifact,
    ArtifactKind,
    ArtifactRef,
    AuditEvent,
    CoordinationMessage,
    MessageStatus,
    ReportSection,
    SubAgentReport,
)
from app.schemas.state import AgentState, get_latest_user_text

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


# ──────────────────────────────────────────────────────────────
# 维度定义与子 Agent Prompt
# ──────────────────────────────────────────────────────────────

_CALORIE_PROMPT = """\
你是「热量与宏量营养素分析」子 Agent。基于用户给出的饮食记录，估算总热量，
并从蛋白质/脂肪/碳水角度给出宏量营养素结构要点。

输出结构要求（不要输出 JSON，按结构化字段返回）：
- summary: 一句话摘要（不超过 60 字）。
- sections 必须包含以下 key（顺序固定）：
  1. findings   —— 主要发现（总热量估算、营养素结构）。
  2. evidence   —— 结论依据（引用了输入中的哪些食物与分量）。
  3. suggestions—— 改进建议。
"""

_RISK_PROMPT = """\
你是「疾病饮食禁忌审查」子 Agent。结合用户已知的健康状况（如痛风、高尿酸、
糖尿病等，若输入未提及则按一般成人审查），审查饮食记录中是否存在禁忌风险。

输出结构要求：
- summary: 一句话摘要（不超过 60 字）。
- sections 必须包含以下 key（顺序固定）：
  1. findings —— 主要发现（是否存在禁忌风险）。
  2. risks    —— 具体风险点与对应食物（逐条列出）。
  3. suggestions —— 规避建议。
"""

_BALANCE_PROMPT = """\
你是「膳食结构均衡评分」子 Agent。从食物多样性、营养素均衡角度对饮食记录
打分（0~100），并给出理由。

输出结构要求：
- summary: 一句话摘要（不超过 60 字）。
- sections 必须包含以下 key（顺序固定）：
  1. findings —— 主要发现。
  2. score    —— 评分（0~100）与评分理由。
  3. suggestions —— 改进建议。
"""

# (key, 标题, system prompt)
DIMENSIONS: list[tuple[str, str, str]] = [
    ("calorie", "热量与宏量营养素分析", _CALORIE_PROMPT),
    ("risk", "疾病饮食禁忌审查", _RISK_PROMPT),
    ("balance", "膳食结构均衡评分", _BALANCE_PROMPT),
]

_REVIEW_PROMPT = """\
你是 NutriLife 的综合评估 Reviewer。以下是三个并行子 Agent 产出的证据报告，
请基于这些证据（而非你自己的臆测）整合出一份简洁、连贯的综合评估结论。

证据报告：
{evidence}

要求：
1. 用中文，3~6 句话。
2. 只使用证据中已经出现的事实与结论，不要编造。
3. 若某个子 Agent 失败或证据缺失，如实说明该维度未能评估。
4. 涉及疾病或体重管理时，提醒用户以专业医生/营养师意见为准。
"""


# ──────────────────────────────────────────────────────────────
# 子 Agent 报告合成（LLM 结构化输出 + 本地兜底）
# ──────────────────────────────────────────────────────────────

def _fallback_report(dimension_key: str, question: str) -> SubAgentReport:
    """子 Agent 报告合成的本地兜底（LLM 不可用时）。"""
    evidence = f"原始输入：{question[:200]}"
    if dimension_key == "calorie":
        return SubAgentReport(
            summary="热量分析（本地兜底）",
            sections=[
                ReportSection(
                    key="findings", title="主要发现",
                    body="未能精确解析饮食记录，无法可靠估算总热量。",
                ),
                ReportSection(key="evidence", title="结论依据", body=evidence),
                ReportSection(
                    key="suggestions", title="改进建议",
                    body="请提供更明确的食物名称与分量，以便精确估算热量。",
                ),
            ],
        )
    if dimension_key == "risk":
        return SubAgentReport(
            summary="禁忌审查（本地兜底）",
            sections=[
                ReportSection(
                    key="findings", title="主要发现",
                    body="未完成禁忌审查，缺少可核对的健康状况信息。",
                ),
                ReportSection(key="risks", title="风险点", body="暂无结论。"),
                ReportSection(
                    key="suggestions", title="规避建议",
                    body="请补充健康状况（如痛风、糖尿病等），再行审查。",
                ),
            ],
        )
    return SubAgentReport(
        summary="均衡评分（本地兜底）",
        sections=[
            ReportSection(
                key="findings", title="主要发现",
                body="未能完成均衡评分，缺少可解析的食物结构。",
            ),
            ReportSection(key="score", title="评分", body="暂无评分。"),
            ReportSection(
                key="suggestions", title="改进建议",
                body="请提供更完整的饮食记录以便评分。",
            ),
        ],
    )


async def _synthesize_report(
    dimension_key: str, prompt: str, question: str
) -> SubAgentReport:
    """调用 LLM 合成子 Agent 报告，失败则降级为本地兜底报告。"""
    try:
        llm = get_chat_llm(temperature=0.0, streaming=False)
        structured_llm = llm.with_structured_output(SubAgentReport)
        result: SubAgentReport = cast(
            SubAgentReport,
            await structured_llm.ainvoke(
                [SystemMessage(content=prompt), HumanMessage(content=question)]
            ),
        )
        if not result.sections:
            raise ValueError("子 Agent 返回了空报告")
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Parallel[{}]: 报告合成失败，使用本地兜底: {}", dimension_key, exc
        )
        return _fallback_report(dimension_key, question)


# ──────────────────────────────────────────────────────────────
# 子 Agent 执行（产物写存储，消息只回传引用）
# ──────────────────────────────────────────────────────────────

async def _run_dimension(
    run_id: str,
    dimension_key: str,
    dimension_title: str,
    prompt: str,
    question: str,
    store: ArtifactStore,
    event_log: EventLog,
) -> CoordinationMessage:
    """执行单个维度子 Agent。

    关键点：子 Agent 产出报告后写入 ``ArtifactStore``，返回的协调消息里
    只放 ``ArtifactRef``（引用），不携带报告原文。
    """
    agent_name = f"subagent:{dimension_key}"
    await event_log.record(
        run_id,
        AuditEvent(
            event_id=uuid.uuid4().hex,
            actor=agent_name,
            action=f"开始处理维度「{dimension_title}」",
            detail="status=running",
        ),
    )

    try:
        report = await _synthesize_report(dimension_key, prompt, question)
        artifact = Artifact.build(
            artifact_id=f"{run_id}:{dimension_key}",
            kind=ArtifactKind.REPORT,
            producer=agent_name,
            summary=report.summary,
            content={"sections": [s.model_dump() for s in report.sections]},
        )
        ref = await store.put(artifact)

        await event_log.record(
            run_id,
            AuditEvent(
                event_id=uuid.uuid4().hex,
                actor=agent_name,
                action=f"产出并写入产物（{dimension_title}）",
                artifact_ref=ref,
                detail=f"summary={report.summary}",
            ),
        )
        return CoordinationMessage(
            message_id=uuid.uuid4().hex,
            from_agent=agent_name,
            to_agent="reviewer",
            status=MessageStatus.SUCCEEDED,
            summary=report.summary,
            refs=[ref],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Parallel[{}]: 子 Agent 执行失败: {}", dimension_key, exc)
        await event_log.record(
            run_id,
            AuditEvent(
                event_id=uuid.uuid4().hex,
                actor=agent_name,
                action=f"处理维度失败（{dimension_title}）",
                detail=str(exc),
            ),
        )
        return CoordinationMessage(
            message_id=uuid.uuid4().hex,
            from_agent=agent_name,
            to_agent="reviewer",
            status=MessageStatus.FAILED,
            summary=f"{dimension_title} 处理失败",
            failure_reason=str(exc),
        )


# ──────────────────────────────────────────────────────────────
# LangGraph 节点
# ──────────────────────────────────────────────────────────────

async def coordinator_node(state: AgentState) -> dict[str, Any]:
    """协调者节点：创建运行上下文，并行 fan-out 三个维度子 Agent。

    并行通过 ``asyncio.gather`` 实现；子 Agent 之间无依赖，可同时执行。
    """
    question = get_latest_user_text(state)
    run_id = f"parallel-{uuid.uuid4().hex[:12]}"
    store = get_artifact_store()
    event_log = get_event_log()

    logger.info("Parallel.coordinator: run_id={} 输入 '{}'", run_id, question[:60])

    for key, title, _ in DIMENSIONS:
        await event_log.record(
            run_id,
            AuditEvent(
                event_id=uuid.uuid4().hex,
                actor="coordinator",
                action=f"分配任务：{title}",
                detail=f"dimension={key}",
            ),
        )

    # 并行执行三个维度子 Agent
    messages: list[CoordinationMessage] = await asyncio.gather(
        *[
            _run_dimension(run_id, key, title, prompt, question, store, event_log)
            for key, title, prompt in DIMENSIONS
        ]
    )

    refs = [ref for msg in messages for ref in msg.refs]
    logger.info(
        "Parallel.coordinator: 完成 {} 个维度，成功 {} / 失败 {}",
        len(messages),
        sum(1 for m in messages if m.status == MessageStatus.SUCCEEDED),
        sum(1 for m in messages if m.status == MessageStatus.FAILED),
    )

    return {
        "parallel_run_id": run_id,
        "parallel_messages": [m.model_dump() for m in messages],
        "parallel_artifacts": [r.model_dump() for r in refs],
        "parallel_audit_trail": [e.model_dump() for e in event_log.snapshot(run_id)],
    }


def _artifact_to_text(artifact: Artifact) -> str:
    """把产物内容（sections 结构）转换为可注入 Reviewer 的文本证据。"""
    content = artifact.content
    if isinstance(content, dict):
        sections = content.get("sections", [])
        lines: list[str] = []
        for section in sections:
            if isinstance(section, dict):
                title = section.get("title") or section.get("key") or "未命名"
                body = str(section.get("body", ""))
                lines.append(f"【{title}】\n{body}")
        return "\n\n".join(lines)
    return str(content)


def _fallback_review(parts: list[str]) -> str:
    """Reviewer 综合结论的本地兜底（LLM 不可用时）。"""
    body = "\n".join(f"- {p}" for p in parts if p)
    return (
        "已完成并行评估，各维度摘要如下（本地兜底，未调用 LLM 做综合）：\n"
        + (body or "暂无可用结论。")
    )


async def reviewer_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Reviewer 节点：按引用读取产物原文，校验版本后综合成最终结论。

    关键点：
        - 消息摘要只用于「定位」；下结论前必须按引用读取原文（证据）。
        - 读取用 ``store.read_ref``（校验版本/哈希）；需要细节时用
          ``store.read_pointer`` 局部读取对应段落。
        - 每次验收都写审计事件，记录「基于哪个版本」。
    """
    run_id = str(state.get("parallel_run_id", ""))
    store = get_artifact_store()
    event_log = get_event_log()

    messages = [
        CoordinationMessage.model_validate(m)
        for m in state.get("parallel_messages", [])
    ]

    evidence_blocks: list[str] = []
    risk_ref: ArtifactRef | None = None

    for msg in messages:
        if msg.status == MessageStatus.FAILED:
            await event_log.record(
                run_id,
                AuditEvent(
                    event_id=uuid.uuid4().hex,
                    actor="reviewer",
                    action=f"标记子 Agent 失败（{msg.from_agent}）",
                    detail=msg.failure_reason,
                ),
            )
            evidence_blocks.append(f"[{msg.from_agent} 失败] {msg.failure_reason}")
            continue

        for ref in msg.refs:
            # 先看摘要定位，再读原文作为证据；read_ref 内部校验版本/哈希。
            artifact = await store.read_ref(ref)
            await event_log.record(
                run_id,
                AuditEvent(
                    event_id=uuid.uuid4().hex,
                    actor="reviewer",
                    action=f"验收产物 {artifact.artifact_id}",
                    artifact_ref=ref,
                    detail=f"summary={artifact.summary}",
                ),
            )
            evidence_blocks.append(
                f"[{msg.from_agent}] {_artifact_to_text(artifact)}"
            )
            # 记录风险报告引用，供下面演示「按指针局部读取对应段落」。
            if "risk" in ref.artifact_id:
                risk_ref = ref

    # 演示：Reviewer 需要细节时，按引用局部读取「风险点」段落（而非整份拉取）。
    if risk_ref is not None:
        try:
            risk_section = await store.read_pointer(risk_ref, "risks")
            if isinstance(risk_section, dict) and risk_section.get("body"):
                await event_log.record(
                    run_id,
                    AuditEvent(
                        event_id=uuid.uuid4().hex,
                        actor="reviewer",
                        action="局部读取风险点段落",
                        artifact_ref=risk_ref.model_copy(update={"pointer": "risks"}),
                        detail="pointer=risks",
                    ),
                )
                logger.info(
                    "Parallel.reviewer: 局部读取 risks 段落 → '{}'",
                    str(risk_section.get("body"))[:60],
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Parallel.reviewer: 局部读取 risks 失败: {}", exc)

    evidence_text = "\n\n".join(evidence_blocks) or "（无可用证据）"

    # 综合生成最终回答（流式，供 astream_events 捕获 token）
    try:
        llm = get_chat_llm(temperature=0.3)
        chunks: list[str] = []
        review_messages: list[BaseMessage] = [
            SystemMessage(content=_REVIEW_PROMPT.format(evidence=evidence_text)),
        ]
        async for chunk in llm.astream(review_messages, config=config):
            if isinstance(chunk.content, str):
                chunks.append(chunk.content)
        answer = "".join(chunks).strip()
        if not answer:
            raise ValueError("Reviewer 返回空回答")
    except Exception as exc:  # noqa: BLE001
        logger.error("Parallel.reviewer: 综合生成失败，使用本地兜底: {}", exc)
        answer = _fallback_review([m.summary for m in messages])

    await event_log.record(
        run_id,
        AuditEvent(
            event_id=uuid.uuid4().hex,
            actor="reviewer",
            action="生成最终综合结论",
            detail=f"evidence_blocks={len(evidence_blocks)}",
        ),
    )

    logger.info("Parallel.reviewer: 最终回答 '{}'", answer[:60])
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "parallel_audit_trail": [e.model_dump() for e in event_log.snapshot(run_id)],
    }


def build_parallel_analysis_graph() -> CompiledStateGraph:
    """构建并编译 PARALLEL_ANALYSIS 子图。"""
    graph = StateGraph(AgentState)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("coordinator")
    graph.add_edge("coordinator", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()
