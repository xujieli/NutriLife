"""并行子 Agent 协作场景（综合膳食评估）的数据模型。

本模块承载「消息 / 产物 / 事件日志」三层职责分离的核心类型定义：

    - ``Artifact`` / ``ArtifactRef``（产物层）：承载真实内容（报告、补丁、
      日志快照、测试结果、结构化数据）。Artifact 一旦写入即不可变，必须带
      ``version`` 与 ``content_hash``。Reviewer 按引用读取时校验二者，杜绝
      「验收 v3 时底层已悄悄变成 v4」的竞态。
    - ``CoordinationMessage``（消息层）：只承载协调信息——任务状态、简短摘要、
      失败原因、证据引用。绝不携带完整产物，避免消息体随产物膨胀。
    - ``AuditEvent``（事件日志层）：记录谁（actor）在何时（timestamp）
      基于哪个版本（artifact_ref）做了什么判断（action），用于事后审计。

关键不变式：
    1. 消息只传 ``ArtifactRef``（引用），不传 ``Artifact.content``（内容）。
    2. 摘要（summary）只用于「定位」，不能替代原始证据；需要细节时必须按
       引用读取 ``Artifact.content``。
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ArtifactKind(StrEnum):
    """产物类型枚举。"""

    REPORT = "report"
    PATCH = "patch"
    LOG_SNAPSHOT = "log_snapshot"
    TEST_RESULT = "test_result"
    STRUCTURED_DATA = "structured_data"


class MessageStatus(StrEnum):
    """协调消息中的任务状态枚举。"""

    ASSIGNED = "assigned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串（用于时间戳字段）。"""
    return datetime.now(UTC).isoformat()


def compute_content_hash(content: Any) -> str:
    """对产物内容计算 sha256 哈希（先做规范序列化，保证跨序列化稳定）。"""
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ArtifactRef(BaseModel):
    """产物引用：消息层携带的最小定位凭据（只有引用，没有内容）。

    ``content_hash`` 用于读取时校验底层内容未被替换/篡改；
    ``pointer`` 是可选的局部引用，指向产物内的某个段落/证据 key，
    供 Reviewer 按需「局部读取对应段落」。
    """

    artifact_id: str = Field(description="产物唯一 ID")
    version: int = Field(ge=1, description="产物版本号（不可变）")
    content_hash: str = Field(description="产物内容 sha256，读取时校验一致性")
    pointer: str | None = Field(
        default=None,
        description="可选局部引用，指向产物内的段落/证据 key（按需局部读取）",
    )


class Artifact(BaseModel):
    """不可变产物：承载真实内容，带版本与内容哈希。

    内容写入后即视为不可变快照；同一 ``artifact_id`` 若内容发生变化，
    由 ``ArtifactStore`` 递增版本号形成新快照，旧版本不复存在。
    """

    artifact_id: str = Field(description="产物唯一 ID")
    kind: ArtifactKind = Field(default=ArtifactKind.REPORT, description="产物类型")
    version: int = Field(default=1, ge=1, description="版本号")
    content_hash: str = Field(description="内容 sha256")
    producer: str = Field(description="产出该产物的子 Agent")
    created_at: str = Field(description="创建时间（UTC ISO 8601）")
    summary: str = Field(description="简短摘要，仅用于定位，不能替代原文")
    content: Any = Field(default=None, description="完整产物内容（不可变）")

    @classmethod
    def build(
        cls,
        artifact_id: str,
        *,
        kind: ArtifactKind,
        producer: str,
        summary: str,
        content: Any,
        version: int = 1,
    ) -> Artifact:
        """构造产物：自动计算内容哈希并打上时间戳。"""
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            version=version,
            content_hash=compute_content_hash(content),
            producer=producer,
            created_at=utc_now_iso(),
            summary=summary,
            content=content,
        )

    def ref(self, pointer: str | None = None) -> ArtifactRef:
        """生成指向本产物的引用（不含内容）。"""
        return ArtifactRef(
            artifact_id=self.artifact_id,
            version=self.version,
            content_hash=self.content_hash,
            pointer=pointer,
        )


class ReportSection(BaseModel):
    """报告中可被局部引用的段落/证据单元。"""

    key: str = Field(description="段落 key，供 ArtifactRef.pointer 局部引用")
    title: str = Field(description="段落标题")
    body: str = Field(description="段落正文")


class SubAgentReport(BaseModel):
    """子 Agent 产出的结构化报告（作为 Artifact.content 的载体）。"""

    summary: str = Field(description="报告简短摘要")
    sections: list[ReportSection] = Field(
        default_factory=list,
        description="报告分段内容（findings / evidence / risks / suggestions 等）",
    )


class CoordinationMessage(BaseModel):
    """协调消息：只承载状态、摘要、失败原因与证据引用，不承载内容。"""

    message_id: str = Field(description="消息唯一 ID")
    from_agent: str = Field(description="发送方（协调者 / 子 Agent / Reviewer）")
    to_agent: str = Field(description="接收方")
    status: MessageStatus = Field(description="任务状态")
    summary: str = Field(description="简短摘要，用于定位，不能替代原文")
    failure_reason: str = Field(default="", description="失败原因（成功时为空）")
    refs: list[ArtifactRef] = Field(
        default_factory=list, description="证据引用列表（不含内容）"
    )
    timestamp: str = Field(default_factory=utc_now_iso, description="消息时间戳")


class AuditEvent(BaseModel):
    """审计事件：谁在何时基于哪个版本做了什么判断。"""

    event_id: str = Field(description="事件唯一 ID")
    timestamp: str = Field(default_factory=utc_now_iso, description="事件时间戳")
    actor: str = Field(description="谁（coordinator / 子 Agent / reviewer）")
    action: str = Field(description="做了什么判断/操作")
    artifact_ref: ArtifactRef | None = Field(
        default=None, description="该判断所依据的产物版本（无则 None）"
    )
    detail: str = Field(default="", description="补充说明")
