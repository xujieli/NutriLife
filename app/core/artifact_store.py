"""产物存储与审计事件日志。

设计要点：
    - ``ArtifactStore`` 是「追加式不可变存储」：同一 ``artifact_id`` 再次写入
      内容不同时自动递增版本号；内容相同时幂等返回原版本。Reviewer 通过引用
      读取时，若底层版本/哈希已变化，抛出 ``ArtifactVersionConflictError``，
      杜绝「验收 v3 时底层已悄悄变成 v4」。
    - ``EventLog`` 是「只追加」审计日志，记录谁在何时基于哪个版本做了什么判断。
    - ``resolve_pointer`` 支持按引用「局部读取对应段落/证据」，让 Reviewer
      不必整份拉取、而是按需取用细节。
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from app.schemas.parallel import Artifact, ArtifactRef, AuditEvent


class ArtifactVersionConflictError(RuntimeError):
    """引用与底层产物版本/哈希不一致（产物已被替换或变更）。"""


def resolve_pointer(content: Any, pointer: str | None) -> Any:
    """按 ``pointer`` 从产物内容中局部读取对应段落/证据。

    支持两种内容形态：
        - ``{"sections": [{"key": ..., ...}, ...]}``：按 section 的 key 匹配；
        - 普通 dict：按顶层 key 匹配。
    找不到目标时返回原文，保证不丢失证据。
    """
    if not pointer:
        return content
    if isinstance(content, dict):
        for section in content.get("sections", []):
            if isinstance(section, dict) and section.get("key") == pointer:
                return section
        if pointer in content:
            return content[pointer]
    return content


class ArtifactStore:
    """进程内不可变产物存储（单例，按 artifact_id 命名空间）。"""

    def __init__(self) -> None:
        self._latest: dict[str, Artifact] = {}
        self._lock = asyncio.Lock()

    async def put(self, artifact: Artifact) -> ArtifactRef:
        """写入产物并返回引用（追加式不可变语义）。

        规则：
            - 首次写入 → 版本取 ``artifact.version``（通常为 1）；
            - 内容相同重复写入 → 幂等，返回已有版本引用，不递增；
            - 内容变化 → 版本号 +1，形成新的不可变快照。
        """
        async with self._lock:
            existing = self._latest.get(artifact.artifact_id)
            if existing is not None and existing.content_hash == artifact.content_hash:
                return existing.ref()

            version = (existing.version + 1) if existing else artifact.version
            stored = artifact.model_copy(update={"version": version})
            self._latest[artifact.artifact_id] = stored
            logger.info(
                "ArtifactStore.put: id={} version={} producer={} hash={}",
                stored.artifact_id,
                stored.version,
                stored.producer,
                stored.content_hash[:8],
            )
            return stored.ref()

    async def read_ref(self, ref: ArtifactRef) -> Artifact:
        """按引用读取产物，并校验版本与哈希一致性。

        Raises:
            ArtifactVersionConflict: 底层产物版本/哈希与引用不一致。
            KeyError: 产物不存在。
        """
        async with self._lock:
            artifact = self._latest.get(ref.artifact_id)

        if artifact is None:
            raise KeyError(f"产物不存在: {ref.artifact_id}")

        if artifact.version != ref.version or artifact.content_hash != ref.content_hash:
            raise ArtifactVersionConflictError(
                f"产物 {ref.artifact_id} 已变更：引用指向 v{ref.version}"
                f"({ref.content_hash[:8]}…)，底层当前为 v{artifact.version}"
                f"({artifact.content_hash[:8]}…)，Reviewer 需重新获取引用。"
            )
        return artifact

    async def read_pointer(self, ref: ArtifactRef, pointer: str | None = None) -> Any:
        """按引用（及可选局部指针）读取内容，读取前校验版本一致性。"""
        artifact = await self.read_ref(ref)
        return resolve_pointer(artifact.content, pointer)


class EventLog:
    """只追加的审计事件日志（按 run_id 分组）。"""

    def __init__(self) -> None:
        self._events: dict[str, list[AuditEvent]] = {}
        self._lock = asyncio.Lock()

    async def record(self, run_id: str, event: AuditEvent) -> None:
        """追加一条审计事件，并打印一行可观测日志。"""
        async with self._lock:
            self._events.setdefault(run_id, []).append(event)

        version = event.artifact_ref.version if event.artifact_ref else "-"
        logger.info(
            "Audit[{}]: {} | actor={} action='{}' based_on_v{} | {}",
            run_id,
            event.timestamp,
            event.actor,
            event.action,
            version,
            event.detail,
        )

    def snapshot(self, run_id: str) -> list[AuditEvent]:
        """返回某次运行的审计事件快照（拷贝，避免外部误改）。"""
        return list(self._events.get(run_id, []))

    def clear(self, run_id: str) -> None:
        """清理某次运行的审计事件。"""
        self._events.pop(run_id, None)


# ──────────────────────────────────────────────────────────────
# 进程内单例（与 get_settings / get_graph 保持同一单例风格）
# ──────────────────────────────────────────────────────────────

_artifact_store: ArtifactStore | None = None
_event_log: EventLog | None = None


def get_artifact_store() -> ArtifactStore:
    """获取进程内产物存储单例。"""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store


def get_event_log() -> EventLog:
    """获取进程内审计事件日志单例。"""
    global _event_log
    if _event_log is None:
        _event_log = EventLog()
    return _event_log
