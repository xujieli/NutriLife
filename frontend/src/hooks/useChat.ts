import { useCallback, useEffect, useRef, useState } from "react";

import { API_BASE } from "@/lib/api";
import type {
  HistoryMessage,
  Intent,
  Message,
  SSEEvent,
  Session,
  SessionItem,
  SessionListResponse,
  User,
  WorkflowStep,
} from "@/types";

/** 历史会话分页大小（每页 20 条）。 */
const PAGE_SIZE = 20;

/** Workflow 子图的节点顺序（与后端 workflow.py 对齐）。 */
const WORKFLOW_STEPS: { id: string; label: string }[] = [
  { id: "extract_food", label: "提取食物" },
  { id: "calculate_calories", label: "计算热量" },
  { id: "check_threshold", label: "判断是否超标" },
  { id: "generate_advice", label: "生成建议" },
];

const WORKFLOW_NODES = new Set(WORKFLOW_STEPS.map((s) => s.id));

function genId(prefix: string): string {
  const rand =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10);
  return `${prefix}-${Date.now()}-${rand}`;
}

/** 后端历史消息 → 前端聊天消息。 */
function toMessage(h: HistoryMessage): Message {
  return {
    id: h.id,
    role: h.role,
    content: h.content,
    createdAt: new Date(h.created_at).getTime(),
  };
}

/** 后端会话项 → 前端 Session。 */
function toSession(item: SessionItem): Session {
  return {
    id: item.session_id,
    title: item.title,
    updatedAt: new Date(item.updated_at).getTime(),
  };
}

/** 根据节点名推断意图（用于在没有 intent 事件时尽早显示徽章）。 */
function intentFromNode(node: string): Intent | undefined {
  if (node === "rag") return "RAG_QUERY";
  if (node === "react") return "REACT_TASK";
  if (node === "general_chat") return "GENERAL_CHAT";
  if (node === "workflow" || WORKFLOW_NODES.has(node)) return "WORKFLOW_TASK";
  return undefined;
}

/** 根据当前节点生成步骤进度（之前为 done，当前为 active，之后为 pending）。 */
function stepsForNode(node: string): WorkflowStep[] {
  const index = WORKFLOW_STEPS.findIndex((s) => s.id === node);
  return WORKFLOW_STEPS.map((s, i) => ({
    id: s.id,
    label: s.label,
    status: i < index ? "done" : i === index ? "active" : "pending",
  }));
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState(false);

  const [user, setUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [hasMore, setHasMore] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const pageRef = useRef(1);
  const userRef = useRef<User | null>(null);

  /** 分页拉取当前用户的历史会话（按时间倒序，每页 20 条）。 */
  const loadSessions = useCallback(
    async (reset = false) => {
      if (!userRef.current) return;
      if (!reset && (loadingSessions || !hasMore)) return;

      const page = reset ? 1 : pageRef.current;
      setLoadingSessions(true);
      try {
        const res = await fetch(
          `${API_BASE}/sessions?page=${page}&page_size=${PAGE_SIZE}`,
          { credentials: "include" }
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as SessionListResponse;
        const items = data.items.map(toSession);
        setSessions((prev) => (reset ? items : [...prev, ...items]));
        setHasMore(data.has_more);
        pageRef.current = page + 1;
      } catch {
        // 静默失败，不打断主流程
      } finally {
        setLoadingSessions(false);
      }
    },
    [hasMore, loadingSessions]
  );

  // 挂载时校验登录态，已登录则自动拉取会话列表
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/me`, {
          credentials: "include",
        });
        if (res.ok) {
          const u = (await res.json()) as User;
          if (!cancelled) {
            setUser(u);
            userRef.current = u;
            void loadSessions(true);
          }
        }
      } catch {
        // 忽略网络异常，按未登录处理
      } finally {
        if (!cancelled) setAuthLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const login = useCallback(
    async (phone: string, code: string) => {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ phone, code }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || "登录失败，请重试");
      }
      setUser(data.user as User);
      userRef.current = data.user as User;
      pageRef.current = 1;
      setSessions([]);
      setHasMore(false);
      void loadSessions(true);
    },
    [loadSessions]
  );

  const logout = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
    } catch {
      // 忽略网络异常
    }
    abortRef.current?.abort();
    setUser(null);
    userRef.current = null;
    setSessions([]);
    setHasMore(false);
    pageRef.current = 1;
    setMessages([]);
    setCurrentSessionId("");
    setIsStreaming(false);
  }, []);

  const newSession = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setCurrentSessionId("");
    setIsStreaming(false);
  }, []);

  const selectSession = useCallback(
    async (sessionId: string) => {
      abortRef.current?.abort();
      setIsStreaming(false);
      setCurrentSessionId(sessionId);
      setMessages([]);

      // 已登录：从服务端拉取该会话的完整聊天记录
      if (user) {
        try {
          const res = await fetch(
            `${API_BASE}/sessions/${sessionId}/messages`,
            { credentials: "include" }
          );
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const history = (await res.json()) as HistoryMessage[];
          setMessages(history.map(toMessage));
        } catch {
          setMessages([]);
        }
      }
    },
    [user]
  );

  const loadMoreSessions = useCallback(() => {
    void loadSessions(false);
  }, [loadSessions]);

  const deleteSessions = useCallback(
    async (sessionIds: string[]) => {
      if (!sessionIds.length) return;
      const res = await fetch(`${API_BASE}/sessions`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ session_ids: sessionIds }),
      });
      if (!res.ok) {
        throw new Error("删除失败，请稍后重试");
      }
      // 同步更新本地列表，移除已删除条目
      setSessions((prev) => prev.filter((s) => !sessionIds.includes(s.id)));
      // 若删除的是当前会话，清空对话区
      if (sessionIds.includes(currentSessionId)) {
        abortRef.current?.abort();
        setMessages([]);
        setCurrentSessionId("");
        setIsStreaming(false);
      }
    },
    [currentSessionId]
  );

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      if (isStreaming) return;

      // 当前会话 ID：首次发送时生成，后续复用
      const sessionId = currentSessionId || genId("session");

      const userMessage: Message = {
        id: genId("user"),
        role: "user",
        content: trimmed,
        createdAt: Date.now(),
      };
      const assistantMessage: Message = {
        id: genId("assistant"),
        role: "assistant",
        content: "",
        isStreaming: true,
        createdAt: Date.now(),
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setCurrentSessionId(sessionId);
      setSessions((prev) => {
        const existing = prev.find((s) => s.id === sessionId);
        const title = trimmed.slice(0, 30);
        if (existing) {
          return prev.map((s) =>
            s.id === sessionId ? { ...s, title, updatedAt: Date.now() } : s
          );
        }
        return [{ id: sessionId, title, updatedAt: Date.now() }, ...prev];
      });
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const patchAssistant = (updater: (m: Message) => Message) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantMessage.id ? updater(m) : m))
        );
      };

      const handleEvent = (event: SSEEvent) => {
        switch (event.type) {
          case "text": {
            const inferred = intentFromNode(event.node);
            const isWorkflow = WORKFLOW_NODES.has(event.node);
            patchAssistant((m) => ({
              ...m,
              content: m.content + event.content,
              ...(inferred && !m.intent ? { intent: inferred } : {}),
              ...(isWorkflow ? { steps: stepsForNode(event.node) } : {}),
            }));
            break;
          }
          case "intent": {
            patchAssistant((m) => ({
              ...m,
              intent: event.content as Intent,
            }));
            break;
          }
          case "tool_call":
          case "tool_result": {
            if (event.node && WORKFLOW_NODES.has(event.node)) {
              patchAssistant((m) => ({
                ...m,
                intent: m.intent ?? "WORKFLOW_TASK",
                steps: stepsForNode(event.node),
              }));
            }
            break;
          }
          case "done": {
            patchAssistant((m) => ({
              ...m,
              content: event.content || m.content,
              intent: (event.intent as Intent) || m.intent,
              sources: event.sources,
              isStreaming: false,
              isError: Boolean(event.error),
              ...(m.steps?.length
                ? { steps: m.steps.map((s) => ({ ...s, status: "done" as const })) }
                : {}),
            }));
            break;
          }
        }
      };

      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ user_input: trimmed, session_id: sessionId }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`请求失败（HTTP ${res.status}）`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine.startsWith("data:")) continue;
            const json = trimmedLine.slice(5).trim();
            if (!json) continue;
            try {
              handleEvent(JSON.parse(json) as SSEEvent);
            } catch {
              // 忽略无法解析的帧
            }
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          patchAssistant((m) => ({
            ...m,
            content: m.content || "抱歉，请求失败，请稍后再试。",
            isStreaming: false,
            isError: true,
          }));
        } else {
          patchAssistant((m) => ({ ...m, isStreaming: false }));
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
        // 已登录时刷新会话列表，确保新会话与最近更新时间正确
        if (user) {
          void loadSessions(true);
        }
      }
    },
    [currentSessionId, isStreaming, user, loadSessions]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return {
    messages,
    sessions,
    currentSessionId,
    isStreaming,
    user,
    authLoading,
    hasMore,
    loadingSessions,
    sendMessage,
    stop,
    newSession,
    selectSession,
    loadMoreSessions,
    deleteSessions,
    login,
    logout,
  };
}
