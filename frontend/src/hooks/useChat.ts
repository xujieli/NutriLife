import { useCallback, useRef, useState } from "react";

import type {
  Intent,
  Message,
  SSEEvent,
  Session,
  WorkflowStep,
} from "@/types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

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
  const abortRef = useRef<AbortController | null>(null);

  const newSession = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setCurrentSessionId("");
    setIsStreaming(false);
  }, []);

  const selectSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId);
  }, []);

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
      }
    },
    [currentSessionId, isStreaming]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return {
    messages,
    sessions,
    currentSessionId,
    isStreaming,
    sendMessage,
    stop,
    newSession,
    selectSession,
  };
}
