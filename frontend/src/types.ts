/** 意图枚举（与后端 Router 的 Intent 对齐）。 */
export type Intent = "RAG_QUERY" | "WORKFLOW_TASK" | "REACT_TASK" | "GENERAL_CHAT";

/** RAG 引用来源。 */
export interface SourceReference {
  source: string;
  snippet: string;
  score: number;
}

/** Workflow 步骤状态。 */
export type StepStatus = "pending" | "active" | "done";

export interface WorkflowStep {
  id: string;
  label: string;
  status: StepStatus;
}

/** 后端 SSE 流式事件协议。 */
export type SSEEvent =
  | {
      type: "text";
      content: string;
      node: string;
      final?: boolean;
    }
  | {
      type: "tool_call";
      content: string;
      node: string;
      input?: Record<string, unknown>;
    }
  | {
      type: "tool_result";
      content: string;
      node: string;
    }
  | {
      type: "intent";
      content: string;
      confidence?: number;
      node: string;
    }
  | {
      type: "done";
      content: string;
      session_id?: string;
      intent?: string;
      sources?: SourceReference[];
      error?: string;
    };

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: Intent;
  sources?: SourceReference[];
  steps?: WorkflowStep[];
  isStreaming?: boolean;
  isError?: boolean;
  createdAt: number;
}

export interface Session {
  id: string;
  title: string;
  updatedAt: number;
}

/** 当前登录用户（后端 /auth/me 与 /auth/login 返回结构）。 */
export interface User {
  id: string;
  phone: string;
  created_at: string;
  last_login_at: string;
}

/** 历史会话列表的分页响应（后端原始结构）。 */
export interface SessionItem {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SessionListResponse {
  items: SessionItem[];
  page: number;
  page_size: number;
  total: number;
  has_more: boolean;
}

/** 后端返回的单条历史消息（/sessions/{id}/messages）。 */
export interface HistoryMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}
