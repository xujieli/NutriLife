import { useEffect, useRef } from "react";
import { Menu, Sparkles } from "lucide-react";

import { ChatInput } from "@/components/ChatInput";
import { MessageBubble } from "@/components/MessageBubble";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Intent, Message } from "@/types";

const INTENT_META: Record<Intent, { icon: string; label: string }> = {
  RAG_QUERY: { icon: "📚", label: "知识问答" },
  WORKFLOW_TASK: { icon: "⚙️", label: "任务执行" },
  REACT_TASK: { icon: "🧠", label: "深度分析" },
  GENERAL_CHAT: { icon: "💬", label: "日常对话" },
};

const EXAMPLE_PROMPTS = [
  { text: "痛风能吃豆制品吗", hint: "知识问答" },
  { text: "记录午餐：汉堡、薯条，计算热量并给建议", hint: "任务执行" },
  { text: "维生素D缺乏有什么症状", hint: "知识问答" },
];

interface ChatInterfaceProps {
  messages: Message[];
  isStreaming: boolean;
  currentIntent: Intent | undefined;
  onSend: (text: string) => void;
  onStop: () => void;
  onToggleSidebar: () => void;
}

function EmptyState({ onSend }: { onSend: (text: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
        <Sparkles className="h-7 w-7" />
      </div>
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">你好，我是 NutriLife</h2>
        <p className="text-sm text-muted-foreground">
          你的私人营养健康助手，可以答疑、记录饮食、计算热量。
        </p>
      </div>
      <div className="grid w-full max-w-md gap-2">
        {EXAMPLE_PROMPTS.map((prompt) => (
          <button
            key={prompt.text}
            onClick={() => onSend(prompt.text)}
            className="group flex items-center justify-between rounded-xl border border-border/70 bg-card px-4 py-3 text-left text-sm transition-colors hover:border-primary/40 hover:bg-accent/40"
          >
            <span className="text-foreground/90">{prompt.text}</span>
            <span className="shrink-0 text-xs text-muted-foreground">
              {prompt.hint}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function ChatInterface({
  messages,
  isStreaming,
  currentIntent,
  onSend,
  onStop,
  onToggleSidebar,
}: ChatInterfaceProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const intentMeta = currentIntent ? INTENT_META[currentIntent] : null;
  const title =
    messages.find((m) => m.role === "user")?.content.slice(0, 24) || "新对话";

  // 流式输出时始终钉在底部；非流式时仅在贴近底部时跟随（用户上翻阅读不打扰）
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (isStreaming) {
      el.scrollTop = el.scrollHeight;
    } else {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
      if (nearBottom) {
        el.scrollTop = el.scrollHeight;
      }
    }
  }, [messages, isStreaming]);

  return (
    <div className="flex h-full flex-col">
      {/* 顶部：标题 + 意图徽章 */}
      <header className="flex shrink-0 items-center gap-2 border-b border-border/60 bg-background/80 px-4 py-3 backdrop-blur md:px-6">
        <Button
          variant="ghost"
          size="icon"
          className="lg:hidden"
          onClick={onToggleSidebar}
          aria-label="打开侧边栏"
        >
          <Menu className="h-5 w-5" />
        </Button>
        <h1 className="min-w-0 truncate text-sm font-semibold">{title}</h1>
        {intentMeta && (
          <Badge variant="teal" className="shrink-0 gap-1">
            <span aria-hidden>{intentMeta.icon}</span>
            {intentMeta.label}
          </Badge>
        )}
      </header>

      {/* 消息区 */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto scrollbar-thin"
      >
        <div className="mx-auto w-full max-w-3xl px-4 py-6 md:px-6">
          {messages.length === 0 ? (
            <EmptyState onSend={onSend} />
          ) : (
            <div className="space-y-5">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 输入区 */}
      <ChatInput isStreaming={isStreaming} onSend={onSend} onStop={onStop} />
    </div>
  );
}
