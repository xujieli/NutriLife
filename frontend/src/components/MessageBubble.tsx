import { useState } from "react";
import { motion } from "framer-motion";
import { Bot, Check, ChevronDown, Copy, FileText, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import { WorkflowSteps } from "@/components/WorkflowSteps";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import type { Message, SourceReference } from "@/types";

/** 从文本中解析「[1] 来源：摘要」形式的引用块（结构化 sources 缺失时的兜底）。 */
function extractSourcesFromText(text: string): SourceReference[] {
  const sources: SourceReference[] = [];
  for (const line of text.split("\n")) {
    const match = line.match(/^\s*\[(\d+)\]\s+(.+?)[：:]\s*(.+)$/);
    if (match) {
      sources.push({
        source: match[2].trim(),
        snippet: match[3].trim(),
        score: 1,
      });
    }
  }
  return sources;
}

function Markdown({ content }: { content: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1.5">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60"
          animate={{ opacity: [0.3, 1, 0.3], y: [0, -2, 0] }}
          transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
        />
      ))}
    </div>
  );
}

function SourcesCard({ sources }: { sources: SourceReference[] }) {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="w-full">
      <CollapsibleTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
        >
          <FileText className="h-3.5 w-3.5" />
          参考来源（{sources.length}）
          <ChevronDown
            className={cn(
              "h-3.5 w-3.5 transition-transform",
              open && "rotate-180"
            )}
          />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-1 space-y-1.5">
        {sources.map((source, i) => (
          <div
            key={i}
            className="rounded-lg border border-border/70 bg-card/70 px-3 py-2 text-xs"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="font-medium text-primary">
                [{i + 1}] {source.source}
              </span>
              {typeof source.score === "number" && (
                <span className="shrink-0 text-muted-foreground">
                  相关度 {Math.round(source.score * 100)}%
                </span>
              )}
            </div>
            {source.snippet && (
              <p className="mt-1 line-clamp-2 text-muted-foreground">
                {source.snippet}
              </p>
            )}
          </div>
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const sources = message.sources ?? extractSourcesFromText(message.content);
  const showTyping =
    message.isStreaming && !message.content && !message.steps?.length;

  const copyContent = async () => {
    if (!message.content) return;
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪贴板不可用时静默失败
    }
  };

  const timeLabel = message.createdAt
    ? new Date(message.createdAt).toLocaleString("zh-CN", {
        month: "numeric",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={cn("flex gap-3", isUser && "flex-row-reverse")}
    >
      <div
        className={cn(
          "mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser
            ? "bg-foreground text-background"
            : "bg-primary text-primary-foreground"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div
        className={cn(
          "flex max-w-[80%] flex-col gap-1.5",
          isUser ? "items-end" : "items-start"
        )}
      >
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "rounded-tr-md bg-foreground text-background"
              : "rounded-tl-md border border-border/70 bg-card shadow-sm"
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : showTyping ? (
            <TypingDots />
          ) : (
            <>
              {message.steps && message.steps.length > 0 && (
                <WorkflowSteps
                  steps={message.steps}
                  isStreaming={message.isStreaming}
                />
              )}
              {message.content && <Markdown content={message.content} />}
              {message.isStreaming && message.content && (
                <span className="ml-0.5 inline-block h-4 w-[3px] animate-pulse rounded-full bg-primary align-text-bottom" />
              )}
            </>
          )}
        </div>

        {sources.length > 0 && !isUser && <SourcesCard sources={sources} />}

        {(timeLabel || message.content) && !message.isStreaming && (
          <div className="flex items-center gap-2 px-1 text-[11px] text-muted-foreground/70">
            {timeLabel && <span>{timeLabel}</span>}
            {message.content && (
              <button
                onClick={copyContent}
                className="inline-flex items-center gap-1 rounded transition-colors hover:text-foreground"
                aria-label="复制消息内容"
              >
                {copied ? (
                  <Check className="h-3 w-3" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
                {copied ? "已复制" : "复制"}
              </button>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
