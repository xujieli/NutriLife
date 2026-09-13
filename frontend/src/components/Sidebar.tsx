import { MessageSquarePlus, MessageSquareText, HeartPulse } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { Session } from "@/types";

interface SidebarProps {
  sessions: Session[];
  currentSessionId: string;
  onNew: () => void;
  onSelect: (sessionId: string) => void;
}

export function Sidebar({
  sessions,
  currentSessionId,
  onNew,
  onSelect,
}: SidebarProps) {
  return (
    <div className="flex h-full flex-col border-r border-border/60 bg-secondary/40">
      {/* 品牌区 */}
      <div className="flex items-center gap-2.5 px-4 pb-3 pt-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <HeartPulse className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-semibold leading-tight">NutriLife</p>
          <p className="text-xs text-muted-foreground">营养健康助手</p>
        </div>
      </div>

      {/* 新建会话 */}
      <div className="px-3 pb-2">
        <Button
          variant="outline"
          className="w-full justify-start gap-2"
          onClick={onNew}
        >
          <MessageSquarePlus className="h-4 w-4" />
          新建会话
        </Button>
      </div>

      {/* 会话列表 */}
      <ScrollArea className="flex-1 px-3">
        <div className="space-y-1 pb-2">
          <p className="px-2 pb-1 pt-2 text-xs font-medium uppercase tracking-wide text-muted-foreground/70">
            历史会话
          </p>
          {sessions.length === 0 ? (
            <p className="px-2 py-6 text-center text-xs text-muted-foreground/60">
              暂无会话，开始你的第一次对话吧
            </p>
          ) : (
            sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => onSelect(session.id)}
                className={cn(
                  "group flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition-colors",
                  session.id === currentSessionId
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                )}
              >
                <MessageSquareText className="h-4 w-4 shrink-0 opacity-60" />
                <span className="truncate">{session.title}</span>
              </button>
            ))
          )}
        </div>
      </ScrollArea>

      {/* 底部提示 */}
      <div className="border-t border-border/60 px-4 py-3">
        <p className="text-[11px] leading-relaxed text-muted-foreground/70">
          由本地 LM Studio 模型驱动
          <br />
          回答仅供参考，请遵医嘱
        </p>
      </div>
    </div>
  );
}
