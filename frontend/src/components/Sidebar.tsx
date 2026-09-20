import { useState } from "react";
import {
  HeartPulse,
  Loader2,
  LogIn,
  LogOut,
  MessageSquarePlus,
  MessageSquareText,
  Trash2,
} from "lucide-react";

import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Session, User } from "@/types";

interface SidebarProps {
  sessions: Session[];
  currentSessionId: string;
  user: User | null;
  hasMore: boolean;
  loadingSessions: boolean;
  onNew: () => void;
  onSelect: (sessionId: string) => void;
  onLoadMore: () => void;
  onLoginClick: () => void;
  onLogout: () => void;
  onDelete: (sessionIds: string[]) => Promise<void>;
}

/** 手机号脱敏：138****5678。 */
function maskPhone(phone: string): string {
  if (phone.length !== 11) return phone;
  return `${phone.slice(0, 3)}****${phone.slice(7)}`;
}

interface DeleteTarget {
  ids: string[];
  title: string;
  description: string;
}

export function Sidebar({
  sessions,
  currentSessionId,
  user,
  hasMore,
  loadingSessions,
  onNew,
  onSelect,
  onLoadMore,
  onLoginClick,
  onLogout,
  onDelete,
}: SidebarProps) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [confirmTarget, setConfirmTarget] = useState<DeleteTarget | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 24) {
      onLoadMore();
    }
  };

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const toggleSelectAll = () => {
    setSelectedIds((prev) =>
      prev.length === sessions.length ? [] : sessions.map((s) => s.id)
    );
  };

  const clearSelection = () => setSelectedIds([]);

  const requestDelete = (ids: string[]) => {
    if (ids.length === 0) return;
    setDeleteError(null);
    setConfirmTarget({
      ids,
      title: ids.length > 1 ? "批量删除会话" : "删除会话",
      description:
        ids.length > 1
          ? `确定删除选中的 ${ids.length} 个会话吗？删除后聊天记录将无法恢复。`
          : "删除后该会话及其全部聊天记录将无法恢复。",
    });
  };

  const confirmDelete = async () => {
    if (!confirmTarget) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await onDelete(confirmTarget.ids);
      setSelectedIds([]);
      setConfirmTarget(null);
    } catch (err) {
      setDeleteError((err as Error).message || "删除失败，请稍后重试");
    } finally {
      setDeleting(false);
    }
  };

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

      {/* 批量操作工具栏 */}
      {user && selectedIds.length > 0 && (
        <div className="flex items-center justify-between gap-2 px-3 pb-2">
          <span className="text-xs text-muted-foreground">
            已选 {selectedIds.length} 项
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="destructive"
              size="sm"
              className="h-8 gap-1 px-2.5 text-xs"
              onClick={() => requestDelete(selectedIds)}
            >
              <Trash2 className="h-3.5 w-3.5" />
              批量删除
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-2.5 text-xs"
              onClick={clearSelection}
            >
              取消
            </Button>
          </div>
        </div>
      )}

      {/* 会话列表（滚动触底加载更多） */}
      <div
        className="scrollbar-thin flex-1 overflow-y-auto px-3"
        onScroll={handleScroll}
      >
        <div className="space-y-1 pb-2">
          <div className="flex items-center justify-between px-2 pb-1 pt-2">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground/70">
              {user ? "历史会话" : "本次会话"}
            </p>
            {user && sessions.length > 0 && (
              <label className="flex cursor-pointer items-center gap-1 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5 rounded accent-primary"
                  checked={
                    selectedIds.length === sessions.length &&
                    sessions.length > 0
                  }
                  onChange={toggleSelectAll}
                />
                全选
              </label>
            )}
          </div>

          {sessions.length === 0 ? (
            <p className="px-2 py-6 text-center text-xs text-muted-foreground/60">
              {user ? "暂无会话，开始你的第一次对话吧" : "登录后同步历史会话记录"}
            </p>
          ) : (
            sessions.map((session) => {
              const selected = selectedIds.includes(session.id);
              return (
                <div
                  key={session.id}
                  className={cn(
                    "group flex items-center gap-1.5 rounded-lg px-2 py-1.5 transition-colors",
                    session.id === currentSessionId
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                  )}
                >
                  {user && (
                    <input
                      type="checkbox"
                      className="h-3.5 w-3.5 shrink-0 cursor-pointer rounded accent-primary"
                      checked={selected}
                      onChange={() => toggleSelect(session.id)}
                      aria-label={`选择会话：${session.title}`}
                    />
                  )}
                  <button
                    className="flex min-w-0 flex-1 items-center gap-2 text-left text-sm"
                    onClick={() => onSelect(session.id)}
                  >
                    <MessageSquareText className="h-4 w-4 shrink-0 opacity-60" />
                    <span className="truncate">{session.title}</span>
                  </button>
                  {user && (
                    <button
                      className={cn(
                        "shrink-0 rounded-md p-1.5 text-muted-foreground/70 transition-colors hover:bg-destructive/10 hover:text-destructive",
                        "opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
                      )}
                      onClick={() => requestDelete([session.id])}
                      aria-label={`删除会话：${session.title}`}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  )}
                </div>
              );
            })
          )}

          {loadingSessions && (
            <div className="flex items-center justify-center gap-1.5 py-2 text-xs text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              加载中…
            </div>
          )}
          {!hasMore && user && sessions.length > 0 && (
            <p className="px-2 py-2 text-center text-xs text-muted-foreground/50">
              已加载全部会话
            </p>
          )}
        </div>
      </div>

      {/* 底部：登录态 / 提示 */}
      <div className="border-t border-border/60 px-3 py-3">
        {user ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">
                  {maskPhone(user.phone)}
                </p>
                <p className="text-[11px] text-muted-foreground">已登录</p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="shrink-0 gap-1.5 text-xs"
                onClick={onLogout}
              >
                <LogOut className="h-3.5 w-3.5" />
                退出
              </Button>
            </div>
          </div>
        ) : (
          <Button
            variant="outline"
            className="w-full justify-start gap-2"
            onClick={onLoginClick}
          >
            <LogIn className="h-4 w-4" />
            登录
          </Button>
        )}
        <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground/70">
          由本地 LM Studio 模型驱动
          <br />
          回答仅供参考，请遵医嘱
        </p>
      </div>

      {/* 删除二次确认弹窗 */}
      <ConfirmDialog
        open={confirmTarget !== null}
        title={confirmTarget?.title ?? ""}
        description={confirmTarget?.description}
        loading={deleting}
        error={deleteError}
        onConfirm={confirmDelete}
        onCancel={() => {
          if (!deleting) {
            setConfirmTarget(null);
            setDeleteError(null);
          }
        }}
      />
    </div>
  );
}
