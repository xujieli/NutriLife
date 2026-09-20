import { useCallback, useState } from "react";
import { Loader2, Smartphone, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { API_BASE } from "@/lib/api";
import { cn } from "@/lib/utils";

const PHONE_RE = /^1[3-9]\d{9}$/;

interface LoginDialogProps {
  open: boolean;
  loading: boolean;
  onClose: () => void;
  onLogin: (phone: string, code: string) => Promise<void>;
}

export function LoginDialog({
  open,
  loading,
  onClose,
  onLogin,
}: LoginDialogProps) {
  const [phone, setPhone] = useState("");
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [hint, setHint] = useState("");
  const [sendingCode, setSendingCode] = useState(false);

  const phoneValid = PHONE_RE.test(phone);
  const codeValid = code.trim().length > 0;
  const canLogin = phoneValid && codeValid && !loading;

  const sendCode = useCallback(async () => {
    if (!phoneValid || sendingCode) return;
    setSendingCode(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/auth/send-code`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ phone }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.detail || "验证码发送失败");
        return;
      }
      setHint(data.hint || "验证码已发送");
    } catch {
      setError("网络异常，请稍后重试");
    } finally {
      setSendingCode(false);
    }
  }, [phone, phoneValid, sendingCode]);

  const submit = useCallback(async () => {
    if (!canLogin) return;
    setError("");
    try {
      await onLogin(phone.trim(), code.trim());
    } catch (err) {
      setError((err as Error).message || "登录失败，请重试");
    }
  }, [canLogin, phone, code, onLogin]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-[2px]"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      <div
        className="w-full max-w-sm rounded-2xl border border-border bg-card p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 标题栏 */}
        <div className="mb-5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Smartphone className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold">手机号登录</h2>
              <p className="text-xs text-muted-foreground">
                登录后自动同步历史会话记录
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="关闭"
            className="h-8 w-8"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* 手机号 */}
        <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
          手机号
        </label>
        <Input
          value={phone}
          onChange={(e) => {
            setPhone(e.target.value.replace(/\D/g, "").slice(0, 11));
            setError("");
          }}
          placeholder="请输入 11 位手机号"
          inputMode="numeric"
          maxLength={11}
          autoFocus
        />

        {/* 验证码 */}
        <label className="mb-1.5 mt-4 block text-xs font-medium text-muted-foreground">
          验证码
        </label>
        <div className="flex gap-2">
          <Input
            value={code}
            onChange={(e) => {
              setCode(e.target.value.replace(/\D/g, "").slice(0, 6));
              setError("");
            }}
            placeholder="验证码"
            inputMode="numeric"
            maxLength={6}
            onKeyDown={(e) => {
              if (e.key === "Enter" && canLogin) submit();
            }}
          />
          <Button
            variant="outline"
            className="shrink-0"
            disabled={!phoneValid || sendingCode}
            onClick={sendCode}
          >
            {sendingCode ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              "获取验证码"
            )}
          </Button>
        </div>

        {/* 提示 / 错误 */}
        {hint && !error && (
          <p className="mt-2 text-xs text-primary">{hint}</p>
        )}
        {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

        {/* 登录按钮 */}
        <Button
          className="mt-5 w-full"
          disabled={!canLogin}
          onClick={submit}
        >
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          登录
        </Button>

        <button
          className={cn(
            "mt-3 w-full text-center text-xs text-muted-foreground transition-colors hover:text-foreground"
          )}
          onClick={onClose}
        >
          暂不登录，直接开始对话
        </button>
      </div>
    </div>
  );
}
