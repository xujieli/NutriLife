import { useState } from "react";

import { ChatInterface } from "@/components/ChatInterface";
import { LoginDialog } from "@/components/LoginDialog";
import { Sidebar } from "@/components/Sidebar";
import { useChat } from "@/hooks/useChat";
import { cn } from "@/lib/utils";

export default function App() {
  const chat = useChat();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loginOpen, setLoginOpen] = useState(false);
  const [loginLoading, setLoginLoading] = useState(false);

  const lastAssistant = [...chat.messages]
    .reverse()
    .find((m) => m.role === "assistant");
  const currentIntent = lastAssistant?.intent;

  const handleLogin = async (phone: string, code: string) => {
    setLoginLoading(true);
    try {
      await chat.login(phone, code);
      setLoginOpen(false);
    } catch (err) {
      throw err;
    } finally {
      setLoginLoading(false);
    }
  };

  return (
    <div className="flex h-full overflow-hidden bg-background">
      {/* 移动端遮罩 */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/30 backdrop-blur-[1px] lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden
        />
      )}

      {/* 左侧边栏 */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 w-72 shrink-0 transform transition-transform duration-200 ease-out lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <Sidebar
          sessions={chat.sessions}
          currentSessionId={chat.currentSessionId}
          user={chat.user}
          hasMore={chat.hasMore}
          loadingSessions={chat.loadingSessions}
          onNew={() => {
            chat.newSession();
            setSidebarOpen(false);
          }}
          onSelect={(id) => {
            chat.selectSession(id);
            setSidebarOpen(false);
          }}
          onLoadMore={chat.loadMoreSessions}
          onLoginClick={() => setLoginOpen(true)}
          onLogout={() => {
            void chat.logout();
          }}
          onDelete={chat.deleteSessions}
        />
      </aside>

      {/* 右侧主区域 */}
      <main className="min-w-0 flex-1">
        <ChatInterface
          messages={chat.messages}
          isStreaming={chat.isStreaming}
          currentIntent={currentIntent}
          onSend={chat.sendMessage}
          onStop={chat.stop}
          onToggleSidebar={() => setSidebarOpen((open) => !open)}
        />
      </main>

      {/* 登录弹窗 */}
      <LoginDialog
        open={loginOpen}
        loading={loginLoading}
        onClose={() => setLoginOpen(false)}
        onLogin={handleLogin}
      />
    </div>
  );
}
