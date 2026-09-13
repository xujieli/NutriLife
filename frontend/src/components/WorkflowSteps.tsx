import { AnimatePresence, motion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";
import type { WorkflowStep } from "@/types";

interface WorkflowStepsProps {
  steps: WorkflowStep[];
  isStreaming?: boolean;
}

/**
 * Workflow 执行进度可视化：展示「提取食物 → 计算热量 → 判断是否超标 → 生成建议」。
 * active 步骤显示脉冲加载动画，done 步骤显示对勾，pending 步骤保持灰显。
 */
export function WorkflowSteps({ steps, isStreaming }: WorkflowStepsProps) {
  if (!steps.length) return null;

  const activeIndex = steps.findIndex((s) => s.status === "active");
  const finished = !isStreaming && steps.every((s) => s.status === "done");

  return (
    <div className="my-3 rounded-xl border border-border/70 bg-accent/30 px-4 py-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium text-accent-foreground/80">
        {finished ? (
          <Check className="h-3.5 w-3.5" />
        ) : (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        )}
        <span>{finished ? "任务执行完成" : "正在执行任务"}</span>
      </div>

      <ol className="space-y-1.5">
        <AnimatePresence initial={false}>
          {steps.map((step, i) => {
            const isActive = step.status === "active";
            const isDone = step.status === "done";
            const isReached = i <= activeIndex || isDone;

            return (
              <motion.li
                key={step.id}
                layout
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: isReached ? 1 : 0.4, x: 0 }}
                className="flex items-center gap-2.5"
              >
                <span
                  className={cn(
                    "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border transition-colors",
                    isDone &&
                      "border-primary bg-primary text-primary-foreground",
                    isActive && "border-primary text-primary",
                    step.status === "pending" && "border-border bg-background"
                  )}
                >
                  {isDone ? (
                    <Check className="h-2.5 w-2.5" strokeWidth={3} />
                  ) : isActive ? (
                    <Loader2 className="h-2.5 w-2.5 animate-spin" />
                  ) : (
                    <span className="h-1 w-1 rounded-full bg-muted-foreground/50" />
                  )}
                </span>
                <span
                  className={cn(
                    "text-xs",
                    isActive
                      ? "font-medium text-accent-foreground"
                      : "text-muted-foreground"
                  )}
                >
                  {i + 1}. {step.label}
                </span>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </ol>
    </div>
  );
}
