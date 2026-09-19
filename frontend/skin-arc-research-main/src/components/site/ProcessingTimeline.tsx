import { motion } from "framer-motion";
import { Check, Loader2, Circle } from "lucide-react";
import { ANALYSIS_STAGES, type Status } from "@/lib/use-analysis";
import { cn } from "@/lib/utils";

export function ProcessingTimeline({ activeStage, status }: { activeStage: number; status: Status }) {
  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex items-center justify-between">
        <span className="label-mono">Processing sequence</span>
        <span className="font-mono text-xs text-primary">
          {Math.max(0, Math.min(activeStage, ANALYSIS_STAGES.length))}/{ANALYSIS_STAGES.length}
        </span>
      </div>

      <ol className="mt-5 space-y-1">
        {ANALYSIS_STAGES.map((stage, i) => {
          const done = activeStage > i;
          const current = activeStage === i && status === "running";
          const idle = status === "idle";
          return (
            <li key={stage.id} className="relative flex items-center gap-3 py-2.5">
              <span
                className={cn(
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border transition-colors duration-300",
                  done
                    ? "border-primary bg-primary/15 text-primary"
                    : current
                      ? "border-primary text-primary"
                      : "border-border text-muted-foreground",
                )}
              >
                {done ? (
                  <Check className="h-3 w-3" />
                ) : current ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Circle className="h-2 w-2 fill-current" />
                )}
              </span>
              <span
                className={cn(
                  "text-sm transition-colors duration-300",
                  done || current ? "text-foreground" : "text-muted-foreground",
                  idle && "opacity-60",
                )}
              >
                {stage.label}
              </span>
              {current ? (
                <motion.span
                  layoutId="stage-bar"
                  className="absolute inset-y-0 -left-3 w-px bg-primary"
                />
              ) : null}
            </li>
          );
        })}
      </ol>

      <div className="mt-4 h-1 w-full overflow-hidden rounded-full bg-secondary/60">
        <motion.div
          className="h-full rounded-full"
          style={{ background: "var(--gradient-line)" }}
          animate={{
            width: `${(Math.max(0, activeStage) / ANALYSIS_STAGES.length) * 100}%`,
          }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}
