import { motion } from "framer-motion";
import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { PIPELINE_NODES } from "@/lib/pipeline-data";
import { SectionHeading, SectionShell } from "./primitives";
import { cn } from "@/lib/utils";

const GROUP_COLOR: Record<string, string> = {
  vision: "border-teal/40",
  model: "border-primary/45",
  agent: "border-electric/45",
  synthesis: "border-primary/45",
};

export function PipelineSection() {
  const [active, setActive] = useState<string | null>(null);

  return (
    <SectionShell id="pipeline">
      <SectionHeading
        eyebrow="How the system works"
        title="Nine deterministic stages, one traceable path"
        description="Every claim in the final report can be walked back through the pipeline to the pixel measurements and the retrieved passage that supports it."
      />

      <div className="mt-14 grid gap-4 lg:grid-cols-3">
        {PIPELINE_NODES.map((node, i) => (
          <motion.button
            key={node.id}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.45, delay: i * 0.05 }}
            onMouseEnter={() => setActive(node.id)}
            onMouseLeave={() => setActive(null)}
            onFocus={() => setActive(node.id)}
            onBlur={() => setActive(null)}
            className={cn(
              "glass group relative overflow-hidden rounded-2xl border p-5 text-left transition-all duration-300 hover:-translate-y-1",
              GROUP_COLOR[node.group],
              active === node.id && "panel-glow",
            )}
          >
            <div className="flex items-center justify-between">
              <span className="label-mono">Stage {String(i + 1).padStart(2, "0")}</span>
              <ChevronRight className="h-4 w-4 text-primary/60 transition-transform group-hover:translate-x-1" />
            </div>
            <h3 className="mt-3 text-base font-medium tracking-tight">{node.label}</h3>

            <motion.p
              initial={false}
              animate={{
                height: active === node.id ? "auto" : 0,
                opacity: active === node.id ? 1 : 0,
              }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden text-sm leading-relaxed text-muted-foreground"
            >
              <span className="mt-3 block">{node.detail}</span>
            </motion.p>

            <span
              className="absolute inset-x-0 bottom-0 h-px opacity-0 transition-opacity duration-300 group-hover:opacity-100"
              style={{ background: "var(--gradient-line)" }}
            />
          </motion.button>
        ))}
      </div>
    </SectionShell>
  );
}
