import { motion } from "framer-motion";
import { lazy, Suspense } from "react";
import { ArrowRight, Boxes, ShieldAlert } from "lucide-react";
import { ClientOnly } from "./ClientOnly";
import { ActionButton, Chip } from "./primitives";

const LesionScene = lazy(() => import("./LesionScene"));

const scrollTo = (id: string) => {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
};

export function Hero() {
  return (
    <section className="relative flex min-h-screen items-center overflow-hidden">
      <div className="absolute inset-0" style={{ background: "var(--gradient-hero)" }} aria-hidden />
      <div className="grid-backdrop absolute inset-0 opacity-40 [mask-image:radial-gradient(70%_60%_at_50%_40%,black,transparent)]" aria-hidden />

      <div className="relative mx-auto grid w-full max-w-7xl grid-cols-1 items-center gap-12 px-5 py-28 sm:px-8 lg:grid-cols-[1.05fr_0.95fr]">
        <motion.div
          initial={{ opacity: 0, y: 26 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="flex flex-wrap items-center gap-2">
            <Chip>Agentic RAG</Chip>
            <Chip tone="muted">Research prototype</Chip>
            <Chip tone="signal">
              <ShieldAlert className="h-3 w-3" /> Not a diagnosis
            </Chip>
          </div>

          <h1 className="mt-7 text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
            <span className="text-gradient">Explainable AI</span>
            <br />
            for Melanoma Analysis
          </h1>

          <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground">
            An evidence-grounded Agentic RAG framework connecting dermoscopic analysis, clinical
            features and medical literature.
          </p>

          <div className="mt-10 flex flex-wrap gap-3">
            <ActionButton onClick={() => scrollTo("workspace")}>
              Analyze Lesion
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </ActionButton>
            <ActionButton variant="ghost" onClick={() => scrollTo("architecture")}>
              <Boxes className="h-4 w-4" />
              Explore Architecture
            </ActionButton>
          </div>

          <dl className="mt-14 grid max-w-lg grid-cols-3 gap-6 border-t border-border pt-7">
            {[
              ["Retrieval", "Qdrant vectors"],
              ["Synthesis", "Gemini, grounded"],
              ["Output", "Cited report"],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="label-mono">{k}</dt>
                <dd className="mt-1 text-sm text-foreground">{v}</dd>
              </div>
            ))}
          </dl>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
          className="relative h-[340px] w-full sm:h-[460px] lg:h-[560px]"
        >
          <div className="absolute inset-0 rounded-full bg-primary/10 blur-3xl" aria-hidden />
          <ClientOnly
            fallback={
              <div className="flex h-full items-center justify-center">
                <div className="h-40 w-40 animate-pulse rounded-full border border-primary/30 bg-primary/5" />
              </div>
            }
          >
            <Suspense fallback={null}>
              <LesionScene />
            </Suspense>
          </ClientOnly>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
          className="label-mono"
        >
          Scroll
        </motion.div>
      </motion.div>
    </section>
  );
}
