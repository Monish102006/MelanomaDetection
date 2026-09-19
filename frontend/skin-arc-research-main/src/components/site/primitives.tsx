import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function SectionShell({
  id,
  children,
  className,
}: {
  id?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={cn("relative mx-auto w-full max-w-7xl px-5 py-24 sm:px-8", className)}>
      {children}
    </section>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <motion.header
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="max-w-3xl"
    >
      <div className="flex items-center gap-3">
        <span className="h-px w-8 bg-primary/70" />
        <span className="label-mono text-primary">{eyebrow}</span>
      </div>
      <h2 className="mt-4 text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h2>
      {description ? (
        <p className="mt-4 text-base leading-relaxed text-muted-foreground">{description}</p>
      ) : null}
    </motion.header>
  );
}

export function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("glass rounded-2xl", className)}>{children}</div>;
}

export function Chip({ children, tone = "cyan" }: { children: ReactNode; tone?: "cyan" | "muted" | "signal" }) {
  const tones = {
    cyan: "border-primary/35 bg-primary/10 text-primary",
    muted: "border-border bg-secondary/50 text-muted-foreground",
    signal: "border-destructive/40 bg-destructive/10 text-destructive",
  } as const;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.16em]",
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

export function ActionButton({
  children,
  variant = "primary",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" }) {
  return (
    <button
      {...props}
      className={cn(
        "group inline-flex items-center justify-center gap-2 rounded-full px-6 py-3 text-sm font-medium transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-45",
        variant === "primary"
          ? "bg-primary text-primary-foreground panel-glow hover:brightness-110"
          : "border border-border bg-secondary/40 text-foreground hover:border-primary/50 hover:bg-secondary/70",
        className,
      )}
    >
      {children}
    </button>
  );
}
