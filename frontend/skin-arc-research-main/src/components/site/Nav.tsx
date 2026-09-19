import { useEffect, useState } from "react";
import { Activity, Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";

const LINKS = [
  { id: "pipeline", label: "Pipeline" },
  { id: "workspace", label: "Workspace" },
  { id: "trace", label: "RAG Trace" },
  { id: "knowledge", label: "Knowledge" },
  { id: "architecture", label: "Architecture" },
  { id: "metrics", label: "System" },
];

export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const go = (id: string) => {
    setOpen(false);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-all duration-500",
        scrolled ? "glass border-b border-border" : "border-b border-transparent",
      )}
    >
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">
        <button onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })} className="flex items-center gap-2.5">
          <span className="relative flex h-8 w-8 items-center justify-center rounded-lg border border-primary/40 bg-primary/10">
            <Activity className="h-4 w-4 text-primary" />
          </span>
          <span className="text-sm font-semibold tracking-tight">
            Melanoma<span className="text-primary">AI</span>
          </span>
        </button>

        <div className="hidden items-center gap-7 md:flex">
          {LINKS.map((l) => (
            <button
              key={l.id}
              onClick={() => go(l.id)}
              className="text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              {l.label}
            </button>
          ))}
          <button
            onClick={() => go("workspace")}
            className="rounded-full border border-primary/40 bg-primary/10 px-4 py-2 text-sm text-primary transition-colors hover:bg-primary/20"
          >
            Analyze Lesion
          </button>
        </div>

        <button
          className="md:hidden"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </nav>

      {open ? (
        <div className="glass border-t border-border px-5 pb-5 md:hidden">
          {LINKS.map((l) => (
            <button
              key={l.id}
              onClick={() => go(l.id)}
              className="block w-full py-3 text-left text-sm text-muted-foreground"
            >
              {l.label}
            </button>
          ))}
        </div>
      ) : null}
    </header>
  );
}
