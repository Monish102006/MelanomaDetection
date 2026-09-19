import { useEffect, useState, type ReactNode } from "react";

/** Renders children only after hydration — required for WebGL / browser-only libs. */
export function ClientOnly({ children, fallback = null }: { children: ReactNode; fallback?: ReactNode }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return <>{mounted ? children : fallback}</>;
}
