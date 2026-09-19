/**
 * API service layer for the MelanomaAI FastAPI backend.
 *
 * The base URL is configurable via VITE_API_BASE_URL so the frontend can be
 * pointed at a local dev backend (http://127.0.0.1:8000) or a hosted one.
 * No secrets (Gemini / Qdrant credentials) are ever held in the frontend —
 * every privileged call happens server-side.
 */

export const API_BASE_URL: string =
  (import.meta.env["VITE_API_BASE_URL"] as string | undefined)?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

export interface AbcdMetrics {
  asymmetry_index: number | null;
  border_irregularity_score: number | null;
  color_variation_score: number | null;
  diameter_mm: number | null;
}

export interface AnalyzeCaseRequest {
  case_id: string;
  prediction: string;
  confidence: number;
  abcd_metrics: AbcdMetrics;
}

export interface EvidenceSource {
  document: string;
  page?: number | string | null;
  snippet?: string | null;
  score?: number | null;
  evidence_level?: string | null;
}

export interface AnalyzeCaseResponse {
  case_id?: string;
  queries?: string[];
  sources?: EvidenceSource[];
  evidence_sufficient?: boolean;
  report?: {
    model_prediction?: string;
    observed_measurements?: string;
    evidence_interpretation?: string;
    limitations?: string;
    clinical_context?: string;
  };
  raw?: unknown;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 60_000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!res.ok) {
      throw new ApiError(`Backend responded with ${res.status} ${res.statusText}`, res.status);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("Request to the analysis backend timed out.", 408);
    }
    throw new ApiError(
      `Unable to reach the analysis backend at ${API_BASE_URL}. Is the FastAPI service running?`,
    );
  }
}

/** POST /analyze-case — returns the evidence-grounded report. */
export function analyzeCase(payload: AnalyzeCaseRequest): Promise<AnalyzeCaseResponse> {
  return request<AnalyzeCaseResponse>("/analyze-case", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Optional health probe used to show backend connectivity state. */
export async function checkBackend(): Promise<boolean> {
  try {
    await request<unknown>("/health", { method: "GET" }, 4000);
    return true;
  } catch {
    return false;
  }
}
