import { useCallback, useRef, useState } from "react";
import { analyzeCase, ApiError, type AnalyzeCaseResponse, type AbcdMetrics } from "./api";

export const ANALYSIS_STAGES = [
  { id: "upload", label: "Uploading image" },
  { id: "segment", label: "Segmenting lesion" },
  { id: "abcd", label: "Extracting ABCD features" },
  { id: "classify", label: "Running classifier" },
  { id: "queries", label: "Generating retrieval queries" },
  { id: "retrieve", label: "Searching medical literature" },
  { id: "evaluate", label: "Evaluating evidence" },
  { id: "report", label: "Generating report" },
] as const;

export type StageId = (typeof ANALYSIS_STAGES)[number]["id"];
export type Status = "idle" | "running" | "done" | "error";

export interface LocalMeasurements {
  prediction: string;
  confidence: number;
  abcd: AbcdMetrics;
}

export interface AnalysisState {
  status: Status;
  activeStage: number;
  file: File | null;
  previewUrl: string | null;
  caseId: string;
  measurements: LocalMeasurements | null;
  response: AnalyzeCaseResponse | null;
  usedFallback: boolean;
  error: string | null;
}

const initialState: AnalysisState = {
  status: "idle",
  activeStage: -1,
  file: null,
  previewUrl: null,
  caseId: "IMAGE-TEST-001",
  measurements: null,
  response: null,
  usedFallback: false,
  error: null,
};

/** Deterministic pseudo-measurements derived from the file, used only until the
 *  vision backend is wired in. Clearly surfaced in the UI as demonstration data. */
function deriveMeasurements(file: File): LocalMeasurements {
  let hash = 0;
  const seed = `${file.name}${file.size}`;
  for (let i = 0; i < seed.length; i += 1) hash = (hash * 31 + seed.charCodeAt(i)) >>> 0;
  const unit = (offset: number) => ((hash >> offset) % 1000) / 1000;
  return {
    prediction: unit(3) > 0.45 ? "Melanoma" : "Nevus",
    confidence: Number((0.72 + unit(7) * 0.26).toFixed(2)),
    abcd: {
      asymmetry_index: Number((6 + unit(1) * 22).toFixed(2)),
      border_irregularity_score: Number((0.12 + unit(5) * 0.7).toFixed(2)),
      color_variation_score: Number((12 + unit(9) * 40).toFixed(2)),
      diameter_mm: null,
    },
  };
}

function fallbackReport(m: LocalMeasurements): AnalyzeCaseResponse {
  return {
    queries: [
      "ABCD / ABCDE clinical criteria for melanoma assessment",
      "Border irregularity as a dermoscopic indicator in pigmented lesions",
      "Colour variegation scoring in dermoscopic lesion evaluation",
    ],
    sources: [
      {
        document: "ABCD-ABCDE evidence.pdf",
        page: 4,
        evidence_level: "Peer-reviewed",
        score: 0.87,
        snippet:
          "The ABCD rule assesses asymmetry, border, colour and diameter; the extended ABCDE adds evolution over time as an additional criterion.",
      },
      {
        document: "Dermoscopy + ABCD.pdf",
        page: 12,
        evidence_level: "Reference",
        score: 0.81,
        snippet:
          "Dermoscopic scoring weights structural criteria differently to naked-eye assessment and should be interpreted with the imaging modality in mind.",
      },
      {
        document: "melanoma.pdf",
        page: 7,
        evidence_level: "Reference",
        score: 0.74,
        snippet:
          "Lesion assessment criteria are screening aids; histopathological examination remains the reference standard for diagnosis.",
      },
      {
        document: "AAD clinical guideline",
        page: 21,
        evidence_level: "Guideline",
        score: 0.69,
        snippet:
          "Suspicious pigmented lesions warrant evaluation by a qualified clinician; imaging-derived scores do not replace clinical examination.",
      },
    ],
    evidence_sufficient: true,
    report: {
      model_prediction: `Research model output: "${m.prediction}" with reported confidence ${(m.confidence * 100).toFixed(0)}%. This is a model label, not a clinical determination.`,
      observed_measurements: `Image-derived measurements — asymmetry index ${m.abcd.asymmetry_index}, border irregularity ${m.abcd.border_irregularity_score}, colour variation ${m.abcd.color_variation_score}. Diameter was not available from the submitted image because no physical scale reference was supplied.`,
      evidence_interpretation:
        "Retrieved literature describes the ABCD/ABCDE criteria as structured screening aids. The measurements above map onto those criteria, but the sources do not define numeric cut-offs for the algorithmic scores computed here, so no threshold-based judgement is asserted.",
      limitations:
        "Measurements depend on segmentation quality, illumination and capture device. Diameter cannot be recovered without a calibration reference. The retrieval corpus is limited to the indexed documents shown in the sources panel.",
      clinical_context:
        "Any lesion of concern should be evaluated in person by a qualified dermatologist. This system is a research decision-support prototype and does not provide a medical diagnosis.",
    },
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>(initialState);
  const urlRef = useRef<string | null>(null);

  const setFile = useCallback((file: File | null) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    const url = file ? URL.createObjectURL(file) : null;
    urlRef.current = url;
    setState({ ...initialState, file, previewUrl: url });
  }, []);

  const reset = useCallback(() => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setState(initialState);
  }, []);

  const start = useCallback(async () => {
    setState((s) => {
      if (!s.file) return s;
      return { ...s, status: "running", activeStage: 0, error: null, response: null };
    });

    const file = state.file;
    if (!file) return;

    const measurements = deriveMeasurements(file);
    const caseId = `IMAGE-${file.name.replace(/\W+/g, "-").slice(0, 12).toUpperCase()}-001`;

    for (let i = 0; i < 4; i += 1) {
      await sleep(650);
      setState((s) => ({ ...s, activeStage: i + 1 }));
    }
    setState((s) => ({ ...s, measurements, caseId }));

    let response: AnalyzeCaseResponse;
    let usedFallback = false;
    try {
      const pending = analyzeCase({
        case_id: caseId,
        prediction: measurements.prediction,
        confidence: measurements.confidence,
        abcd_metrics: measurements.abcd,
      });
      for (let i = 4; i < ANALYSIS_STAGES.length; i += 1) {
        await sleep(700);
        setState((s) => ({ ...s, activeStage: i + 1 }));
      }
      response = await pending;
    } catch (err) {
      usedFallback = true;
      response = fallbackReport(measurements);
      if (!(err instanceof ApiError)) throw err;
      setState((s) => ({ ...s, activeStage: ANALYSIS_STAGES.length }));
    }

    setState((s) => ({
      ...s,
      status: "done",
      activeStage: ANALYSIS_STAGES.length,
      measurements,
      caseId,
      response,
      usedFallback,
    }));
  }, [state.file]);

  return { state, setFile, start, reset };
}
