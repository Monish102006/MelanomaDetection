export interface PipelineNode {
  id: string;
  label: string;
  detail: string;
  group: "vision" | "model" | "agent" | "synthesis";
}

export const PIPELINE_NODES: PipelineNode[] = [
  {
    id: "image",
    label: "Image",
    detail: "A dermoscopic or clinical lesion image is submitted by the researcher.",
    group: "vision",
  },
  {
    id: "segmentation",
    label: "Lesion Segmentation",
    detail: "The lesion region is isolated from surrounding skin to bound all measurements.",
    group: "vision",
  },
  {
    id: "abcd",
    label: "ABCD Feature Extraction",
    detail:
      "Asymmetry, border irregularity, colour variation and diameter are computed from the mask.",
    group: "vision",
  },
  {
    id: "prediction",
    label: "Model Prediction",
    detail: "A research classifier returns a label and a confidence score. Not a diagnosis.",
    group: "model",
  },
  {
    id: "query",
    label: "Agentic Query Generation",
    detail: "The agent turns measurements into targeted literature queries.",
    group: "agent",
  },
  {
    id: "qdrant",
    label: "Qdrant Retrieval",
    detail: "Dense vector search over the indexed dermatology corpus returns candidate passages.",
    group: "agent",
  },
  {
    id: "evaluation",
    label: "Evidence Evaluation",
    detail:
      "Retrieved passages are scored for sufficiency; insufficient evidence triggers re-querying.",
    group: "agent",
  },
  {
    id: "gemini",
    label: "Gemini Synthesis",
    detail: "The LLM composes an explanation strictly grounded in the retrieved passages.",
    group: "synthesis",
  },
  {
    id: "report",
    label: "Evidence-Grounded Report",
    detail: "Every statement is traceable to a cited document and page.",
    group: "synthesis",
  },
];

export interface KnowledgeDoc {
  id: string;
  title: string;
  type: "Evidence review" | "Technical paper" | "Clinical guideline" | "Atlas chapter";
  pages: number;
  evidenceLevel: "Guideline" | "Peer-reviewed" | "Reference";
  topic: string;
  summary: string;
}

export const KNOWLEDGE_DOCS: KnowledgeDoc[] = [
  {
    id: "abcd-abcde",
    title: "ABCD-ABCDE evidence.pdf",
    type: "Evidence review",
    pages: 14,
    evidenceLevel: "Peer-reviewed",
    topic: "Clinical criteria",
    summary:
      "Origins, refinement and reported diagnostic performance of the ABCD and ABCDE clinical criteria for pigmented lesion assessment.",
  },
  {
    id: "dermoscopy-abcd",
    title: "Dermoscopy + ABCD.pdf",
    type: "Atlas chapter",
    pages: 22,
    evidenceLevel: "Reference",
    topic: "Dermoscopy",
    summary:
      "Dermoscopic pattern analysis and how ABCD scoring is applied to dermoscopic rather than naked-eye images.",
  },
  {
    id: "explainable-melanoma",
    title: "Explainable Melanoma Diagnosis paper",
    type: "Technical paper",
    pages: 11,
    evidenceLevel: "Peer-reviewed",
    topic: "Explainability",
    summary:
      "Feature-attribution and criteria-aligned explanation methods for melanoma classification models.",
  },
  {
    id: "aad-guideline",
    title: "AAD clinical guideline",
    type: "Clinical guideline",
    pages: 38,
    evidenceLevel: "Guideline",
    topic: "Care pathway",
    summary:
      "American Academy of Dermatology guidance on the evaluation and management of primary cutaneous melanoma.",
  },
  {
    id: "melanoma-pdf",
    title: "melanoma.pdf",
    type: "Evidence review",
    pages: 19,
    evidenceLevel: "Reference",
    topic: "Epidemiology",
    summary:
      "Background reference on melanoma subtypes, risk factors and the role of early lesion assessment.",
  },
];

export interface MetricCard {
  label: string;
  value: string;
  note: string;
}

/**
 * Configuration facts about the running system. Values marked as "reported by
 * backend" are replaced at runtime when the FastAPI service exposes them.
 * No accuracy / sensitivity numbers are fabricated here.
 */
export const SYSTEM_FACTS: MetricCard[] = [
  { label: "Documents indexed", value: "—", note: "Reported by backend index" },
  { label: "Chunks indexed", value: "—", note: "Reported by backend index" },
  { label: "Embedding model", value: "Backend-configured", note: "Dense text embeddings" },
  { label: "Vector database", value: "Qdrant", note: "Cosine similarity search" },
  { label: "LLM", value: "Gemini", note: "Grounded synthesis only" },
  { label: "Retrieval pipeline", value: "Agentic RAG", note: "Query → retrieve → evaluate → loop" },
  { label: "Evidence sources", value: `${KNOWLEDGE_DOCS.length} documents`, note: "Curated corpus" },
];

export const ARCHITECTURE_STEPS = [
  "User image",
  "Segmentation",
  "ABCD features",
  "Model prediction",
  "Agent",
  "Query generation",
  "Qdrant",
  "Medical literature",
  "Evidence evaluation",
  "Gemini",
  "Final report",
];
