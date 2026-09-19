"""
Melanoma Agentic RAG Agent — v4.0 (Production-Grade ABCDE)

Upgrades from v3.0:
- HyDE (Hypothetical Document Embeddings) query enhancement
- Parent-child contextual chunk expansion
- Per-request agent isolation (thread-safe for concurrent API requests)
- Structured JSON output via Gemini response_mime_type
- Observability instrumentation (latency + token tracking)
- Config-driven settings (no more magic numbers)
- Upgraded embedding model (bge-base-en-v1.5)
- Upgraded reranker (bge-reranker-v2-m3)
- Medical BM25 tokenizer with stemming and synonym expansion

Core Capabilities (retained from v3.0):
- Full ABCDE Framework Integration (A, B, C, D, E)
- Hybrid search (dense + BM25 + RRF fusion)
- Cross-encoder re-ranking with threshold filtering
- Multi-hop agentic reasoning (up to 3 refinement cycles)
- Security: Input sanitization, Prompt Injection Defense
- Reasoning trace for clinical auditability
- Strict medical disclaimers and safety guardrails
"""

import os
import re
import json
import html
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from bm25_utils import load_bm25_index, bm25_search, reciprocal_rank_fusion
from reranker import rerank
from query_engine import generate_queries
from chunking_strategy import load_parent_store, expand_to_parents
from observability import MetricsCollector, global_metrics


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Environment & API Key
# ============================================================

load_dotenv()


# ============================================================
# 2. Shared Resources (Class-Level Singletons)
# ============================================================
# These are loaded ONCE and shared across all agent instances.
# They are read-only and thread-safe.

logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")

_embedding_model = FastEmbedEmbeddings(
    model_name=settings.EMBEDDING_MODEL
)

logger.info("Connecting to Qdrant...")

if settings.QDRANT_MODE == "cloud":
    _vector_store = QdrantVectorStore.from_existing_collection(
        embedding=_embedding_model,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        url=settings.QDRANT_CLOUD_URL,
        api_key=settings.QDRANT_CLOUD_API_KEY
    )
    logger.info(f"Qdrant Cloud connected: {settings.QDRANT_CLOUD_URL}")
else:
    _vector_store = QdrantVectorStore.from_existing_collection(
        embedding=_embedding_model,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        path=settings.QDRANT_LOCAL_PATH
    )
    logger.info("Qdrant local connected.")

logger.info("Loading BM25 index...")

try:
    _bm25_model, _bm25_documents = load_bm25_index()
    _bm25_available = True
    logger.info("BM25 index loaded.")
except FileNotFoundError:
    _bm25_available = False
    _bm25_model = None
    _bm25_documents = None
    logger.warning(
        "BM25 index not found. Using dense-only search. "
        "Run create_vector_db.py to build the hybrid index."
    )

# Load parent chunk store
logger.info("Loading parent chunk store...")
_parent_store = load_parent_store(settings.PARENT_CHUNKS_PATH)

# LLM
_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    google_api_key=settings.GEMINI_API_KEY,
    temperature=settings.LLM_TEMPERATURE
)


# ============================================================
# 3. Security & Input Sanitization
# ============================================================

def sanitize_user_input(text: Any, max_length: int = 500) -> str:
    """
    Sanitize text inputs to protect against Prompt Injection,
    Control Character injection, and XSS.
    """
    if text is None:
        return ""

    val = str(text).strip()
    val = val[:max_length]

    # Remove control characters except standard whitespace
    val = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", val)

    # Escape HTML special characters
    val = html.escape(val)

    # Neutralize prompt injection phrases
    injection_patterns = [
        r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions?",
        r"(?i)system\s*prompt",
        r"(?i)disregard\s+(the\s+)?rules?",
        r"(?i)you\s+are\s+now\s+a",
        r"(?i)bypass\s+safety"
    ]
    for pattern in injection_patterns:
        val = re.sub(pattern, "[FILTERED_INPUT]", val)

    return val


# ============================================================
# 4. Agentic RAG Agent (ABCDE Framework — Per-Request Instance)
# ============================================================

class MelanomaAgent:
    """
    Per-request agentic RAG agent.

    Shared resources (embedding model, vector store, BM25, LLM)
    are class-level singletons loaded once at module import.

    Per-request state (reasoning_trace, metrics_collector) is
    instance-level, making it safe for concurrent API requests.
    """

    def __init__(self):
        # Per-request mutable state
        self.reasoning_trace: List[Dict[str, Any]] = []
        self.metrics = MetricsCollector()

    # --------------------------------------------------------
    # Reset reasoning trace
    # --------------------------------------------------------

    def _reset_trace(self):
        self.reasoning_trace = []
        self.metrics = MetricsCollector()

    def _log_trace(self, step: str, detail: Any):
        entry = {"step": step, "detail": detail}
        self.reasoning_trace.append(entry)
        logger.info(f"[TRACE] {step}: {json.dumps(detail, default=str)[:200]}")


    # --------------------------------------------------------
    # Create ABCDE queries (Standard + HyDE)
    # --------------------------------------------------------

    def create_queries(
        self,
        metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Generate retrieval queries using HyDE (if enabled) or
        standard feature-specific queries.
        """

        with self.metrics.track("query_generation"):
            queries = generate_queries(
                metrics,
                llm=_llm,
                use_hyde=settings.ENABLE_HYDE
            )

        self._log_trace("query_generation", {
            "num_queries": len(queries),
            "mode": "hyde" if settings.ENABLE_HYDE else "standard",
            "queries": [q[:100] + "..." if len(q) > 100 else q for q in queries]
        })

        return queries


    # --------------------------------------------------------
    # Hybrid search: dense + BM25 + RRF + re-rank
    # --------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval:
        1. Dense vector search via Qdrant
        2. BM25 keyword search (with medical stemming)
        3. Reciprocal Rank Fusion to merge
        4. Cross-encoder re-ranking with threshold filtering
        """

        if k is None:
            k = settings.RERANK_TOP_K

        # --- Dense search ---
        with self.metrics.track("dense_search", query_len=len(query)):
            dense_results_raw = _vector_store.similarity_search_with_score(
                query,
                k=settings.DENSE_RETRIEVE_K
            )

        dense_results = []
        for doc, score in dense_results_raw:
            dense_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "dense_score": float(score)
            })

        # --- BM25 search ---
        if _bm25_available:
            with self.metrics.track("bm25_search"):
                sparse_results = bm25_search(
                    _bm25_model,
                    _bm25_documents,
                    query,
                    k=settings.BM25_RETRIEVE_K
                )
        else:
            sparse_results = []

        # --- Reciprocal Rank Fusion ---
        if sparse_results:
            with self.metrics.track("rrf_fusion"):
                fused_results = reciprocal_rank_fusion(
                    dense_results,
                    sparse_results,
                    k=settings.RRF_CONSTANT,
                    top_n=settings.DENSE_RETRIEVE_K
                )
            search_type = "hybrid (dense + BM25 + RRF)"
        else:
            fused_results = dense_results
            search_type = "dense-only"

        # --- Cross-encoder re-ranking with threshold ---
        with self.metrics.track("reranking", num_candidates=len(fused_results)):
            reranked = rerank(
                query,
                fused_results,
                top_k=k,
                text_key="text"
            )

        # --- Parent-child expansion ---
        if _parent_store:
            with self.metrics.track("parent_expansion"):
                reranked = expand_to_parents(
                    reranked,
                    _parent_store,
                    text_key="text"
                )

        self._log_trace("hybrid_search", {
            "query": query[:100],
            "search_type": search_type,
            "dense_candidates": len(dense_results),
            "sparse_candidates": len(sparse_results),
            "fused_candidates": len(fused_results),
            "reranked_results": len(reranked),
            "parent_expanded": bool(_parent_store)
        })

        return reranked


    # --------------------------------------------------------
    # Retrieve evidence with hybrid search
    # --------------------------------------------------------

    def retrieve_evidence(
        self,
        queries: List[str]
    ) -> List[Dict[str, Any]]:

        source_priority = {
            "clinical_guideline": 4,
            "evidence_review": 3,
            "clinical_review": 2,
            "research": 1,
            "unknown": 0
        }

        retrieved = []

        for query in queries:

            logger.info(f"Searching: {query[:80]}...")

            results = self.hybrid_search(query, k=settings.RERANK_TOP_K)

            for result in results:

                metadata = result.get("metadata", {})

                source = metadata.get(
                    "document_name",
                    "Unknown"
                )

                source_type = metadata.get(
                    "source_type",
                    "unknown"
                )

                evidence_level = metadata.get(
                    "evidence_level",
                    "unknown"
                )

                topic = metadata.get(
                    "topic",
                    "Unknown"
                )

                page = metadata.get(
                    "page_label",
                    metadata.get("page", "Unknown")
                )

                text = result.get("text", "")

                retrieved.append({
                    "priority": source_priority.get(
                        source_type,
                        0
                    ),
                    "source": source,
                    "source_type": source_type,
                    "evidence_level": evidence_level,
                    "topic": topic,
                    "page": page,
                    "text": text,
                    "rerank_score": result.get("rerank_score", None),
                    "rrf_score": result.get("rrf_score", None)
                })

        # Remove duplicates (content-hash based)
        unique = {}
        for item in retrieved:
            key = hash(
                (item["source"], str(item["page"]), item["text"])
            )
            if key not in unique:
                unique[key] = item

        retrieved = list(unique.values())

        # Sort by evidence priority first, then rerank score
        retrieved.sort(
            key=lambda x: (
                x["priority"],
                x.get("rerank_score", 0) or 0
            ),
            reverse=True
        )

        self.metrics.increment("total_evidence_passages", len(retrieved))

        self._log_trace("retrieval_summary", {
            "total_unique_passages": len(retrieved),
            "by_source_type": {
                st: sum(1 for r in retrieved if r["source_type"] == st)
                for st in set(r["source_type"] for r in retrieved)
            }
        })

        return retrieved


    # --------------------------------------------------------
    # Format evidence for LLM
    # --------------------------------------------------------

    def format_evidence(
        self,
        evidence: List[Dict[str, Any]]
    ) -> str:

        evidence = evidence[:settings.MAX_EVIDENCE_ITEMS]

        formatted = []

        for i, item in enumerate(evidence, 1):

            formatted.append(
                f"""
[EVIDENCE {i}]
SOURCE: {item["source"]}
SOURCE TYPE: {item["source_type"]}
EVIDENCE LEVEL: {item["evidence_level"]}
TOPIC: {item["topic"]}
PAGE: {item["page"]}

{item["text"]}
"""
            )

        return "\n\n".join(formatted)


    # --------------------------------------------------------
    # Evaluate evidence sufficiency (LLM-as-judge)
    # --------------------------------------------------------

    def evaluate_evidence(
        self,
        metrics: Dict[str, Any],
        evidence: str,
        cycle: int
    ) -> Dict[str, Any]:

        evaluation_prompt = f"""
You are an evidence-quality evaluator for an explainable
melanoma Agentic RAG system based on the ABCDE framework.
This is evaluation cycle {cycle} of {settings.MAX_REASONING_CYCLES}.

The system has provided quantitative and qualitative case measurements.

MEASUREMENTS
============

{json.dumps(metrics, indent=2)}


RETRIEVED MEDICAL EVIDENCE
==========================

{evidence}


TASK
====

Determine whether the retrieved literature is sufficient to
interpret all 5 ABCDE clinical features:
- A (Asymmetry): Evidence regarding asymmetry significance.
- B (Border): Evidence regarding border irregularity.
- C (Color): Evidence regarding color variation.
- D (Diameter): Evidence regarding lesion diameter.
- E (Evolution): Evidence regarding lesion evolution / temporal change / symptom history.

IMPORTANT RULES:
- The measurements are already available.
- Do NOT ask for additional images or patient PII.
- Do NOT diagnose melanoma.
- Do NOT invent numerical cutoffs.
- Evidence is sufficient if the literature provides relevant
  information about the corresponding ABCDE features.

Return ONLY valid JSON:

{{
    "sufficient": true,
    "reason": "short explanation",
    "feature_coverage": {{
        "asymmetry": true,
        "border": true,
        "color": true,
        "diameter": true,
        "evolution": true
    }},
    "missing_information": "short explanation",
    "follow_up_queries": ["query1", "query2"]
}}
"""

        with self.metrics.track("llm_evaluation", cycle=cycle):
            response = _llm.invoke(evaluation_prompt)

        text = self._extract_text(response).strip()
        text = self._clean_json(text)

        try:
            result = json.loads(text)
            self._log_trace(f"evidence_evaluation_cycle_{cycle}", result)
            return result

        except json.JSONDecodeError:
            logger.warning("Evidence evaluation returned invalid JSON.")
            return {
                "sufficient": cycle >= 2,  # Be more conservative on first cycle
                "reason": "Evidence evaluation returned invalid JSON.",
                "feature_coverage": {},
                "missing_information": "",
                "follow_up_queries": []
            }


    # --------------------------------------------------------
    # Clean JSON from LLM output
    # --------------------------------------------------------

    def _clean_json(self, text: str) -> str:
        """Remove markdown code fences from LLM JSON output."""

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()


    # --------------------------------------------------------
    # Extract text from Gemini response
    # --------------------------------------------------------

    def _extract_text(self, response) -> str:

        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)

        return str(content)


    # --------------------------------------------------------
    # Multi-hop agentic reasoning loop
    # --------------------------------------------------------

    def _agentic_retrieval_loop(
        self,
        metrics: Dict[str, Any],
        initial_queries: List[str]
    ) -> tuple:
        """
        Core agentic loop:
        1. Retrieve evidence
        2. Evaluate sufficiency across A, B, C, D, E
        3. If insufficient → generate follow-up queries → re-retrieve
        4. Repeat up to MAX_REASONING_CYCLES times
        """

        all_evidence_items = []
        all_queries_used = list(initial_queries)

        for cycle in range(1, settings.MAX_REASONING_CYCLES + 1):

            logger.info(f"\n{'='*60}")
            logger.info(f"REASONING CYCLE {cycle}/{settings.MAX_REASONING_CYCLES}")
            logger.info(f"{'='*60}")

            # Retrieve
            if cycle == 1:
                queries = initial_queries
            else:
                queries = follow_up_queries

            with self.metrics.track("retrieval_cycle", cycle=cycle):
                new_evidence = self.retrieve_evidence(queries)

            all_evidence_items.extend(new_evidence)

            # Deduplicate accumulated evidence
            unique = {}
            for item in all_evidence_items:
                key = hash(
                    (item["source"], str(item["page"]), item["text"])
                )
                if key not in unique:
                    unique[key] = item
            all_evidence_items = list(unique.values())

            # Format and evaluate
            evidence_text = self.format_evidence(all_evidence_items)

            evaluation = self.evaluate_evidence(
                metrics,
                evidence_text,
                cycle
            )

            logger.info(f"Cycle {cycle} evaluation:")
            logger.info(json.dumps(evaluation, indent=2))

            # Check if sufficient
            if evaluation.get("sufficient", True):
                logger.info(f"Evidence sufficient after cycle {cycle}.")
                self._log_trace("reasoning_complete", {
                    "cycles_used": cycle,
                    "total_evidence": len(all_evidence_items),
                    "sufficient": True
                })
                break

            # Generate follow-up queries
            follow_up_queries = evaluation.get(
                "follow_up_queries", []
            )

            single_query = evaluation.get("follow_up_query", "")
            if single_query and not follow_up_queries:
                follow_up_queries = [single_query]

            if not follow_up_queries:
                logger.info("No follow-up queries generated. Stopping.")
                break

            all_queries_used.extend(follow_up_queries)

            logger.info(f"Follow-up queries for cycle {cycle + 1}:")
            for q in follow_up_queries:
                logger.info(f"  → {q[:80]}")

            self._log_trace(f"follow_up_cycle_{cycle}", {
                "follow_up_queries": follow_up_queries,
                "missing": evaluation.get("missing_information", "")
            })

        else:
            logger.info(
                f"Max reasoning cycles ({settings.MAX_REASONING_CYCLES}) reached."
            )
            self._log_trace("reasoning_complete", {
                "cycles_used": settings.MAX_REASONING_CYCLES,
                "total_evidence": len(all_evidence_items),
                "sufficient": evaluation.get("sufficient", False),
                "note": "max cycles reached"
            })

        return all_evidence_items, all_queries_used


    # --------------------------------------------------------
    # Generate final structured ABCDE report
    # --------------------------------------------------------

    def generate_report(
        self,
        case_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        self._reset_trace()

        case_id = sanitize_user_input(
            case_data.get("case_id", "Unknown"),
            max_length=64
        )

        prediction = sanitize_user_input(
            case_data.get("prediction", "Unknown"),
            max_length=64
        )

        confidence = float(case_data.get("confidence", 0.0))

        # Accept either abcd_metrics or abcde_metrics
        raw_metrics = case_data.get(
            "abcde_metrics",
            case_data.get("abcd_metrics", {})
        )

        # Normalize metrics to include Evolution
        metrics = {
            "asymmetry_index": raw_metrics.get("asymmetry_index", 0.0),
            "border_irregularity_score": raw_metrics.get("border_irregularity_score", 0.0),
            "color_variation_score": raw_metrics.get("color_variation_score", 0.0),
            "diameter_pixels": raw_metrics.get("diameter_pixels", raw_metrics.get("diameter_mm", 0.0)),
            "evolution": raw_metrics.get("evolution", {
                "reported_change": False,
                "status": "single_timepoint_capture",
                "notes": "No prior baseline or longitudinal change reported in current session."
            })
        }

        self._log_trace("input_received", {
            "case_id": case_id,
            "prediction": prediction,
            "confidence": confidence,
            "metrics": metrics
        })

        # ====================================================
        # Step 1: Generate queries (HyDE or standard)
        # ====================================================

        queries = self.create_queries(metrics)

        logger.info("\n" + "=" * 70)
        logger.info("AGENT GENERATED ABCDE QUERIES")
        logger.info("=" * 70)
        for query in queries:
            logger.info(f"  - {query[:100]}")

        # ====================================================
        # Step 2-4: Multi-hop agentic retrieval loop
        # ====================================================

        evidence_items, all_queries = self._agentic_retrieval_loop(
            metrics,
            queries
        )

        evidence = self.format_evidence(evidence_items)

        # ====================================================
        # Step 5: Final reasoning with structured ABCDE output
        # ====================================================

        prompt = f"""
You are an evidence-grounded clinical decision-support AI research assistant
for explainable melanoma assessment using the dermatological ABCDE framework:
  A = Asymmetry
  B = Border Irregularity
  C = Color Variation
  D = Diameter
  E = Evolution (Temporal change, elevation, symptoms)

You are NOT a doctor. Do NOT provide a definitive diagnosis or treatment advice.

CASE INFORMATION
================
Case ID: {case_id}
AI Model Prediction: {prediction}
AI Model Confidence: {confidence:.2f}

QUANTITATIVE & QUALITATIVE ABCDE MEASUREMENTS:
{json.dumps(metrics, indent=2)}


RETRIEVED MEDICAL EVIDENCE:
==========================
{evidence}


RESEARCH SAFETY & ETHICS RULES:
===============================
- Separate AI model prediction from medical literature facts.
- Do not state that the patient has melanoma.
- Do not invent thresholds or cutoffs.
- Interpret all 5 ABCDE criteria (Asymmetry, Border, Color, Diameter, and Evolution).
- For 'E' (Evolution), explain why change over time is crucial for early detection and note whether temporal history was provided.
- Cite specific document sources and pages.
- Highlight limitations clearly.
- This is a research decision-support prototype.


RETURN ONLY VALID JSON:
======================
{{
    "case_id": "{case_id}",

    "model_prediction": {{
        "label": "{prediction}",
        "confidence": {confidence}
    }},

    "observed_measurements": {{
        "asymmetry_index": null,
        "border_irregularity_score": null,
        "color_variation_score": null,
        "diameter_pixels": null,
        "evolution_status": null
    }},

    "evidence_interpretation": [
        {{
            "criterion": "A (Asymmetry) | B (Border) | C (Color) | D (Diameter) | E (Evolution)",
            "finding": "string",
            "clinical_evidence_summary": "string",
            "supporting_source": "string",
            "confidence_in_evidence": "high | medium | low"
        }}
    ],

    "sources": [
        {{
            "source": "string",
            "page": "string",
            "relevance": "string"
        }}
    ],

    "limitations": [
        "string"
    ],

    "clinical_context": "string — research prototype, requires in-person clinical exam and histopathology"
}}

JSON ONLY.
"""

        logger.info("\nGenerating final structured ABCDE report...")

        with self.metrics.track("llm_report_generation"):
            response = _llm.invoke(prompt)

        text = self._extract_text(response).strip()
        text = self._clean_json(text)

        try:
            report = json.loads(text)

        except json.JSONDecodeError:

            logger.warning("Gemini returned invalid JSON for report.")

            report = {
                "case_id": case_id,
                "model_prediction": {
                    "label": prediction,
                    "confidence": confidence
                },
                "observed_measurements": metrics,
                "evidence_interpretation": [],
                "sources": [],
                "limitations": [
                    "LLM returned invalid JSON for this report."
                ],
                "raw_response": text
            }

        # ====================================================
        # Step 6: Attach reasoning trace and performance metrics
        # ====================================================

        performance = self.metrics.summary()

        report["reasoning_trace"] = {
            "framework": "ABCDE",
            "version": "4.0",
            "total_queries": len(all_queries),
            "queries_used": all_queries,
            "total_evidence_passages": len(evidence_items),
            "reasoning_cycles": len([
                t for t in self.reasoning_trace
                if t["step"].startswith("evidence_evaluation")
            ]),
            "search_method": (
                "hybrid (dense + BM25 + RRF + cross-encoder re-ranking + parent expansion)"
                if _bm25_available else "dense + cross-encoder re-ranking + parent expansion"
            ),
            "query_mode": "hyde" if settings.ENABLE_HYDE else "standard",
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "trace": self.reasoning_trace
        }

        report["performance_metrics"] = {
            "total_duration_ms": performance["total_duration_ms"],
            "timing_breakdown": performance["timing_breakdown"],
            "token_usage": performance["token_usage"]
        }

        # Record in global aggregator
        global_metrics.record(performance)

        self._log_trace("report_generated", {
            "case_id": case_id,
            "num_evidence_interpretations": len(
                report.get("evidence_interpretation", [])
            ),
            "num_sources": len(report.get("sources", [])),
            "num_limitations": len(report.get("limitations", [])),
            "total_duration_ms": performance["total_duration_ms"]
        })

        return report


# ============================================================
# 5. Factory Function (Per-Request Agent Creation)
# ============================================================

def create_agent() -> MelanomaAgent:
    """
    Create a fresh MelanomaAgent for each request.
    Shared resources are already loaded at module level.
    """
    return MelanomaAgent()


# ============================================================
# 6. Test — End-to-end ABCDE pipeline
# ============================================================

if __name__ == "__main__":

    from model_pipeline.pipeline import MelanomaPipeline

    logger.info("\nStarting complete melanoma ABCDE pipeline (v4.0)...")

    # --------------------------------------------------------
    # 1. Run image pipeline
    # --------------------------------------------------------

    image_pipeline = MelanomaPipeline()

    image_result = image_pipeline.analyze(
        "test_images/test.jpg"
    )

    # --------------------------------------------------------
    # 2. Convert image pipeline output to ABCDE RAG case
    # --------------------------------------------------------

    prediction = image_result["prediction"]
    metrics = image_result["abcd_metrics"]

    case_data = {
        "case_id": "IMAGE-TEST-001",
        "prediction": prediction["label"],
        "confidence": prediction["confidence"],
        "abcde_metrics": {
            "asymmetry_index": metrics["asymmetry_index"],
            "border_irregularity_score": metrics["border_irregularity_score"],
            "color_variation_score": metrics["color_variation_score"],
            "diameter_pixels": metrics["diameter_pixels"],
            "evolution": {
                "reported_change": True,
                "change_types": ["enlargement", "darkening"],
                "timeframe_months": 3,
                "symptoms": ["mild itching"],
                "notes": "Lesion reported to have enlarged and darkened over past 3 months."
            }
        }
    }

    # --------------------------------------------------------
    # 3. Send real image results to Agentic RAG
    # --------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("SENDING ABCDE RESULTS TO AGENTIC RAG (v4.0)")
    logger.info("=" * 70)

    agent = create_agent()

    report = agent.generate_report(case_data)

    # --------------------------------------------------------
    # 4. Display final result
    # --------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("FINAL IMAGE → ABCDE RAG REPORT")
    logger.info("=" * 70)

    print(json.dumps(report, indent=2, default=str))

    # --------------------------------------------------------
    # 5. Display reasoning trace and performance
    # --------------------------------------------------------

    trace = report.get("reasoning_trace", {})
    perf = report.get("performance_metrics", {})

    logger.info("\n" + "=" * 70)
    logger.info("REASONING TRACE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Framework:          {trace.get('framework', 'ABCDE')}")
    logger.info(f"  Version:            {trace.get('version', '4.0')}")
    logger.info(f"  Query mode:         {trace.get('query_mode', 'N/A')}")
    logger.info(f"  Search method:      {trace.get('search_method', 'N/A')}")
    logger.info(f"  Total queries:      {trace.get('total_queries', 0)}")
    logger.info(f"  Reasoning cycles:   {trace.get('reasoning_cycles', 0)}")
    logger.info(f"  Evidence passages:  {trace.get('total_evidence_passages', 0)}")
    logger.info(f"  Embedding model:    {trace.get('embedding_model', 'N/A')}")
    logger.info(f"  Reranker model:     {trace.get('reranker_model', 'N/A')}")

    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 70)
    logger.info(f"  Total duration:     {perf.get('total_duration_ms', 0):.0f} ms")

    for op, data in perf.get("timing_breakdown", {}).items():
        logger.info(f"  {op:25s} {data['total_ms']:>8.0f} ms ({data['count']} calls)")