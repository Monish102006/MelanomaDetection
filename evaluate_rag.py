"""
RAG Evaluation Script — Melanoma Agentic RAG v4.0 (RAGAS + Heuristic Hybrid)

Upgrades from v3.0:
- Real RAGAS metrics integration (faithfulness, answer_relevancy, context_precision)
- Latency tracking per test case and per pipeline stage
- Token usage reporting
- Combined scoring: RAGAS formal metrics + heuristic structural checks
- Comparison report showing before/after upgrade improvements

Evaluates retrieval and generation quality across all 5 ABCDE criteria:
1. Context Relevance — Are retrieved passages relevant to ABCDE rules?
2. Faithfulness & Safety — Adherence to medical disclaimers & literature facts.
3. Answer Completeness — Full coverage of A, B, C, D, E criteria.
4. Reasoning Quality — Hybrid search, re-ranking, multi-query auditability.
5. Performance — Latency and token usage tracking.

Usage:
    python evaluate_rag.py
"""

import os
import json
import time
import logging
from typing import Dict, Any, List

from dotenv import load_dotenv

from agent import create_agent

# ============================================================
# Setup
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


# ============================================================
# ABCDE Clinical Test Cases
# ============================================================

TEST_CASES = [
    {
        "name": "High-risk melanoma with rapid Evolution (ABCDE)",
        "case_data": {
            "case_id": "EVAL-ABCDE-001",
            "prediction": "Melanoma",
            "confidence": 0.94,
            "abcde_metrics": {
                "asymmetry_index": 38.5,
                "border_irregularity_score": 0.76,
                "color_variation_score": 45.2,
                "diameter_pixels": 320.0,
                "evolution": {
                    "reported_change": True,
                    "change_types": ["rapid_enlargement", "darkening", "elevation"],
                    "timeframe_months": 2.5,
                    "symptoms": ["itching", "bleeding_on_contact"],
                    "notes": "Patient noticed lesion growing and darkening over 10 weeks."
                }
            }
        },
        "expected_features": [
            "asymmetry", "border", "color", "diameter", "evolution"
        ],
        "expected_source_types": [
            "evidence_review", "clinical_guideline"
        ]
    },
    {
        "name": "Low-risk benign nevus (Stable baseline)",
        "case_data": {
            "case_id": "EVAL-ABCDE-002",
            "prediction": "Nevus",
            "confidence": 0.88,
            "abcde_metrics": {
                "asymmetry_index": 6.5,
                "border_irregularity_score": 0.12,
                "color_variation_score": 11.0,
                "diameter_pixels": 85.0,
                "evolution": {
                    "reported_change": False,
                    "status": "stable_over_years",
                    "notes": "No reported change in size, shape, or color for 3+ years."
                }
            }
        },
        "expected_features": [
            "asymmetry", "border", "color", "diameter", "evolution"
        ],
        "expected_source_types": ["evidence_review"]
    },
    {
        "name": "Small-Diameter Melanoma with Evolution E-flag",
        "case_data": {
            "case_id": "EVAL-ABCDE-003",
            "prediction": "Melanoma",
            "confidence": 0.72,
            "abcde_metrics": {
                "asymmetry_index": 24.0,
                "border_irregularity_score": 0.42,
                "color_variation_score": 28.0,
                "diameter_pixels": 70.0,
                "evolution": {
                    "reported_change": True,
                    "change_types": ["new_focal_pigment_spot"],
                    "timeframe_months": 4.0,
                    "symptoms": ["pruritus"],
                    "notes": "Small lesion exhibiting distinct focal evolution."
                }
            }
        },
        "expected_features": [
            "asymmetry", "border", "color", "diameter", "evolution"
        ],
        "expected_source_types": ["evidence_review", "dermoscopy_review"]
    },
    {
        "name": "Prominent Color Variegation & Border Notch",
        "case_data": {
            "case_id": "EVAL-ABCDE-004",
            "prediction": "Melanoma",
            "confidence": 0.89,
            "abcde_metrics": {
                "asymmetry_index": 19.5,
                "border_irregularity_score": 0.65,
                "color_variation_score": 58.0,
                "diameter_pixels": 210.0,
                "evolution": {
                    "reported_change": True,
                    "change_types": ["color_darkening"],
                    "timeframe_months": 6.0,
                    "symptoms": [],
                    "notes": "Gradual darkening from tan to dark brown/black."
                }
            }
        },
        "expected_features": [
            "asymmetry", "border", "color", "diameter", "evolution"
        ],
        "expected_source_types": ["evidence_review", "dermoscopy_review"]
    },
    {
        "name": "Single Static Image (No history provided)",
        "case_data": {
            "case_id": "EVAL-ABCDE-005",
            "prediction": "Suspicious Lesion",
            "confidence": 0.65,
            "abcde_metrics": {
                "asymmetry_index": 16.0,
                "border_irregularity_score": 0.35,
                "color_variation_score": 25.0,
                "diameter_pixels": 140.0,
                "evolution": {
                    "reported_change": False,
                    "status": "single_timepoint_capture",
                    "notes": "No temporal history available."
                }
            }
        },
        "expected_features": [
            "asymmetry", "border", "color", "diameter", "evolution"
        ],
        "expected_source_types": ["evidence_review", "clinical_guideline"]
    }
]


# ============================================================
# Heuristic Evaluation Metrics (from v3.0, retained)
# ============================================================

def evaluate_context_relevance(
    report: Dict[str, Any],
    case: Dict[str, Any]
) -> Dict[str, Any]:
    """Metric 1: Context Relevance (heuristic keyword check)."""

    sources = report.get("sources", [])
    trace = report.get("reasoning_trace", {})

    total_passages = trace.get("total_evidence_passages", 0)
    num_sources = len(sources)

    relevant_keywords = [
        "abcd", "abcde", "melanoma", "asymmetry", "border",
        "color", "diameter", "evolution", "evolving", "dermoscopy",
        "lesion", "pigmented", "clinical", "guideline", "criteria"
    ]

    relevant_count = 0
    for source in sources:
        source_text = json.dumps(source).lower()
        if any(kw in source_text for kw in relevant_keywords):
            relevant_count += 1

    relevance_ratio = (
        relevant_count / num_sources
        if num_sources > 0
        else 0
    )

    return {
        "metric": "context_relevance",
        "score": round(relevance_ratio, 3),
        "total_sources": num_sources,
        "relevant_sources": relevant_count,
        "total_passages_retrieved": total_passages,
        "pass": relevance_ratio >= 0.6
    }


def evaluate_faithfulness(
    report: Dict[str, Any]
) -> Dict[str, Any]:
    """Metric 2: Faithfulness & Safety (heuristic)."""

    report_text = json.dumps(report).lower()

    safety_checks = {
        "no_diagnosis_claim": not any(
            phrase in report_text for phrase in [
                "patient has melanoma",
                "diagnosed with melanoma",
                "confirms melanoma",
                "definitely melanoma"
            ]
        ),
        "mentions_limitations": (
            "limitation" in report_text or
            "limited" in report_text
        ),
        "mentions_research_context": (
            "research" in report_text or
            "not a diagnosis" in report_text or
            "decision-support" in report_text or
            "histopathol" in report_text
        ),
        "has_sources": len(report.get("sources", [])) > 0,
        "has_evidence_interpretation": (
            len(report.get("evidence_interpretation", [])) > 0
        )
    }

    passed = sum(safety_checks.values())
    total = len(safety_checks)

    return {
        "metric": "faithfulness",
        "score": round(passed / total, 3),
        "checks": safety_checks,
        "pass": passed >= 4
    }


def evaluate_answer_completeness(
    report: Dict[str, Any],
    expected_features: List[str]
) -> Dict[str, Any]:
    """Metric 3: Answer Completeness (ABCDE coverage)."""

    interpretations = report.get("evidence_interpretation", [])

    covered_features = set()
    for interp in interpretations:
        criterion = str(interp.get("criterion", "")).lower()
        finding = str(interp.get("finding", "")).lower()
        summary = str(interp.get("clinical_evidence_summary", interp.get("evidence_summary", ""))).lower()

        combined = f"{criterion} {finding} {summary}"

        if any(kw in combined for kw in ["asymmetry", "asymmetr", "a (", "criterion a"]):
            covered_features.add("asymmetry")
        if any(kw in combined for kw in ["border", "irregulari", "b (", "criterion b"]):
            covered_features.add("border")
        if any(kw in combined for kw in ["color", "colour", "variegation", "variation", "c (", "criterion c"]):
            covered_features.add("color")
        if any(kw in combined for kw in ["diameter", "size", "d (", "criterion d"]):
            covered_features.add("diameter")
        if any(kw in combined for kw in ["evolution", "evolving", "change", "temporal", "e (", "criterion e"]):
            covered_features.add("evolution")

    expected = set(expected_features)
    covered = covered_features.intersection(expected)

    completeness = (
        len(covered) / len(expected)
        if expected
        else 1.0
    )

    return {
        "metric": "answer_completeness",
        "score": round(completeness, 3),
        "expected_features": list(expected),
        "covered_features": list(covered),
        "missing_features": list(expected - covered),
        "pass": completeness >= 0.8
    }


def evaluate_reasoning_trace(
    report: Dict[str, Any]
) -> Dict[str, Any]:
    """Metric 4: Reasoning Quality & Auditability."""

    trace = report.get("reasoning_trace", {})

    checks = {
        "has_trace": bool(trace),
        "is_abcde": trace.get("framework") == "ABCDE" or "abcde" in json.dumps(trace).lower(),
        "multiple_queries": trace.get("total_queries", 0) >= 4,
        "has_reasoning_cycles": trace.get("reasoning_cycles", 0) >= 1,
        "used_hybrid_search": "hybrid" in trace.get("search_method", ""),
        "has_evidence": trace.get("total_evidence_passages", 0) >= 5
    }

    passed = sum(checks.values())
    total = len(checks)

    return {
        "metric": "reasoning_quality",
        "score": round(passed / total, 3),
        "checks": checks,
        "queries_used": trace.get("total_queries", 0),
        "reasoning_cycles": trace.get("reasoning_cycles", 0),
        "search_method": trace.get("search_method", "unknown"),
        "query_mode": trace.get("query_mode", "unknown"),
        "pass": passed >= 4
    }


# ============================================================
# Performance Metrics
# ============================================================

def evaluate_performance(
    report: Dict[str, Any],
    wall_time_ms: float
) -> Dict[str, Any]:
    """Metric 5: Performance — latency and cost tracking."""

    perf = report.get("performance_metrics", {})
    timing = perf.get("timing_breakdown", {})
    tokens = perf.get("token_usage", {})

    return {
        "metric": "performance",
        "wall_time_ms": round(wall_time_ms, 2),
        "agent_duration_ms": perf.get("total_duration_ms", 0),
        "timing_breakdown": {
            op: round(data["total_ms"], 2)
            for op, data in timing.items()
        },
        "total_tokens": tokens.get("total_tokens", 0),
        "llm_calls": len(tokens.get("calls", []))
    }


# ============================================================
# RAGAS Evaluation (Formal LLM-Graded)
# ============================================================

def evaluate_with_ragas(
    report: Dict[str, Any],
    case: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run RAGAS formal evaluation metrics.
    Falls back gracefully if RAGAS is not installed or fails.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        # Construct the RAGAS evaluation dataset
        trace = report.get("reasoning_trace", {})
        evidence_texts = []

        # Extract evidence texts from reasoning trace
        for step in trace.get("trace", []):
            if step.get("step", "").startswith("retrieval_summary"):
                pass  # Summary step, skip

        # Use sources from the report
        for source in report.get("sources", []):
            evidence_texts.append(json.dumps(source))

        # Construct RAGAS-compatible data
        question = (
            f"Analyze skin lesion case {case['case_data']['case_id']} "
            f"using the ABCDE framework with the given measurements."
        )

        answer = json.dumps(
            report.get("evidence_interpretation", []),
            default=str
        )

        contexts = evidence_texts if evidence_texts else ["No contexts available"]

        # RAGAS expects a Dataset
        eval_data = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        })

        result = ragas_evaluate(
            eval_data,
            metrics=[faithfulness, answer_relevancy, context_precision]
        )

        return {
            "metric": "ragas",
            "available": True,
            "faithfulness": round(result["faithfulness"], 3),
            "answer_relevancy": round(result["answer_relevancy"], 3),
            "context_precision": round(result["context_precision"], 3),
            "avg_score": round(
                (result["faithfulness"] + result["answer_relevancy"] + result["context_precision"]) / 3,
                3
            )
        }

    except ImportError:
        logger.warning("RAGAS or datasets not installed. Skipping formal RAGAS evaluation.")
        return {
            "metric": "ragas",
            "available": False,
            "note": "Install ragas and datasets packages for formal evaluation."
        }

    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {
            "metric": "ragas",
            "available": False,
            "error": str(e)
        }


# ============================================================
# Run Evaluation
# ============================================================

def run_evaluation():

    logger.info("=" * 70)
    logger.info("MELANOMA AGENTIC RAG (ABCDE v4.0) — EVALUATION SUITE")
    logger.info("=" * 70)

    results = []

    for i, test_case in enumerate(TEST_CASES, 1):

        logger.info(f"\n{'='*60}")
        logger.info(f"TEST CASE {i}/{len(TEST_CASES)}: {test_case['name']}")
        logger.info(f"{'='*60}")

        # Run the agent with timing
        start_time = time.perf_counter()
        agent = create_agent()
        report = agent.generate_report(test_case["case_data"])
        wall_time_ms = (time.perf_counter() - start_time) * 1000

        # Heuristic evaluations
        relevance = evaluate_context_relevance(report, test_case)
        faithfulness = evaluate_faithfulness(report)
        completeness = evaluate_answer_completeness(report, test_case["expected_features"])
        reasoning = evaluate_reasoning_trace(report)
        performance = evaluate_performance(report, wall_time_ms)

        # RAGAS formal evaluation
        ragas_result = evaluate_with_ragas(report, test_case)

        # Aggregate heuristic score
        heuristic_avg = (
            relevance["score"] +
            faithfulness["score"] +
            completeness["score"] +
            reasoning["score"]
        ) / 4

        case_result = {
            "test_case": test_case["name"],
            "case_id": test_case["case_data"]["case_id"],
            "metrics": {
                "context_relevance": relevance,
                "faithfulness": faithfulness,
                "answer_completeness": completeness,
                "reasoning_quality": reasoning,
                "performance": performance,
                "ragas": ragas_result
            },
            "heuristic_score": round(heuristic_avg, 3),
            "all_passed": all([
                relevance["pass"],
                faithfulness["pass"],
                completeness["pass"],
                reasoning["pass"]
            ])
        }

        results.append(case_result)

        # Display results
        logger.info(f"\n  Context Relevance:    {relevance['score']:.3f} {'✓' if relevance['pass'] else '✗'}")
        logger.info(f"  Faithfulness:         {faithfulness['score']:.3f} {'✓' if faithfulness['pass'] else '✗'}")
        logger.info(f"  Answer Completeness:  {completeness['score']:.3f} {'✓' if completeness['pass'] else '✗'}")
        logger.info(f"  Reasoning Quality:    {reasoning['score']:.3f} {'✓' if reasoning['pass'] else '✗'}")
        logger.info(f"  Heuristic Score:      {heuristic_avg:.3f}")
        logger.info(f"  Wall Time:            {wall_time_ms:.0f} ms")

        if ragas_result.get("available"):
            logger.info(f"  RAGAS Faithfulness:   {ragas_result['faithfulness']:.3f}")
            logger.info(f"  RAGAS Relevancy:      {ragas_result['answer_relevancy']:.3f}")
            logger.info(f"  RAGAS Precision:      {ragas_result['context_precision']:.3f}")

    # ========================================================
    # Summary
    # ========================================================

    logger.info("\n\n")
    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY (ABCDE v4.0)")
    logger.info("=" * 70)

    total_passed = sum(1 for r in results if r["all_passed"])
    avg_heuristic = sum(r["heuristic_score"] for r in results) / len(results)

    avg_relevance = sum(
        r["metrics"]["context_relevance"]["score"] for r in results
    ) / len(results)

    avg_faithfulness = sum(
        r["metrics"]["faithfulness"]["score"] for r in results
    ) / len(results)

    avg_completeness = sum(
        r["metrics"]["answer_completeness"]["score"] for r in results
    ) / len(results)

    avg_reasoning = sum(
        r["metrics"]["reasoning_quality"]["score"] for r in results
    ) / len(results)

    avg_wall_time = sum(
        r["metrics"]["performance"]["wall_time_ms"] for r in results
    ) / len(results)

    logger.info(f"  Test Cases:           {len(results)}")
    logger.info(f"  All Passed:           {total_passed}/{len(results)}")
    logger.info(f"")
    logger.info(f"  Avg Context Relevance:    {avg_relevance:.3f}")
    logger.info(f"  Avg Faithfulness:         {avg_faithfulness:.3f}")
    logger.info(f"  Avg Answer Completeness:  {avg_completeness:.3f}")
    logger.info(f"  Avg Reasoning Quality:    {avg_reasoning:.3f}")
    logger.info(f"")
    logger.info(f"  HEURISTIC SCORE:          {avg_heuristic:.3f}")
    logger.info(f"  Avg Wall Time:            {avg_wall_time:.0f} ms")

    # RAGAS summary
    ragas_available = any(
        r["metrics"]["ragas"].get("available", False) for r in results
    )
    if ragas_available:
        ragas_results_valid = [
            r for r in results if r["metrics"]["ragas"].get("available", False)
        ]
        if ragas_results_valid:
            avg_ragas = sum(
                r["metrics"]["ragas"]["avg_score"] for r in ragas_results_valid
            ) / len(ragas_results_valid)
            logger.info(f"  RAGAS AVG SCORE:          {avg_ragas:.3f}")

    logger.info("=" * 70)

    # Save results
    output_path = "./evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "framework": "ABCDE",
                "version": "4.0",
                "total_cases": len(results),
                "passed": total_passed,
                "avg_context_relevance": avg_relevance,
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_completeness": avg_completeness,
                "avg_reasoning_quality": avg_reasoning,
                "heuristic_score": avg_heuristic,
                "avg_wall_time_ms": avg_wall_time,
                "upgrades": [
                    "bge-base-en-v1.5 embeddings (768-dim)",
                    "bge-reranker-v2-m3 cross-encoder",
                    "Medical BM25 stemming + synonyms",
                    "Parent-child contextual chunking",
                    "HyDE query enhancement",
                    "PyMuPDF layout-aware PDF parsing"
                ]
            },
            "results": results
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_evaluation()
