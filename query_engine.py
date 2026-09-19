"""
Query Engine — HyDE (Hypothetical Document Embeddings) v4.0

Provides two query generation strategies:

1. **Standard Queries**: Feature-specific keyword queries based on
   ABCDE metrics (same as v3.0 agent.create_queries).

2. **HyDE Queries**: Uses the LLM to generate hypothetical medical
   textbook paragraphs describing each ABCDE measurement.  These
   synthetic paragraphs embed much closer to actual medical literature
   in vector space than short keyword queries, dramatically improving
   dense retrieval recall.

Usage:
    from query_engine import generate_queries

    queries = generate_queries(metrics, llm, use_hyde=True)
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ============================================================
# Standard Queries (Feature-Specific Keywords)
# ============================================================

def generate_standard_queries(
    metrics: Dict[str, Any]
) -> List[str]:
    """
    Generate targeted retrieval queries based on all 5 ABCDE
    criteria.  This is the same logic from v3.0 agent.create_queries,
    extracted into a standalone module.

    Parameters
    ----------
    metrics : dict
        ABCDE metrics dictionary with keys:
        asymmetry_index, border_irregularity_score,
        color_variation_score, diameter_pixels, evolution
    """

    queries = [
        "ABCD ABCDE clinical prediction rules for cutaneous melanoma detection"
    ]

    # A — Asymmetry
    if metrics.get("asymmetry_index", 0) > 0:
        queries.append(
            "clinical significance of asymmetry in melanoma "
            "and suspicious pigmented lesions"
        )

    # B — Border
    if metrics.get("border_irregularity_score", 0) > 0:
        queries.append(
            "clinical significance of irregular border and "
            "notched margins in melanoma dermoscopy"
        )

    # C — Color
    if metrics.get("color_variation_score", 0) > 0:
        queries.append(
            "clinical significance of color variegation multiple "
            "colors and pigmentation in melanoma"
        )

    # D — Diameter
    if metrics.get("diameter_pixels", 0) > 0 or metrics.get("diameter_mm", 0) > 0:
        queries.append(
            "clinical significance of lesion diameter size and "
            "small diameter melanoma ABCDE"
        )

    # E — Evolution
    evolution_data = metrics.get("evolution", {})
    if isinstance(evolution_data, dict) and evolution_data.get("reported_change"):
        queries.append(
            "diagnostic importance of lesion evolution change in "
            "size shape color or symptoms in melanoma"
        )
    else:
        queries.append(
            "clinical significance of evolution E criterion in "
            "ABCDE rule for early melanoma detection"
        )

    # Management & guidelines
    queries.append(
        "clinical evaluation and management guidelines of "
        "suspicious primary cutaneous melanoma"
    )

    return queries


# ============================================================
# HyDE Queries (Hypothetical Document Embeddings)
# ============================================================

HYDE_PROMPT_TEMPLATE = """You are a dermatology textbook author.
Write a short medical textbook paragraph (2-3 sentences) that would
appear in a clinical dermatology reference describing a skin lesion
with the following {criterion} measurement.

Measurement: {description}

Write ONLY the textbook paragraph. No headings, no bullet points.
Sound like a professional dermatology reference (e.g., Fitzpatrick's).
"""


def generate_hyde_queries(
    metrics: Dict[str, Any],
    llm: Any
) -> List[str]:
    """
    Generate HyDE (Hypothetical Document Embeddings) queries.

    For each ABCDE criterion, asks the LLM to write a hypothetical
    medical textbook paragraph describing a lesion with those specific
    measurements.  The resulting paragraph embeds much closer to actual
    medical passages in vector space than a short keyword query.

    Parameters
    ----------
    metrics : dict
        ABCDE metrics dictionary
    llm : ChatGoogleGenerativeAI
        LLM instance for generation

    Returns
    -------
    List of hypothetical paragraphs to use as dense search queries
    """

    hyde_configs = []

    # A — Asymmetry
    asym = metrics.get("asymmetry_index", 0)
    if asym > 0:
        hyde_configs.append({
            "criterion": "asymmetry",
            "description": (
                f"Asymmetry index of {asym:.1f}%. "
                f"{'High asymmetry suggesting irregular growth pattern.' if asym > 20 else 'Mild asymmetry within normal variation range.'}"
            )
        })

    # B — Border
    border = metrics.get("border_irregularity_score", 0)
    if border > 0:
        hyde_configs.append({
            "criterion": "border irregularity",
            "description": (
                f"Border irregularity score of {border:.2f} (0-1 scale). "
                f"{'Highly irregular borders with notching and pseudopods.' if border > 0.5 else 'Mildly irregular borders.'}"
            )
        })

    # C — Color
    color = metrics.get("color_variation_score", 0)
    if color > 0:
        hyde_configs.append({
            "criterion": "color variation",
            "description": (
                f"Color variation score of {color:.1f}. "
                f"{'Significant polychromasia with multiple color components.' if color > 30 else 'Relatively homogeneous pigmentation.'}"
            )
        })

    # D — Diameter
    diameter = metrics.get("diameter_pixels", 0)
    diameter_mm = metrics.get("diameter_mm", 0)
    if diameter > 0 or (diameter_mm and diameter_mm > 0):
        size_desc = f"{diameter_mm:.1f}mm" if diameter_mm else f"{diameter:.0f} pixels"
        hyde_configs.append({
            "criterion": "diameter",
            "description": (
                f"Lesion diameter of {size_desc}. "
                f"{'Exceeds the 6mm diameter threshold in the ABCDE rule.' if (diameter_mm and diameter_mm > 6) or diameter > 150 else 'Below the traditional 6mm threshold but may still warrant evaluation.'}"
            )
        })

    # E — Evolution
    evolution = metrics.get("evolution", {})
    if isinstance(evolution, dict):
        if evolution.get("reported_change"):
            changes = evolution.get("change_types", [])
            timeframe = evolution.get("timeframe_months", "unknown")
            symptoms = evolution.get("symptoms", [])
            hyde_configs.append({
                "criterion": "evolution (temporal change)",
                "description": (
                    f"Lesion showing evolution over {timeframe} months. "
                    f"Reported changes: {', '.join(changes) if changes else 'general change'}. "
                    f"{'Associated symptoms: ' + ', '.join(symptoms) + '.' if symptoms else 'No associated symptoms.'}"
                )
            })
        else:
            hyde_configs.append({
                "criterion": "evolution (baseline assessment)",
                "description": (
                    "Single timepoint capture with no prior baseline for comparison. "
                    "Evolution criterion (E in ABCDE) cannot be assessed from a static image alone "
                    "and requires longitudinal monitoring or patient history."
                )
            })

    # Generate hypothetical paragraphs
    queries = []
    original_temp = getattr(llm, 'temperature', 0.1)

    try:
        # Use slightly creative temperature for HyDE
        llm.temperature = 0.3

        for config in hyde_configs:
            try:
                prompt = HYDE_PROMPT_TEMPLATE.format(**config)
                response = llm.invoke(prompt)

                content = response.content
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", str(block)) if isinstance(block, dict) else str(block)
                        for block in content
                    )

                if content and len(content.strip()) > 20:
                    queries.append(content.strip())
                    logger.info(
                        f"HyDE query for {config['criterion']}: "
                        f"{content.strip()[:80]}..."
                    )

            except Exception as e:
                logger.warning(
                    f"HyDE generation failed for {config['criterion']}: {e}"
                )

    finally:
        # Restore original temperature
        llm.temperature = original_temp

    # Always add standard management query (not HyDE)
    queries.append(
        "clinical evaluation and management guidelines of "
        "suspicious primary cutaneous melanoma"
    )

    return queries


# ============================================================
# Unified Query Generator
# ============================================================

def generate_queries(
    metrics: Dict[str, Any],
    llm: Any = None,
    use_hyde: bool = True
) -> List[str]:
    """
    Generate retrieval queries using either HyDE or standard mode.

    Parameters
    ----------
    metrics : dict
        ABCDE metrics
    llm : ChatGoogleGenerativeAI or None
        Required for HyDE mode
    use_hyde : bool
        If True, attempt HyDE generation; falls back to standard on error

    Returns
    -------
    List of query strings for retrieval
    """

    if use_hyde and llm is not None:
        try:
            hyde_queries = generate_hyde_queries(metrics, llm)
            if hyde_queries and len(hyde_queries) >= 2:
                logger.info(
                    f"Generated {len(hyde_queries)} HyDE queries "
                    f"(hypothetical document embeddings)"
                )
                return hyde_queries
            else:
                logger.warning(
                    "HyDE returned insufficient queries. "
                    "Falling back to standard queries."
                )
        except Exception as e:
            logger.error(
                f"HyDE generation failed: {e}. "
                f"Falling back to standard queries."
            )

    standard = generate_standard_queries(metrics)
    logger.info(f"Generated {len(standard)} standard queries")
    return standard
