"""
Parent-Child Contextual Chunking Strategy — Melanoma Agentic RAG v4.0

Implements a two-tier chunking approach:

- **Child chunks** (500 chars): Small, focused chunks used for embedding
  and vector search.  Smaller chunks produce more precise embeddings
  that match specific clinical statements.

- **Parent chunks** (2000 chars): Larger context windows that encompass
  one or more child chunks.  After retrieval and reranking, the system
  swaps matched child chunks for their parent context, giving the LLM
  a much richer and more complete passage to reason over.

Each child stores a `parent_id` key in its metadata so the swap is O(1).

Usage:
    from chunking_strategy import create_parent_child_chunks, load_parent_store, expand_to_parents

    children, parent_store = create_parent_child_chunks(documents)
    # children → embed into Qdrant & BM25
    # parent_store → save to disk as JSON

    # At retrieval time:
    expanded = expand_to_parents(reranked_results, parent_store)
"""

import uuid
import json
import os
import logging
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


# ============================================================
# Sentence-aware separators
# ============================================================

SEPARATORS = [
    "\n\n",       # Paragraph break
    "\n",         # Line break
    ". ",         # Sentence end
    "; ",         # Clause break
    ", ",         # Comma
    " "           # Word break (last resort)
]


# ============================================================
# Create Parent-Child Chunks
# ============================================================

def create_parent_child_chunks(
    documents: List[Document],
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100
) -> Tuple[List[Document], Dict[str, Dict[str, Any]]]:
    """
    Split documents into a two-tier parent-child chunk hierarchy.

    Parameters
    ----------
    documents : List[Document]
        Raw LangChain Document objects (one per PDF page).
    parent_chunk_size : int
        Size of parent chunks (used for LLM context).
    parent_chunk_overlap : int
        Overlap between parent chunks.
    child_chunk_size : int
        Size of child chunks (used for embedding & retrieval).
    child_chunk_overlap : int
        Overlap between child chunks.

    Returns
    -------
    (children, parent_store)
        children: List[Document] — child chunks with `parent_id` in metadata
        parent_store: Dict[str, Dict] — mapping of parent_id to parent text & metadata
    """

    # Step 1: Create parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False
    )

    parent_chunks = parent_splitter.split_documents(documents)

    logger.info(f"Created {len(parent_chunks)} parent chunks (size={parent_chunk_size})")

    # Step 2: Create child chunks from each parent
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False
    )

    parent_store: Dict[str, Dict[str, Any]] = {}
    all_children: List[Document] = []

    for parent in parent_chunks:
        parent_id = f"parent-{uuid.uuid4().hex[:12]}"

        # Store parent
        parent_store[parent_id] = {
            "text": parent.page_content,
            "metadata": {
                k: str(v) for k, v in parent.metadata.items()
            }
        }

        # Split parent into children
        child_docs = child_splitter.split_documents([parent])

        for child in child_docs:
            child.metadata["parent_id"] = parent_id
            all_children.append(child)

    logger.info(
        f"Created {len(all_children)} child chunks "
        f"(size={child_chunk_size}) from {len(parent_chunks)} parents"
    )
    logger.info(
        f"Average children per parent: "
        f"{len(all_children) / max(len(parent_chunks), 1):.1f}"
    )

    return all_children, parent_store


# ============================================================
# Save / Load Parent Store
# ============================================================

def save_parent_store(
    parent_store: Dict[str, Dict[str, Any]],
    path: str
):
    """Save the parent chunk store to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False, indent=2)

    logger.info(f"Parent store saved: {path} ({len(parent_store)} parents)")


def load_parent_store(
    path: str
) -> Dict[str, Dict[str, Any]]:
    """Load the parent chunk store from a JSON file."""
    if not os.path.exists(path):
        logger.warning(f"Parent store not found at {path}. Parent expansion disabled.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        store = json.load(f)

    logger.info(f"Parent store loaded: {len(store)} parents")
    return store


# ============================================================
# Expand Child Results to Parent Context
# ============================================================

def expand_to_parents(
    child_results: List[Dict[str, Any]],
    parent_store: Dict[str, Dict[str, Any]],
    text_key: str = "text"
) -> List[Dict[str, Any]]:
    """
    After retrieval & reranking, swap child chunk text for
    the larger parent context.  Deduplicates parents so the
    LLM doesn't see the same passage twice.

    Parameters
    ----------
    child_results : List[Dict]
        Reranked child chunks (each has metadata.parent_id)
    parent_store : Dict
        Mapping of parent_id → {text, metadata}
    text_key : str
        Key in child_results that contains the chunk text

    Returns
    -------
    List[Dict] — deduplicated parent-level results with original
    child rerank scores preserved (best child score wins)
    """
    if not parent_store:
        # Fallback: return child results as-is
        return child_results

    seen_parents = {}
    expanded = []

    for child in child_results:
        metadata = child.get("metadata", {})
        parent_id = metadata.get("parent_id", "")

        if not parent_id or parent_id not in parent_store:
            # No parent available — keep child as-is
            expanded.append(child)
            continue

        if parent_id in seen_parents:
            # Already included this parent — skip to avoid duplication
            # But update rerank score if this child scored higher
            existing = seen_parents[parent_id]
            child_score = child.get("rerank_score", 0) or 0
            existing_score = existing.get("rerank_score", 0) or 0
            if child_score > existing_score:
                existing["rerank_score"] = child_score
            continue

        # Swap child text for parent text
        parent = parent_store[parent_id]
        expanded_result = {
            text_key: parent["text"],
            "metadata": {**metadata, **parent.get("metadata", {})},
            "rerank_score": child.get("rerank_score"),
            "rrf_score": child.get("rrf_score"),
            "parent_id": parent_id,
            "expanded_from_child": True
        }

        expanded.append(expanded_result)
        seen_parents[parent_id] = expanded_result

    logger.info(
        f"Expanded {len(child_results)} children → "
        f"{len(expanded)} parent contexts "
        f"({len(seen_parents)} unique parents)"
    )

    return expanded
