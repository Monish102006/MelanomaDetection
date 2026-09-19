"""
BM25 Keyword Search Utilities — v4.0 (Medical NLP Tokenizer)

Upgraded from v3.0:
- NLTK Porter Stemmer for medical term normalization
  (e.g. "asymmetrical" → "asymmetr", "melanocytic" → "melanocyt")
- English stopword removal
- Medical synonym expansion for key dermatological terms
- Improved deduplication in RRF fusion

Builds and queries a BM25 (sparse keyword) index alongside
the dense vector store.  At index time the chunked texts and
their metadata are pickled to disk so the agent can reload
them without re-parsing the PDFs.
"""

import os
import re
import json
import pickle
import logging
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ============================================================
# NLTK Setup (lazy-loaded)
# ============================================================

_stemmer = None
_stopwords = None


def _ensure_nltk():
    """Lazy-load NLTK components on first use."""
    global _stemmer, _stopwords

    if _stemmer is not None:
        return

    try:
        import nltk
        from nltk.stem import PorterStemmer

        # Download required data (silent if already present)
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        from nltk.corpus import stopwords

        _stemmer = PorterStemmer()
        _stopwords = set(stopwords.words("english"))

        logger.info("NLTK stemmer and stopwords loaded.")

    except ImportError:
        logger.warning(
            "NLTK not installed. Using basic tokenizer without "
            "stemming or stopword removal."
        )
        _stemmer = None
        _stopwords = set()


# ============================================================
# Medical Synonym Expansion
# ============================================================

MEDICAL_SYNONYMS = {
    "melanoma": ["melanocytic", "malignant", "neoplasm"],
    "nevus": ["nevi", "mole", "naevus", "naevi"],
    "asymmetry": ["asymmetrical", "asymmetric"],
    "border": ["margin", "edge", "boundary", "perimeter"],
    "color": ["colour", "pigment", "pigmentation", "chromatic"],
    "diameter": ["size", "dimension", "measurement"],
    "evolution": ["evolving", "change", "changing", "temporal", "dynamic"],
    "dermoscopy": ["dermatoscopy", "dermoscopic", "dermatoscopic"],
    "lesion": ["growth", "spot", "mark"],
    "biopsy": ["histopathology", "histopathological", "pathology"],
    "irregular": ["irregularity", "notched", "ragged"],
    "variegation": ["variation", "heterogeneous", "multicolored"],
    "pruritus": ["itching", "itch", "pruritic"],
    "erythema": ["redness", "erythematous"],
    "ulceration": ["ulcerated", "erosion"],
}


def expand_medical_terms(tokens: List[str]) -> List[str]:
    """
    Expand medical terms with their synonyms to improve
    BM25 recall.  Only adds synonyms that are not already
    in the token list.

    Example:
        ["melanoma", "border"] →
        ["melanoma", "melanocytic", "malignant", "neoplasm",
         "border", "margin", "edge", "boundary", "perimeter"]
    """
    expanded = list(tokens)

    for token in tokens:
        for base_term, synonyms in MEDICAL_SYNONYMS.items():
            # If token matches a base term, add its synonyms
            if token == base_term or (_stemmer and _stemmer.stem(token) == _stemmer.stem(base_term)):
                for syn in synonyms:
                    stemmed_syn = _stemmer.stem(syn) if _stemmer else syn
                    if stemmed_syn not in expanded:
                        expanded.append(stemmed_syn)

    return expanded


# ============================================================
# Tokenizer (Medical NLP)
# ============================================================

def tokenize(text: str, expand_synonyms: bool = False) -> List[str]:
    """
    Medical-aware tokenizer with stemming and stopword removal.

    Pipeline:
    1. Lowercase
    2. Extract alphanumeric tokens
    3. Remove English stopwords
    4. Apply Porter stemming
    5. (Optional) Expand medical synonyms

    Parameters
    ----------
    text : str
        Input text to tokenize
    expand_synonyms : bool
        If True, add medical synonym expansions (use for queries, not corpus)
    """
    _ensure_nltk()

    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)

    # Remove stopwords
    if _stopwords:
        tokens = [t for t in tokens if t not in _stopwords]

    # Apply stemming
    if _stemmer:
        tokens = [_stemmer.stem(t) for t in tokens]

    # Expand medical synonyms (for query-time only)
    if expand_synonyms:
        tokens = expand_medical_terms(tokens)

    return tokens


# ============================================================
# Build and Save
# ============================================================

def build_bm25_index(
    chunks: list,
    index_path: str = None,
    docs_path: str = None,
) -> BM25Okapi:
    """
    Build a BM25 index from LangChain Document chunks
    and persist it to disk.

    Parameters
    ----------
    chunks : list[Document]
        LangChain Document objects with .page_content and .metadata
    index_path : str
        Where to save the pickled BM25 model
    docs_path : str
        Where to save the document texts and metadata as JSON
    """
    # Import config for defaults
    try:
        from config import settings
        _index_path = index_path or settings.BM25_INDEX_PATH
        _docs_path = docs_path or settings.BM25_DOCS_PATH
    except ImportError:
        _index_path = index_path or "./qdrant_data/bm25_index.pkl"
        _docs_path = docs_path or "./qdrant_data/bm25_documents.json"

    logger.info(f"Building BM25 index from {len(chunks)} chunks (with medical stemming)...")

    # Tokenize corpus (no synonym expansion for indexing)
    corpus = [tokenize(chunk.page_content, expand_synonyms=False) for chunk in chunks]
    bm25 = BM25Okapi(corpus)

    # Save BM25 model
    os.makedirs(os.path.dirname(_index_path), exist_ok=True)

    with open(_index_path, "wb") as f:
        pickle.dump(bm25, f)

    # Save document texts and metadata
    documents = []
    for chunk in chunks:
        documents.append({
            "text": chunk.page_content,
            "metadata": {
                k: str(v) for k, v in chunk.metadata.items()
            }
        })

    with open(_docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    logger.info(
        f"BM25 index saved: {_index_path} "
        f"({len(chunks)} documents, medical stemming enabled)"
    )

    return bm25


# ============================================================
# Load
# ============================================================

def load_bm25_index(
    index_path: str = None,
    docs_path: str = None,
) -> tuple:
    """
    Load the BM25 index and document store from disk.

    Returns
    -------
    (bm25, documents) where documents is a list of dicts
    with 'text' and 'metadata' keys.
    """
    try:
        from config import settings
        _index_path = index_path or settings.BM25_INDEX_PATH
        _docs_path = docs_path or settings.BM25_DOCS_PATH
    except ImportError:
        _index_path = index_path or "./qdrant_data/bm25_index.pkl"
        _docs_path = docs_path or "./qdrant_data/bm25_documents.json"

    if not os.path.exists(_index_path):
        raise FileNotFoundError(
            f"BM25 index not found at {_index_path}. "
            f"Run create_vector_db.py first."
        )

    with open(_index_path, "rb") as f:
        bm25 = pickle.load(f)

    with open(_docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(
        f"BM25 index loaded: {len(documents)} documents"
    )

    return bm25, documents


# ============================================================
# Search (with synonym expansion)
# ============================================================

def bm25_search(
    bm25: BM25Okapi,
    documents: List[Dict[str, Any]],
    query: str,
    k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search the BM25 index with medical synonym expansion
    and return the top-k results with their scores.

    Synonym expansion is applied at query time only, so that
    a query for "melanoma border" also matches documents containing
    "melanocytic margin" or "malignant boundary".

    Returns
    -------
    List of dicts with keys: text, metadata, bm25_score
    """

    # Tokenize query WITH synonym expansion for better recall
    query_tokens = tokenize(query, expand_synonyms=True)
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text": documents[idx]["text"],
                "metadata": documents[idx]["metadata"],
                "bm25_score": float(scores[idx])
            })

    return results


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Merge dense (vector) and sparse (BM25) result lists
    using Reciprocal Rank Fusion (RRF).

    Uses a content hash for deduplication to avoid the
    collision issues of the previous text[:200] approach.

    Parameters
    ----------
    dense_results : list
        Results from dense vector search, each with 'text' key
    sparse_results : list
        Results from BM25 search, each with 'text' key
    k : int
        RRF constant (default 60)
    top_n : int
        Number of fused results to return
    """

    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    def _hash_doc(doc: Dict[str, Any]) -> str:
        """Content-based hash for deduplication."""
        text = doc.get("text", "")
        # Use hash of full text for collision-free dedup
        return str(hash(text))

    # Score dense results
    for rank, doc in enumerate(dense_results):
        key = _hash_doc(doc)
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc
        doc_map[key]["search_type"] = "hybrid"

    # Score sparse results
    for rank, doc in enumerate(sparse_results):
        key = _hash_doc(doc)
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc
        doc_map[key]["search_type"] = "hybrid"

    # Sort by fused score
    sorted_keys = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )[:top_n]

    fused = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc["rrf_score"] = round(scores[key], 6)
        fused.append(doc)

    return fused
