"""
Centralized Configuration — Melanoma Agentic RAG v4.0

All configuration values are defined here and loaded from
environment variables (.env).  This eliminates magic numbers
scattered across agent.py, api.py, create_vector_db.py, etc.

Usage:
    from config import settings
    print(settings.EMBEDDING_MODEL)
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables.
    Values in .env override defaults.
    """

    # ── API Keys ──────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(
        ..., description="Google Generative AI API key"
    )
    API_AUTH_KEY: str = Field(
        default="", description="API authentication key for protected endpoints (leave empty to disable auth)"
    )

    # ── LLM ───────────────────────────────────────────────────
    LLM_MODEL: str = Field(
        default="gemini-3.1-flash-lite",
        description="Gemini model name for generation"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.1, ge=0.0, le=2.0,
        description="LLM temperature for report generation"
    )
    LLM_EVAL_TEMPERATURE: float = Field(
        default=0.0, ge=0.0, le=2.0,
        description="LLM temperature for evidence evaluation (should be 0 for determinism)"
    )

    # ── Embedding Model ───────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Sentence embedding model (upgraded from bge-small to bge-base)"
    )

    # ── Reranker ──────────────────────────────────────────────
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranker model (cached locally)"
    )
    RERANKER_MIN_SCORE: float = Field(
        default=-5.0,
        description="Minimum reranker score threshold; passages below this are discarded"
    )

    # ── Qdrant ────────────────────────────────────────────────
    QDRANT_MODE: str = Field(
        default="local",
        description="'local' for disk-based Qdrant, 'cloud' for Qdrant Cloud"
    )
    QDRANT_LOCAL_PATH: str = Field(
        default="./qdrant_data",
        description="Local disk path for Qdrant storage"
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="melanoma_guidelines",
        description="Qdrant collection name"
    )
    QDRANT_CLOUD_URL: str = Field(
        default="",
        description="Qdrant Cloud cluster URL (e.g. https://xyz.us-east.aws.cloud.qdrant.io)"
    )
    QDRANT_CLOUD_API_KEY: str = Field(
        default="",
        description="Qdrant Cloud API key"
    )

    # ── Chunking ──────────────────────────────────────────────
    CHILD_CHUNK_SIZE: int = Field(
        default=500,
        description="Child chunk size for retrieval embeddings (focused)"
    )
    CHILD_CHUNK_OVERLAP: int = Field(
        default=100,
        description="Child chunk overlap"
    )
    PARENT_CHUNK_SIZE: int = Field(
        default=2000,
        description="Parent chunk size for LLM context (comprehensive)"
    )
    PARENT_CHUNK_OVERLAP: int = Field(
        default=200,
        description="Parent chunk overlap"
    )

    # ── Retrieval ─────────────────────────────────────────────
    DENSE_RETRIEVE_K: int = Field(
        default=10,
        description="Number of dense vector search results"
    )
    BM25_RETRIEVE_K: int = Field(
        default=10,
        description="Number of BM25 keyword search results"
    )
    RERANK_TOP_K: int = Field(
        default=5,
        description="Number of passages after cross-encoder reranking"
    )
    RRF_CONSTANT: int = Field(
        default=60,
        description="Reciprocal Rank Fusion constant k"
    )

    # ── Agentic Reasoning ─────────────────────────────────────
    MAX_REASONING_CYCLES: int = Field(
        default=3,
        description="Maximum multi-hop reasoning cycles"
    )
    MAX_EVIDENCE_ITEMS: int = Field(
        default=15,
        description="Maximum evidence passages to include in LLM prompt"
    )
    ENABLE_HYDE: bool = Field(
        default=True,
        description="Enable HyDE (Hypothetical Document Embeddings) query enhancement"
    )

    # ── BM25 ──────────────────────────────────────────────────
    BM25_INDEX_PATH: str = Field(
        default="./qdrant_data/bm25_index.pkl",
        description="Path to pickled BM25 index"
    )
    BM25_DOCS_PATH: str = Field(
        default="./qdrant_data/bm25_documents.json",
        description="Path to BM25 document texts/metadata"
    )
    PARENT_CHUNKS_PATH: str = Field(
        default="./qdrant_data/parent_chunks.json",
        description="Path to parent chunk store"
    )

    # ── Medical Literature ────────────────────────────────────
    MEDICAL_LITERATURE_DIR: str = Field(
        default="./medical_literature/",
        description="Directory containing source PDF files"
    )

    # ── API Security ──────────────────────────────────────────
    MAX_FILE_SIZE_BYTES: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum upload file size in bytes (10 MB)"
    )
    RATE_LIMIT: str = Field(
        default="10/minute",
        description="Rate limit for expensive endpoints (slowapi format)"
    )
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
        description="Allowed CORS origins (no wildcards when credentials are enabled)"
    )

    # ── Observability ─────────────────────────────────────────
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable performance metrics collection"
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format: 'json' for structured, 'text' for human-readable"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# ── Global singleton ──────────────────────────────────────────

settings = Settings()
