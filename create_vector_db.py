"""
Create the Qdrant vector database AND BM25 keyword index — v4.0

Upgrades from v3.0:
- Embedding model: bge-base-en-v1.5 (768-dim, 3x more capacity)
- PDF parsing: PyMuPDF for layout-aware extraction (tables, columns)
- Chunking: Parent-child strategy (small child chunks for retrieval,
  large parent chunks for LLM context)
- Qdrant Cloud: Toggle between local disk and cloud deployment
- Auto-metadata: Automatically detects new PDFs without hardcoding
- BM25: Medical stemming tokenizer with synonym expansion
"""

import os
import glob
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

from config import settings
from bm25_utils import build_bm25_index
from chunking_strategy import (
    create_parent_child_chunks,
    save_parent_store
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Load all PDF documents using PyMuPDF (layout-aware)
# ============================================================

logger.info("1. Loading medical PDFs with PyMuPDF (layout-aware parser)...")

try:
    import fitz  # PyMuPDF

    def load_pdfs_pymupdf(pdf_dir: str):
        """
        Load PDFs using PyMuPDF which preserves:
        - Table structures
        - Multi-column layouts
        - Figure captions
        - Proper text ordering
        """
        documents = []
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {pdf_dir}"
            )

        for pdf_path in sorted(pdf_files):
            filename = os.path.basename(pdf_path)
            logger.info(f"  Loading: {filename}")

            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text with layout preservation
                text = page.get_text("text")

                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "page": page_num,
                            "page_label": str(page_num + 1),
                            "total_pages": len(doc),
                            "document_name": filename
                        }
                    ))

            doc.close()

        return documents

    documents = load_pdfs_pymupdf(settings.MEDICAL_LITERATURE_DIR)

except ImportError:
    logger.warning(
        "PyMuPDF not installed. Falling back to pypdf. "
        "Install PyMuPDF for better table/column extraction: pip install PyMuPDF"
    )
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    loader = PyPDFDirectoryLoader(settings.MEDICAL_LITERATURE_DIR)
    documents = loader.load()

logger.info(f"Loaded {len(documents)} pages from all PDFs.")


# ============================================================
# 2. Add document-level metadata (auto-detection + known sources)
# ============================================================

logger.info("2. Adding source metadata...")

# Known source metadata for specific PDFs
SOURCE_METADATA = {
    "melanoma.pdf": {
        "source_type": "research_paper",
        "evidence_level": "research",
        "topic": "explainable_melanoma_ai"
    },
    "ABCD-ABCDE evidence.pdf": {
        "source_type": "evidence_review",
        "evidence_level": "evidence_review",
        "topic": "ABCD_ABCDE_clinical_rules"
    },
    "Dermoscopy + ABCD.pdf": {
        "source_type": "dermoscopy_review",
        "evidence_level": "clinical_review",
        "topic": "dermoscopy_ABCD"
    },
    "Guidelines-of-care-for-the-management-of-primary-c.pdf": {
        "source_type": "clinical_guideline",
        "evidence_level": "clinical_guideline",
        "topic": "primary_cutaneous_melanoma_management"
    }
}

# Auto-detection keywords for new PDFs
AUTO_DETECT_KEYWORDS = {
    "clinical_guideline": ["guideline", "recommendation", "consensus", "management", "care"],
    "evidence_review": ["evidence", "review", "systematic", "meta-analysis"],
    "dermoscopy_review": ["dermoscopy", "dermatoscopy", "dermoscopic"],
    "research_paper": ["study", "research", "method", "algorithm", "model"]
}


def detect_source_type(filename: str, text_sample: str = "") -> dict:
    """
    Auto-detect source type for unknown PDFs based on filename
    and first-page content.
    """
    name_lower = filename.lower()
    text_lower = text_sample.lower()
    combined = name_lower + " " + text_lower

    for source_type, keywords in AUTO_DETECT_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return {
                "source_type": source_type,
                "evidence_level": source_type,
                "topic": "auto_detected"
            }

    return {
        "source_type": "research_paper",
        "evidence_level": "research",
        "topic": "auto_detected"
    }


for document in documents:
    source_path = document.metadata.get("source", "")
    filename = os.path.basename(source_path)

    if filename in SOURCE_METADATA:
        metadata = SOURCE_METADATA[filename]
    else:
        # Auto-detect for new PDFs
        metadata = detect_source_type(
            filename,
            document.page_content[:500]
        )
        logger.info(
            f"  Auto-detected metadata for '{filename}': "
            f"{metadata['source_type']}"
        )

    document.metadata.update(metadata)

    if "document_name" not in document.metadata:
        document.metadata["document_name"] = filename

logger.info("Metadata added successfully.")


# ============================================================
# 3. Create parent-child chunk hierarchy
# ============================================================

logger.info("3. Creating parent-child chunk hierarchy...")

child_chunks, parent_store = create_parent_child_chunks(
    documents,
    parent_chunk_size=settings.PARENT_CHUNK_SIZE,
    parent_chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
    child_chunk_size=settings.CHILD_CHUNK_SIZE,
    child_chunk_overlap=settings.CHILD_CHUNK_OVERLAP
)

logger.info(
    f"Created {len(child_chunks)} child chunks and "
    f"{len(parent_store)} parent chunks."
)


# ============================================================
# 4. Show example chunk
# ============================================================

logger.info("4. Example child chunk metadata:")
logger.info(f"  {child_chunks[0].metadata}")
logger.info(f"  Text preview: {child_chunks[0].page_content[:100]}...")
logger.info(f"  Parent ID: {child_chunks[0].metadata.get('parent_id', 'N/A')}")


# ============================================================
# 5. Load embedding model (upgraded)
# ============================================================

logger.info(f"5. Loading embedding model: {settings.EMBEDDING_MODEL}")

embedding_model = FastEmbedEmbeddings(
    model_name=settings.EMBEDDING_MODEL
)

logger.info("Embedding model loaded.")


# ============================================================
# 6. Create Qdrant database (dense vectors)
# ============================================================

if settings.QDRANT_MODE == "cloud":
    # ── Qdrant Cloud ──
    logger.info("6. Creating Qdrant Cloud collection...")

    if not settings.QDRANT_CLOUD_URL or not settings.QDRANT_CLOUD_API_KEY:
        raise ValueError(
            "QDRANT_MODE=cloud but QDRANT_CLOUD_URL and/or "
            "QDRANT_CLOUD_API_KEY are not set in .env"
        )

    from qdrant_client import QdrantClient

    cloud_client = QdrantClient(
        url=settings.QDRANT_CLOUD_URL,
        api_key=settings.QDRANT_CLOUD_API_KEY
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=child_chunks,
        embedding=embedding_model,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        url=settings.QDRANT_CLOUD_URL,
        api_key=settings.QDRANT_CLOUD_API_KEY
    )

    logger.info(
        f"Stored {len(child_chunks)} child chunks in "
        f"Qdrant Cloud ({settings.QDRANT_CLOUD_URL})"
    )

else:
    # ── Qdrant Local ──
    logger.info("6. Creating local Qdrant database...")

    vector_store = QdrantVectorStore.from_documents(
        documents=child_chunks,
        embedding=embedding_model,
        path=settings.QDRANT_LOCAL_PATH,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        force_recreate=True
    )

    logger.info(
        f"Stored {len(child_chunks)} child chunks in "
        f"local Qdrant ({settings.QDRANT_LOCAL_PATH})"
    )


# ============================================================
# 7. Save parent chunk store
# ============================================================

logger.info("7. Saving parent chunk store...")

save_parent_store(
    parent_store,
    settings.PARENT_CHUNKS_PATH
)


# ============================================================
# 8. Build BM25 index (sparse keywords with medical stemming)
# ============================================================

logger.info("8. Building BM25 index with medical stemming...")

bm25 = build_bm25_index(child_chunks)

logger.info("BM25 index created successfully.")


# ============================================================
# Done
# ============================================================

logger.info("")
logger.info("=" * 60)
logger.info("SUCCESS — HYBRID INDEX CREATED (v4.0)")
logger.info("=" * 60)
logger.info(f"  Embedding model:  {settings.EMBEDDING_MODEL}")
logger.info(f"  Qdrant mode:      {settings.QDRANT_MODE}")
logger.info(f"  Child chunks:     {len(child_chunks)} (size={settings.CHILD_CHUNK_SIZE})")
logger.info(f"  Parent chunks:    {len(parent_store)} (size={settings.PARENT_CHUNK_SIZE})")
logger.info(f"  BM25 index:       {len(child_chunks)} documents (medical stemming)")
logger.info(f"  PDF parser:       PyMuPDF (layout-aware)")
logger.info("=" * 60)