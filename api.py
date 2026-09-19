"""
Melanoma Agentic RAG API — v4.0 (Production-Grade)

Upgrades from v3.0:
- API key authentication middleware (X-API-Key header)
- Rate limiting via slowapi (configurable per endpoint)
- Async execution: LLM/pipeline calls offloaded to thread pool
- Per-request agent isolation (thread-safe, no race conditions)
- SSE streaming endpoints for real-time reasoning trace
- Metrics endpoint (/metrics) for performance monitoring
- Fixed CORS: no wildcard when credentials are enabled
- Pydantic v2 schemas with strict validation

Retained from v3.0:
- File upload security (size, MIME, magic-byte, path traversal prevention)
- Prompt injection defense
- Guaranteed temp file cleanup
"""

import os
import uuid
import asyncio
import logging
from io import BytesIO
from typing import Dict, Any, Optional, List
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from config import settings
from agent import create_agent, sanitize_user_input
from model_pipeline.pipeline import MelanomaPipeline
from observability import global_metrics


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Constants & Security Constraints
# ============================================================

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/x-ms-bmp"
}


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Melanoma Agentic RAG API (ABCDE)",
    description=(
        "Evidence-grounded, explainable melanoma clinical "
        "decision-support API based on the ABCDE framework. "
        "Production-grade with auth, rate limiting, and streaming."
    ),
    version="4.0"
)

# CORS Middleware — explicit origins (no wildcard with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# Rate Limiting
# ============================================================

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    rate_limit_available = True
    logger.info(f"Rate limiting enabled: {settings.RATE_LIMIT}")

except ImportError:
    rate_limit_available = False
    logger.warning("slowapi not installed. Rate limiting disabled.")

    # Create a no-op limiter decorator
    class _NoOpLimiter:
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    limiter = _NoOpLimiter()


# ============================================================
# Authentication Middleware
# ============================================================

async def verify_api_key(request: Request):
    """
    Verify the X-API-Key header.
    If API_AUTH_KEY is empty in .env, authentication is disabled.
    """
    if not settings.API_AUTH_KEY:
        return  # Auth disabled

    api_key = request.headers.get("X-API-Key", "")
    if api_key != settings.API_AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-API-Key header."
        )


# ============================================================
# Initialize Pipeline (shared, read-only)
# ============================================================

pipeline = MelanomaPipeline()


# ============================================================
# Pydantic Schemas with Strict Validation
# ============================================================

class EvolutionInput(BaseModel):
    reported_change: bool = Field(
        default=False,
        description="Whether the lesion has changed over time"
    )
    change_types: Optional[List[str]] = Field(
        default=[],
        description="List of changes (e.g. enlargement, darkening, elevation)"
    )
    timeframe_months: Optional[float] = Field(
        default=None, ge=0, le=120,
        description="Timeframe of change in months"
    )
    symptoms: Optional[List[str]] = Field(
        default=[],
        description="Symptoms like itching, bleeding, or pain"
    )
    notes: Optional[str] = Field(
        default="", max_length=500,
        description="Clinician or patient notes regarding evolution"
    )


class AbcdeMetricsInput(BaseModel):
    asymmetry_index: Optional[float] = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Asymmetry percentage (0-100)"
    )
    border_irregularity_score: Optional[float] = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Border irregularity (0-1)"
    )
    color_variation_score: Optional[float] = Field(
        default=0.0, ge=0.0, le=255.0,
        description="Color variation score"
    )
    diameter_pixels: Optional[float] = Field(
        default=0.0, ge=0.0,
        description="Diameter in pixels"
    )
    diameter_mm: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Diameter in millimeters (if calibrated)"
    )
    evolution: Optional[EvolutionInput] = Field(
        default_factory=EvolutionInput,
        description="Evolution criterion metadata"
    )


class CaseInput(BaseModel):
    case_id: str = Field(
        ..., min_length=1, max_length=64,
        pattern=r"^[a-zA-Z0-9_\-\.]+$"
    )
    prediction: str = Field(..., min_length=1, max_length=64)
    confidence: float = Field(..., ge=0.0, le=1.0)
    abcde_metrics: Optional[AbcdeMetricsInput] = None
    abcd_metrics: Optional[Dict[str, Any]] = None  # Backward compatibility

    @field_validator("case_id", "prediction", mode="before")
    def sanitize_strings(cls, v):
        return sanitize_user_input(v, max_length=64)


# ============================================================
# Health Probe Endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "melanoma-agentic-rag",
        "framework": "ABCDE",
        "version": "4.0",
        "features": {
            "hyde_queries": settings.ENABLE_HYDE,
            "qdrant_mode": settings.QDRANT_MODE,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "authentication": bool(settings.API_AUTH_KEY),
            "rate_limiting": rate_limit_available,
            "streaming": True
        }
    }


# ============================================================
# Home Endpoint
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Melanoma Agentic RAG API (ABCDE v4.0) is running",
        "version": "4.0",
        "endpoints": [
            "/health",
            "/analyze-case",
            "/analyze-image",
            "/analyze-case/stream",
            "/metrics",
            "/docs"
        ]
    }


# ============================================================
# Metrics Endpoint
# ============================================================

@app.get("/metrics", dependencies=[Depends(verify_api_key)])
def get_metrics():
    """
    Returns aggregated performance metrics across recent requests.
    Includes latency percentiles, token usage, and error rates.
    """
    return global_metrics.get_stats()


# ============================================================
# Analyze Case Endpoint (JSON — Async)
# ============================================================

@app.post("/analyze-case", dependencies=[Depends(verify_api_key)])
@limiter.limit(settings.RATE_LIMIT)
async def analyze_case(case: CaseInput, request: Request):
    try:
        logger.info(f"Analyzing case: {case.case_id}")
        case_dict = case.model_dump()

        # Handle backward compatibility with abcd_metrics
        if case_dict.get("abcde_metrics") is None and case_dict.get("abcd_metrics") is not None:
            case_dict["abcde_metrics"] = case_dict["abcd_metrics"]

        # Run in thread pool to avoid blocking the event loop
        def _generate():
            agent = create_agent()
            return agent.generate_report(case_dict)

        report = await asyncio.to_thread(_generate)
        return report

    except Exception as e:
        logger.error(f"Error analyzing case {case.case_id}: {str(e)}", exc_info=True)
        global_metrics.record_error()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the clinical case. Please try again."
        )


# ============================================================
# Analyze Case with SSE Streaming
# ============================================================

@app.post("/analyze-case/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit(settings.RATE_LIMIT)
async def analyze_case_stream(case: CaseInput, request: Request):
    """
    Same as /analyze-case but returns Server-Sent Events (SSE)
    streaming the reasoning trace in real time.
    """
    import json as _json

    case_dict = case.model_dump()

    if case_dict.get("abcde_metrics") is None and case_dict.get("abcd_metrics") is not None:
        case_dict["abcde_metrics"] = case_dict["abcd_metrics"]

    async def event_generator():
        try:
            # Run the agent in a thread
            def _generate():
                agent = create_agent()
                return agent.generate_report(case_dict)

            # Emit start event
            yield {
                "event": "started",
                "data": _json.dumps({
                    "case_id": case_dict.get("case_id", ""),
                    "message": "Analysis started"
                })
            }

            report = await asyncio.to_thread(_generate)

            # Emit reasoning trace steps
            trace = report.get("reasoning_trace", {})
            for step in trace.get("trace", []):
                yield {
                    "event": step.get("step", "trace"),
                    "data": _json.dumps(step.get("detail", {}), default=str)
                }

            # Emit performance metrics
            perf = report.get("performance_metrics", {})
            yield {
                "event": "performance",
                "data": _json.dumps(perf, default=str)
            }

            # Emit final report
            yield {
                "event": "report",
                "data": _json.dumps(report, default=str)
            }

            # Signal completion
            yield {
                "event": "done",
                "data": _json.dumps({"status": "complete"})
            }

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            global_metrics.record_error()
            yield {
                "event": "error",
                "data": _json.dumps({"error": str(e)})
            }

    try:
        from sse_starlette.sse import EventSourceResponse
        return EventSourceResponse(event_generator())
    except ImportError:
        # Fallback: run synchronously if sse-starlette not installed
        logger.warning("sse-starlette not installed. Running synchronously.")
        def _generate():
            agent = create_agent()
            return agent.generate_report(case_dict)
        report = await asyncio.to_thread(_generate)
        return report


# ============================================================
# Analyze Image Endpoint (File Upload — Async)
# ============================================================

@app.post("/analyze-image", dependencies=[Depends(verify_api_key)])
@limiter.limit(settings.RATE_LIMIT)
async def analyze_image(
    file: UploadFile = File(...),
    request: Request = None
):
    temp_path = None
    try:
        # 1. Validate file extension
        filename = sanitize_user_input(file.filename or "upload.jpg", max_length=128)
        _, ext = os.path.splitext(filename.lower())
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # 2. Validate MIME type
        content_type = (file.content_type or "").lower()
        if content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported media type '{content_type}'. Must be a valid image."
            )

        # 3. Read content & validate file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
            )
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        # 4. Verify image integrity with PIL
        try:
            img = Image.open(BytesIO(content))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is corrupt or not a valid image format."
            )

        # 5. Save to isolated temporary file
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        secure_filename = f"{uuid.uuid4().hex}_{ext.lstrip('.')}"
        temp_path = os.path.join(temp_dir, secure_filename)

        with open(temp_path, "wb") as f:
            f.write(content)

        # 6. Run pipeline in thread pool
        _temp_path = temp_path

        def _analyze():
            image_result = pipeline.analyze(_temp_path)
            prediction = image_result["prediction"]
            metrics = image_result["abcd_metrics"]

            case_data = {
                "case_id": f"IMG-{uuid.uuid4().hex[:8].upper()}",
                "prediction": prediction["label"],
                "confidence": float(prediction["confidence"]),
                "abcde_metrics": {
                    "asymmetry_index": metrics["asymmetry_index"],
                    "border_irregularity_score": metrics["border_irregularity_score"],
                    "color_variation_score": metrics["color_variation_score"],
                    "diameter_pixels": metrics["diameter_pixels"],
                    "evolution": {
                        "reported_change": False,
                        "status": "single_timepoint_capture",
                        "notes": "Single dermoscopic capture. Longitudinal history should be evaluated by clinician."
                    }
                }
            }

            agent = create_agent()
            report = agent.generate_report(case_data)

            return {
                "case_id": case_data["case_id"],
                "image_prediction": image_result["prediction"],
                "abcde_metrics": case_data["abcde_metrics"],
                "rag_report": report
            }

        result = await asyncio.to_thread(_analyze)
        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error processing image: {str(e)}", exc_info=True)
        global_metrics.record_error()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during image processing and evidence retrieval."
        )

    finally:
        # Guaranteed cleanup of temporary files
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_path}: {e}")