"""FastAPI transport layer for the melanoma analysis workflow."""

import io
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np


import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
PROJECT_ROOT = Path(__file__).resolve().parent.parent

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from .auth import router as auth_router, get_current_user
from .admin import router as admin_router
from .schemas import (
    AnalysisResponse, AnalysisListItem, PhaseStatus,
    DiagnosisResultOut, DiagnosisInfoOut, FeatureScoreOut,
    SegmentationInfoOut, ExplainabilityInfoOut, PreprocessingInfoOut,
    PipelineInfoOut, MetadataInfoOut, MedicalEvidenceOut,
    ExplanationOut, ReportOut, PreprocessResponse, ScaleCalibrationOut,
    EvolutionOut, EvolutionChangesOut, EvolutionComparisonOut,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
REPORTS_DIR = OUTPUTS_DIR / "reports"
UPLOADS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}

app = FastAPI(
    title="Melanoma Supervisor API",
    version="2.0.0",
    description="AI-assisted CaseState-driven melanoma analysis workflow",
)

@app.on_event("startup")
async def startup_event():
    from .auth import seed_admin
    await seed_admin()

    # ── Per-user data isolation setup ─────────────────────────────────────
    from .db import get_db as _get_db
    _startup_db = _get_db()
    if _startup_db is not None:
        # Create compound index for efficient per-user history queries
        try:
            await _startup_db["analyses"].create_index(
                [("user_id", 1), ("diagnosis.metadata.timestamp", -1)],
                name="user_id_timestamp",
                background=True,
            )
        except Exception as _idx_err:
            logger.warning(f"Could not create analyses index: {_idx_err}")
        # Tag all pre-existing legacy records (no user_id) with a sentinel
        # value so they are NEVER returned to any authenticated user.
        try:
            _legacy = await _startup_db["analyses"].update_many(
                {"user_id": {"$exists": False}},
                {"$set": {"user_id": "__legacy__"}},
            )
            if _legacy.modified_count:
                logger.info(
                    f"Tagged {_legacy.modified_count} legacy analysis record(s) "
                    "with user_id='__legacy__' to prevent cross-user data leakage."
                )
        except Exception as _leg_err:
            logger.warning(f"Could not tag legacy analyses: {_leg_err}")
    # ── End per-user data isolation setup ─────────────────────────────────
cors_origins = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(auth_router)
app.include_router(admin_router)
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

analyses_store: Dict[str, AnalysisResponse] = {}
_supervisor_instance = None
_supervisor_init_lock = threading.Lock()
from .db import get_db, ping_db


def _config() -> dict:
    try:
        with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file) or {}
    except Exception:
        logger.exception("Configuration could not be loaded")
        return {}


def _resolve_project_path(value: Optional[str], default: Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _configured_model_paths() -> tuple[Path, Path]:
    """Return the checkpoints actually used by the diagnosis engine."""
    config = _config()
    paths = config.get("paths", {})
    checkpoints_dir = _resolve_project_path(
        paths.get("checkpoints"), PROJECT_ROOT / "checkpoints"
    )
    classification_path = _resolve_project_path(
        paths.get("classification_checkpoint"),
        PROJECT_ROOT / "checkpoints" / "best_swin_checkpoint.pth",
    )
    segmentation_path = _resolve_project_path(
        config.get("segmentation", {}).get("checkpoint"),
        checkpoints_dir / "best_segformer_melanoma.pth",
    )
    return classification_path, segmentation_path


def _models_available() -> bool:
    """Check the configured classification and segmentation checkpoints."""
    return all(path.is_file() for path in _configured_model_paths())


def _get_supervisor():
    global _supervisor_instance
    if _supervisor_instance is None:
        with _supervisor_init_lock:
            if _supervisor_instance is None:
                from src.agents.orchestrator import OrchestratorAgent
                _supervisor_instance = OrchestratorAgent(str(PROJECT_ROOT / "config.yaml"))
                logger.info("Supervisor Agent initialized")
    return _supervisor_instance


def _output_url(path: Optional[str], folder: str) -> Optional[str]:
    if not path:
        return None
    return f"/static/outputs/{folder}/{Path(path).name}"


def _workflow_to_response(state, original_image_url: Optional[str]) -> AnalysisResponse:
    phase_names = {
        "diagnosis": (1, "Diagnosis"),
        "medical_evidence": (2, "Medical Evidence"),
        "explanation": (3, "Explanation"),
        "report": (4, "Report Generation"),
    }
    # CaseState records decisions/actions instead of fixed phases. Preserve a
    # compact legacy projection for the existing UI clients.
    phases = []
    if hasattr(state, "phases"):
        for key, (number, name) in phase_names.items():
            phase = state.phases[key]
            phases.append(PhaseStatus(phase=number, name=name, status=phase.status,
                started_at=phase.started_at, completed_at=phase.completed_at, error=phase.error))
    else:
        actions = {event.action: event for event in state.tool_executions}
        for number, (action, name) in enumerate((("analyze_image", "Vision"), ("request_clinical_context", "Clinical Context"), ("retrieve_evidence", "Evidence"), ("generate_report", "Report")), 1):
            event = actions.get(action)
            phases.append(PhaseStatus(phase=number, name=name,
                status="completed" if event and event.result in {"completed", "updated", "sufficient", "follow_up_required"} else "skipped",
                error=None if not event or event.result != "failed" else "action failed"))

    diagnosis_out = None
    if state.diagnosis_result is not None:
        diag = state.diagnosis_result
        features = {
            name: FeatureScoreOut(
                name=feat.name,
                score_numeric=feat.score_numeric,
                score_label=feat.score_label,
            )
            for name, feat in diag.clinical_features.items()
        }
        diagnosis_out = DiagnosisResultOut(
            diagnosis=DiagnosisInfoOut(
                prediction=diag.diagnosis.prediction,
                confidence=diag.diagnosis.confidence,
                fusion_score=state.classification.get("fusion_score") if hasattr(state, "classification") else None,
                fusion_weights=state.classification.get("fusion_weights") if hasattr(state, "classification") else None,
            ),
            probabilities=diag.probabilities,
            melanoma_probability=diag.probabilities.get("melanoma"),
            non_melanoma_probability=diag.probabilities.get(
                "non_melanoma", diag.probabilities.get("not_melanoma")
            ),
            classification_threshold=diag.classification_threshold,
            measurements=diag.measurements,
            clinical_features=features,
            clinical_interpretations=diag.clinical_interpretations,
            segmentation=SegmentationInfoOut(
                status=diag.segmentation.status,
                mask_url=_output_url(diag.segmentation.mask_path, "masks"),
                overlay_url=_output_url(diag.segmentation.overlay_path, "masks"),
                area_px=diag.segmentation.area_px,
                perimeter_px=diag.segmentation.perimeter_px,
            ),
            explainability=ExplainabilityInfoOut(
                attention_inside_lesion=diag.explainability.attention_inside_lesion,
                attention_outside_lesion=diag.explainability.attention_outside_lesion,
                centroid_distance=diag.explainability.centroid_distance,
                mask_cam_iou=diag.explainability.mask_cam_iou,
            ),
            preprocessing=PreprocessingInfoOut(
                hair_removal=diag.preprocessing.hair_removal,
                illumination_normalization=diag.preprocessing.illumination_normalization,
            ),
            pipeline=PipelineInfoOut(
                classification_model=diag.pipeline.classification_model,
                segmentation_model=diag.pipeline.segmentation_model,
                feature_extractor=diag.pipeline.feature_extractor,
                pipeline_version=diag.pipeline.pipeline_version,
                classification_threshold=(
                    diag.pipeline.classification_threshold
                    if diag.pipeline.classification_threshold is not None
                    else diag.classification_threshold
                ),
            ),
            metadata=MetadataInfoOut(
                timestamp=diag.metadata.timestamp,
                image_id=diag.metadata.request_id or state.analysis_id,
            ),
            processing_status=diag.processing_status,
            warnings=diag.warnings,
        )

    # Scale calibration
    scale_cal_out = None
    if diagnosis_out and state.diagnosis_result and state.diagnosis_result.scale_calibration:
        sc = state.diagnosis_result.scale_calibration
        scale_cal_out = ScaleCalibrationOut(
            pixels_per_mm=sc.pixels_per_mm,
            method=sc.method,
            confidence=sc.confidence,
            reference_type=sc.reference_type,
            reference_diameter_px=sc.reference_diameter_px,
            reference_bbox_px=sc.reference_bbox_px,
            reference_diameter_mm=sc.reference_diameter_mm,
            detected=sc.detected,
            calibration_confidence=sc.calibration_confidence,
            calibration_valid=sc.calibration_valid,
            calibration_method=sc.calibration_method,
            reference_object_type=sc.reference_object_type,
            reference_length_px=sc.reference_length_px,
            reference_length_mm=sc.reference_length_mm,
            calibration_reason=sc.calibration_reason,
            scale_region=sc.scale_region,
            orientation=sc.orientation,
            angle_degrees=sc.angle_degrees,
            interval_mm=sc.interval_mm,
            tick_positions_px=sc.tick_positions_px,
            tick_spacing_px=sc.tick_spacing_px,
            validated_tick_count=sc.validated_tick_count,
            reference_points_px=sc.reference_points_px,
            reprojection_error_px=sc.reprojection_error_px,
            calibration_uncertainty=sc.calibration_uncertainty,
            warnings=sc.warnings,
            homography=sc.homography,
            axis_endpoints_px=sc.axis_endpoints_px,
            tick_points_px=sc.tick_points_px,
            interval_residuals_px=sc.interval_residuals_px,
        )
        diagnosis_out.scale_calibration = scale_cal_out

    evidence = None
    if state.retrieved_evidence is not None:
        evidence = [MedicalEvidenceOut(
            source=item.source,
            title=item.title,
            relevance_score=item.relevance_score,
            snippet=item.content[:300],
        ) for item in state.retrieved_evidence]

    explanation = None
    if state.explanation_result is not None:
        result = state.explanation_result
        if isinstance(result, dict):
            # Legacy template-mode or fallback dict — map generously
            explanation = ExplanationOut(
                summary=result.get("summary", result.get("model_observation", "Explanation unavailable.")),
                reasoning=[s for s in [
                    result.get("literature_context", ""),
                    result.get("model_observation", ""),
                ] if s],
                confidence_assessment="requires-review" if result.get("uncertainty") else "moderate",
                limitations=result.get("uncertainty", []),
                evidence_citations=result.get("evidence_citations", []),
                grad_cam_detail=result.get("grad_cam_detail"),
                flags=result.get("flags", []),
            )
        else:
            # ExplanationResult dataclass from EvidenceGroundedExplanationService
            grad_cam_url = _output_url(
                getattr(state, "grad_cam_saved_path", None), "gradcam_samples"
            )
            # Also check explanation object for grad_cam_url attribute
            if not grad_cam_url and hasattr(result, "grad_cam_url"):
                grad_cam_url = result.grad_cam_url
            explanation = ExplanationOut(
                summary=result.summary,
                reasoning=result.reasoning,
                grad_cam_url=grad_cam_url,
                confidence_assessment=result.confidence_assessment,
                next_steps=result.next_steps,
                limitations=result.limitations,
                evidence_citations=[
                    dict(c) if isinstance(c, dict) else vars(c) if hasattr(c, '__dict__') else c
                    for c in (result.evidence_citations or [])
                ],
                grad_cam_detail=getattr(result, "grad_cam_detail", None),
                flags=getattr(result, "flags", []),
            )

    report = ReportOut()
    if state.report_result is not None and state.report_result.pdf_path:
        report.pdf_url = _output_url(state.report_result.pdf_path, "reports")

    evolution_assessment = None
    if "evolution" in state.clinical_context and isinstance(state.clinical_context["evolution"], dict):
        ev_dict = state.clinical_context["evolution"]
        if "status" in ev_dict:
            evolution_assessment = EvolutionOut(
                status=ev_dict.get("status", "unable_to_assess"),
                confidence=ev_dict.get("confidence", 0.0),
                observations=ev_dict.get("observations", []),
                changes=EvolutionChangesOut(**ev_dict.get("changes", {})),
                comparison=EvolutionComparisonOut(**ev_dict.get("comparison", {})),
                limitations=ev_dict.get("limitations", []),
            )

    return AnalysisResponse(
        analysis_id=state.analysis_id,
        status=state.overall_status,
        phases=phases,
        diagnosis=diagnosis_out,
        evidence=evidence,
        explanation=explanation,
        report=report,
        evolution_assessment=evolution_assessment,
        original_image_url=original_image_url,
        flags=state.flags,
        error_code=state.error_code,
        message=state.user_message,
        retryable=state.retryable,
        execution_trace=state.public_trace() if hasattr(state, "public_trace") else [],
        clinical_questions=getattr(state, "pending_clinical_questions", []),
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed for %s", request.url.path)
    return JSONResponse(status_code=422, content={
        "error_code": "invalid_request",
        "message": "Please provide a supported image file.",
    })


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error for %s", request.url.path)
    return JSONResponse(status_code=500, content={
        "error_code": "internal_error",
        "message": "The request could not be completed. Please try again later.",
    })


@app.get("/")
def root():
    return {"message": "Melanoma Supervisor API", "version": "2.0.0"}


@app.get("/api/health")
async def health():
    db = get_db()
    mongo_status = await ping_db() if db is not None else False
    count = len(analyses_store)
    if mongo_status:
        try:
            count = await db["analyses"].count_documents({})
        except Exception:
            pass
    return {
        "status": "healthy",
        "models_loaded": _models_available(),
        "mongodb_connected": mongo_status,
        "analyses_count": count,
    }


@app.get("/api/analyses", response_model=list[AnalysisListItem])
async def list_analyses(current_user: dict = Depends(get_current_user)):
    db = get_db()
    items = []
    user_id = str(current_user.get("id") or current_user.get("_id", ""))
    if db is not None:
        try:
            query = {"user_id": user_id}
            cursor = db["analyses"].find(query).sort("_id", -1)
            async for doc in cursor:
                try:
                    diagnosis = doc.get("diagnosis") or {}
                    if not isinstance(diagnosis, dict):
                        diagnosis = {}

                    diag_meta = diagnosis.get("metadata") or {}
                    if not isinstance(diag_meta, dict):
                        diag_meta = {}

                    diag_info = diagnosis.get("diagnosis") or {}
                    if not isinstance(diag_info, dict):
                        diag_info = {}

                    status = str(doc.get("status") or "completed")

                    orig_url = doc.get("original_image_url") or ""
                    image_name = (
                        diag_meta.get("image_id")
                        or (Path(orig_url).name if orig_url else None)
                        or doc.get("analysis_id", "Unknown")
                    )

                    if diag_info.get("prediction"):
                        prediction = str(diag_info["prediction"])
                    elif status in ("failed", "error"):
                        prediction = "Analysis Failed"
                    elif status == "image_quality_insufficient":
                        prediction = "Quality Insufficient"
                    elif status == "segmentation_failed":
                        prediction = "Segmentation Failed"
                    else:
                        prediction = status.replace("_", " ").title() if status else "Unknown"

                    raw_conf = diag_info.get("confidence", 0.0)
                    try:
                        confidence = float(raw_conf) if raw_conf is not None else 0.0
                    except (ValueError, TypeError):
                        confidence = 0.0

                    raw_ts = diag_meta.get("timestamp")
                    if not raw_ts and doc.get("execution_trace"):
                        traces = doc.get("execution_trace")
                        if isinstance(traces, list) and traces and isinstance(traces[0], dict):
                            raw_ts = traces[0].get("timestamp")

                    ts = None
                    if isinstance(raw_ts, str):
                        try:
                            ts = datetime.fromisoformat(raw_ts)
                        except Exception:
                            pass
                    elif isinstance(raw_ts, datetime):
                        ts = raw_ts

                    if ts is None:
                        if "_id" in doc and hasattr(doc["_id"], "generation_time"):
                            ts = doc["_id"].generation_time
                        else:
                            ts = datetime.now(timezone.utc)

                    items.append(AnalysisListItem(
                        analysis_id=doc["analysis_id"],
                        image_name=image_name,
                        prediction=prediction,
                        confidence=confidence,
                        timestamp=ts,
                        status=status,
                    ))
                except Exception as e:
                    logger.warning(f"Skipping document {doc.get('analysis_id')} due to parse error: {e}")
            return items
        except Exception as e:
            logger.error(f"MongoDB error in list_analyses: {e}")
            # Fall back to in-memory store

    for analysis in analyses_store.values():
        if getattr(analysis, "user_id", None) == user_id:
            diag = analysis.diagnosis
            diag_meta = getattr(diag, "metadata", None) if diag else None
            diag_info = getattr(diag, "diagnosis", None) if diag else None
            status = getattr(analysis, "status", "completed") or "completed"

            orig_url = getattr(analysis, "original_image_url", "") or ""
            image_name = (
                (getattr(diag_meta, "image_id", None) if diag_meta else None)
                or (Path(orig_url).name if orig_url else None)
                or analysis.analysis_id
            )

            if diag_info and getattr(diag_info, "prediction", None):
                prediction = str(diag_info.prediction)
            elif status in ("failed", "error"):
                prediction = "Analysis Failed"
            elif status == "image_quality_insufficient":
                prediction = "Quality Insufficient"
            elif status == "segmentation_failed":
                prediction = "Segmentation Failed"
            else:
                prediction = status.replace("_", " ").title() if status else "Unknown"

            raw_conf = getattr(diag_info, "confidence", 0.0) if diag_info else 0.0
            try:
                confidence = float(raw_conf) if raw_conf is not None else 0.0
            except (ValueError, TypeError):
                confidence = 0.0

            ts = getattr(diag_meta, "timestamp", None) if diag_meta else None
            if not ts:
                ts = datetime.now(timezone.utc)

            items.append(AnalysisListItem(
                analysis_id=analysis.analysis_id,
                image_name=image_name,
                prediction=prediction,
                confidence=confidence,
                timestamp=ts,
                status=status,
            ))
    return sorted(items, key=lambda item: item.timestamp, reverse=True)


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return a single analysis. Only the owning user may access it."""
    user_id = str(current_user.get("id") or current_user.get("_id", ""))
    db = get_db()
    if db is not None:
        try:
            doc = await db["analyses"].find_one({"analysis_id": analysis_id})
            if doc:
                # Enforce ownership: if the document has a user_id it MUST match.
                doc_owner = doc.get("user_id", "")
                if doc_owner and doc_owner != user_id:
                    raise HTTPException(status_code=404, detail={
                        "error_code": "analysis_not_found",
                        "message": "Analysis not found.",
                    })
                return AnalysisResponse(**doc)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"MongoDB error in get_analysis: {e}")

    # In-memory fallback — also check ownership
    analysis = analyses_store.get(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail={
            "error_code": "analysis_not_found",
            "message": "Analysis not found.",
        })
    # If the in-memory record was saved with a user_id, verify it
    stored_owner = getattr(analysis, 'user_id', None)
    if stored_owner and stored_owner != user_id:
        raise HTTPException(status_code=404, detail={
            "error_code": "analysis_not_found",
            "message": "Analysis not found.",
        })
    return analysis


@app.post("/api/preprocess", response_model=PreprocessResponse)
async def preprocess_image(file: UploadFile = File(...)):
    """Run hair-removal preprocessing and return a before/after comparison."""
    preview_id = str(uuid.uuid4())[:8]
    extension = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_MIME_TYPES or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail={
            "error_code": "unsupported_format",
            "message": "Unsupported image format.",
        })

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={
            "error_code": "file_too_large",
            "message": "Image file is too large.",
        })

    try:
        with Image.open(io.BytesIO(contents)) as candidate:
            candidate.verify()
        with Image.open(io.BytesIO(contents)) as candidate:
            image = candidate.convert("RGB")
            image_array = np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail={
            "error_code": "image_decode_failed",
            "message": "The selected file is not a readable image.",
        })

    # Save original for comparison
    orig_path = UPLOADS_DIR / f"{preview_id}_original.png"
    Image.fromarray(image_array).save(orig_path, format="PNG")

    # Run hair removal only
    from src.data.preprocessing import DermoscopyPreprocessor
    preprocessor = DermoscopyPreprocessor(
        hair_removal=True, illumination_normalization=False,
    )
    preprocessed = await run_in_threadpool(preprocessor.process, image_array)

    # Detect hair pixels by comparing original and preprocessed
    import cv2
    gray_orig = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray_orig, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_pixel_count = int((hair_mask > 0).sum())
    hair_detected = hair_pixel_count > 500  # meaningful threshold

    # Save preprocessed image
    preproc_path = UPLOADS_DIR / f"{preview_id}_preprocessed.png"
    Image.fromarray(preprocessed).save(preproc_path, format="PNG")

    return PreprocessResponse(
        preprocessed_url=f"/static/uploads/{preproc_path.name}",
        hair_detected=hair_detected,
        hair_pixel_count=hair_pixel_count,
        original_url=f"/static/uploads/{orig_path.name}",
    )


async def _save_analysis(analysis: AnalysisResponse, user_id: str = ""):
    analyses_store[analysis.analysis_id] = analysis
    db = get_db()
    if db is not None:
        try:
            doc = analysis.model_dump(mode='json')
            if user_id:
                doc["user_id"] = user_id
            await db["analyses"].update_one(
                {"analysis_id": analysis.analysis_id},
                {"$set": doc},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save analysis {analysis.analysis_id} to MongoDB: {e}")

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    historical_files: list[UploadFile] = File(default=[]),
    scale_method: str = Form("auto"),
    scale_reference_mm: Optional[float] = Form(None),
    scale_reference_key: Optional[str] = Form(None),
    clinical_context_json: Optional[str] = Form(None),
    historical_analysis_ids: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """Run the canonical Supervisor Agent workflow for one image/case.
    Requires a valid JWT — the authenticated user's ID is attached to the analysis.
    """
    _caller_user_id = str(current_user.get("id") or current_user.get("_id", ""))
    analysis_id = str(uuid.uuid4())[:8]
    extension = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_MIME_TYPES or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail={
            "error_code": "unsupported_format",
            "message": "Unsupported image format. Please upload a JPEG, JPG, PNG, or WEBP image.",
        })

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={
            "error_code": "file_too_large",
            "message": "Image file is too large. Please upload an image no larger than 10 MB.",
        })

    file_path = UPLOADS_DIR / f"{analysis_id}.png"
    try:
        with Image.open(io.BytesIO(contents)) as candidate:
            candidate.verify()
        with Image.open(io.BytesIO(contents)) as candidate:
            image = candidate.convert("RGB")
            image.save(file_path, format="PNG", optimize=True)
    except (UnidentifiedImageError, OSError, ValueError):
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail={
            "error_code": "image_decode_failed",
            "message": "The selected file is not a readable image. Please choose another image.",
        })

    historical_image_paths = []
    for idx, hfile in enumerate(historical_files or []):
        if not hfile.filename: continue
        hfile_path = UPLOADS_DIR / f"{analysis_id}_hist_{idx}.png"
        try:
            hcontents = await hfile.read(MAX_UPLOAD_BYTES + 1)
            with Image.open(io.BytesIO(hcontents)) as hcand:
                hcand.verify()
            with Image.open(io.BytesIO(hcontents)) as hcand:
                himg = hcand.convert("RGB")
                himg.save(hfile_path, format="PNG", optimize=True)
            historical_image_paths.append(str(hfile_path))
        except Exception:
            logger.warning(f"Failed to process historical image {idx}")

    if historical_analysis_ids:
        try:
            hist_ids = json.loads(historical_analysis_ids)
            if isinstance(hist_ids, list):
                for hist_id in hist_ids:
                    hist_path = UPLOADS_DIR / f"{hist_id}.png"
                    if hist_path.exists():
                        historical_image_paths.append(str(hist_path))
        except Exception as e:
            logger.warning(f"Failed to parse historical_analysis_ids: {e}")


    if not _models_available():
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail={
            "error_code": "models_unavailable",
            "message": "The analysis service is temporarily unavailable. Please try again later.",
        })

    original_image_url = f"/static/uploads/{file_path.name}"
    try:
        clinical_context = json.loads(clinical_context_json) if clinical_context_json else None
        if clinical_context is not None and not isinstance(clinical_context, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=422, detail="clinical_context_json must be a JSON object")
    try:
        state = await run_in_threadpool(
            _get_supervisor().run,
            str(file_path),
            analysis_id=analysis_id,
            save_mask=False,
            scale_method=scale_method,
            scale_reference_mm=scale_reference_mm,
            scale_reference_key=scale_reference_key,
            clinical_context=clinical_context,
            historical_images=historical_image_paths,
        )
        analysis = _workflow_to_response(state, original_image_url)
    except Exception:
        logger.exception("Orchestrator pipeline failed for request %s", analysis_id)
        now = datetime.now()
        analysis = AnalysisResponse(
            analysis_id=analysis_id,
            status="failed",
            phases=[
                PhaseStatus(phase=1, name="Diagnosis", status="failed", started_at=now, completed_at=now, error="Analysis could not be completed."),
                PhaseStatus(phase=2, name="Medical Evidence", status="skipped"),
                PhaseStatus(phase=3, name="Explanation", status="skipped"),
                PhaseStatus(phase=4, name="Report Generation", status="skipped"),
            ],
            original_image_url=original_image_url,
            error_code="pipeline_failed",
            message="Analysis could not be completed. Please try again with a clearer image.",
            retryable=True,
        )

    await _save_analysis(analysis, user_id=_caller_user_id)
    return analysis


