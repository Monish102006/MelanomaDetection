"""
API Schemas
============

Pydantic models for API request/response serialization.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models (mirror of engine.diagnosis but API-facing)
# ---------------------------------------------------------------------------

class DiagnosisInfoOut(BaseModel):
    prediction: str
    confidence: float
    fusion_score: Optional[float] = None
    fusion_weights: Optional[Dict[str, float]] = None


class FeatureScoreOut(BaseModel):
    name: str
    score_numeric: float
    score_label: str


class SegmentationInfoOut(BaseModel):
    status: str = "completed"
    mask_url: Optional[str] = None
    overlay_url: Optional[str] = None
    area_px: Optional[int] = None
    perimeter_px: Optional[float] = None


class ExplainabilityInfoOut(BaseModel):
    attention_inside_lesion: Optional[float] = None
    attention_outside_lesion: Optional[float] = None
    centroid_distance: Optional[float] = None
    mask_cam_iou: Optional[float] = None


class PreprocessingInfoOut(BaseModel):
    hair_removal: bool = False
    illumination_normalization: bool = False


class ScaleCalibrationOut(BaseModel):
    """Reference-object calibration data."""
    pixels_per_mm: Optional[float] = None
    method: str = "none"
    confidence: float = 0.0
    reference_type: str = "none"
    reference_diameter_px: Optional[float] = None
    reference_bbox_px: Optional[tuple[int, int, int, int]] = None
    reference_diameter_mm: Optional[float] = None
    detected: bool = False
    calibration_confidence: float = 0.0
    calibration_valid: bool = False
    calibration_method: str = "none"
    reference_object_type: str = "none"
    reference_length_px: Optional[float] = None
    reference_length_mm: Optional[float] = None
    calibration_reason: str = "reference calibration not available"
    scale_region: Optional[Dict[str, int]] = None
    orientation: str = "unknown"
    angle_degrees: Optional[float] = None
    interval_mm: Optional[float] = None
    tick_positions_px: Optional[list[float]] = None
    tick_spacing_px: Optional[float] = None
    validated_tick_count: int = 0
    reference_points_px: Optional[list[tuple[float, float]]] = None
    reprojection_error_px: Optional[float] = None
    calibration_uncertainty: Optional[float] = None
    warnings: list[str] = []
    homography: Optional[list[list[float]]] = None
    axis_endpoints_px: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    tick_points_px: Optional[list[tuple[float, float]]] = None
    interval_residuals_px: Optional[list[float]] = None


class PipelineInfoOut(BaseModel):
    classification_model: str = "SwinV2 Base"
    segmentation_model: str = "SegFormer"
    feature_extractor: str = "OpenCV + scikit-image"
    pipeline_version: str = "1.0"
    classification_threshold: Optional[float] = None


class MetadataInfoOut(BaseModel):
    timestamp: datetime
    image_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Phase outputs
# ---------------------------------------------------------------------------

class PhaseStatus(BaseModel):
    """Status of a single pipeline phase."""
    phase: int
    name: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class DiagnosisResultOut(BaseModel):
    """Phase 1 output: full diagnosis result."""
    diagnosis: DiagnosisInfoOut
    probabilities: Dict[str, float]
    melanoma_probability: Optional[float] = None
    non_melanoma_probability: Optional[float] = None
    classification_threshold: Optional[float] = None
    measurements: Dict[str, Dict[str, Any]]
    clinical_features: Dict[str, FeatureScoreOut]
    clinical_interpretations: Dict[str, str] = Field(default_factory=dict)
    segmentation: SegmentationInfoOut
    explainability: ExplainabilityInfoOut
    preprocessing: PreprocessingInfoOut
    scale_calibration: Optional[ScaleCalibrationOut] = None
    pipeline: PipelineInfoOut
    metadata: MetadataInfoOut
    processing_status: str = "completed"
    warnings: List[str] = Field(default_factory=list)


class MedicalEvidenceOut(BaseModel):
    """Phase 2 output: retrieved medical evidence."""
    source: str
    title: str
    relevance_score: float
    snippet: str


class ExplanationOut(BaseModel):
    """Phase 3 output: AI-generated explanation."""
    summary: str
    reasoning: List[str]
    grad_cam_url: Optional[str] = None
    confidence_assessment: str
    next_steps: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
    evidence_citations: List[Dict[str, Any]] = Field(default_factory=list)
    grad_cam_detail: Optional[str] = None
    flags: List[str] = Field(default_factory=list)


class ReportOut(BaseModel):
    """Phase 4 output: generated report."""
    pdf_url: Optional[str] = None
    report_html: Optional[str] = None




# ---------------------------------------------------------------------------
# Evolution Assessment schemas
# ---------------------------------------------------------------------------

class EvolutionChangesOut(BaseModel):
    size: str = "Not assessed"
    shape: str = "Not assessed"
    color: str = "Not assessed"
    structure: str = "Not assessed"
    other: str = "Not assessed"


class EvolutionComparisonOut(BaseModel):
    images_compared: int
    time_interval: str
    same_lesion_confidence: float


class EvolutionOut(BaseModel):
    status: str
    confidence: float
    observations: List[str] = Field(default_factory=list)
    changes: EvolutionChangesOut = Field(default_factory=EvolutionChangesOut)
    comparison: EvolutionComparisonOut
    limitations: List[str] = Field(default_factory=list)


class EvolutionRequest(BaseModel):
    analysis_ids: List[str]
    timestamps: Optional[List[Optional[str]]] = None

# ---------------------------------------------------------------------------
# Full pipeline response
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """Complete pipeline response combining all phases."""
    analysis_id: str
    status: str  # running | completed | failed | image_quality_insufficient | segmentation_failed
    phases: List[PhaseStatus]
    diagnosis: Optional[DiagnosisResultOut] = None
    evidence: Optional[List[MedicalEvidenceOut]] = None
    explanation: Optional[ExplanationOut] = None
    report: Optional[ReportOut] = None
    evolution_assessment: Optional[EvolutionOut] = None
    original_image_url: Optional[str] = None
    flags: List[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    message: Optional[str] = None
    retryable: bool = False
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    clinical_questions: List[Dict[str, str]] = Field(default_factory=list)


class AnalysisListItem(BaseModel):
    """Minimal info for listing past analyses."""
    analysis_id: str
    image_name: str
    prediction: str
    confidence: float
    timestamp: datetime
    status: str


class PreprocessResponse(BaseModel):
    """Response from the hair-removal preview endpoint."""
    preprocessed_url: str
    hair_detected: bool
    hair_pixel_count: int = 0
    original_url: str
