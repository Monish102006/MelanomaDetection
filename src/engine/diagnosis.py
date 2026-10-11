"""
DiagnosisResult Schema
=======================

Pydantic-based data model that serves as the standardized interface
between Phase 1 (Core Diagnosis Engine) and Phase 2 agents
(Medical Retrieval, Explainability, Report Generation).
"""

from datetime import datetime
from typing import Dict, Optional, Any

from pydantic import BaseModel, Field


class DiagnosisInfo(BaseModel):
    prediction: str                                # "Melanoma" | "Benign"
    confidence: float = Field(ge=0.0, le=100.0)    # percentage


class FeatureScore(BaseModel):
    """Score for a single clinical feature (e.g., asymmetry)."""
    name: str                                     # "asymmetry", "border", "color"
    score_numeric: float = Field(ge=0.0, le=1.0)  # normalized 0.0–1.0
    score_label: str                              # "High", "Irregular", "Multiple"
    details: Optional[Dict[str, Any]] = None      # Extra information (e.g., evolution JSON)


class SegmentationInfo(BaseModel):
    status: str = "completed"
    mask_path: Optional[str] = None
    overlay_path: Optional[str] = None
    area_px: Optional[int] = None
    perimeter_px: Optional[float] = None


class ExplainabilityInfo(BaseModel):
    """Quantitative Grad-CAM metrics measuring classifier attention alignment."""
    attention_inside_lesion: Optional[float] = None   # AIL: fraction of attention inside mask (0–1)
    attention_outside_lesion: Optional[float] = None   # AOL: 1 - AIL
    centroid_distance: Optional[float] = None          # normalized distance between GradCAM and mask centroids
    mask_cam_iou: Optional[float] = None               # IoU between thresholded GradCAM and mask
    grad_cam_path: Optional[str] = None              # Path to stored Grad-CAM visual


class PreprocessingInfo(BaseModel):
    """Records which preprocessing steps were applied to the input image."""
    hair_removal: bool = False
    illumination_normalization: bool = False


class ScaleCalibrationInfo(BaseModel):
    """Reference-object scale calibration data."""
    pixels_per_mm: Optional[float] = None
    method: str = "none"            # none | auto | circle | ruler | manual
    confidence: float = 0.0
    reference_type: str = "none"     # none | coin | sticker | ruler | manual | circle_unknown
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
    scale_region: Optional[dict[str, int]] = None
    orientation: str = "unknown"
    angle_degrees: Optional[float] = None
    interval_mm: Optional[float] = None
    tick_positions_px: Optional[list[float]] = None
    tick_spacing_px: Optional[float] = None
    validated_tick_count: int = 0
    reference_points_px: Optional[list[tuple[float, float]]] = None
    reprojection_error_px: Optional[float] = None
    calibration_uncertainty: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    homography: Optional[list[list[float]]] = None
    axis_endpoints_px: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    tick_points_px: Optional[list[tuple[float, float]]] = None
    interval_residuals_px: Optional[list[float]] = None


class PipelineInfo(BaseModel):
    classification_model: str = "SwinV2 Base"
    segmentation_model: str = "SegFormer"
    feature_extractor: str = "OpenCV + scikit-image"
    pipeline_version: str = "1.0"
    classification_threshold: Optional[float] = None


class MetadataInfo(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    image_path: str
    request_id: Optional[str] = None


class DiagnosisResult(BaseModel):
    """
    Standardized diagnostic output from the Core Diagnosis Engine.
    """
    diagnosis: DiagnosisInfo
    probabilities: Dict[str, float]
    classification_threshold: Optional[float] = None
    measurements: Dict[str, Dict[str, Any]]
    clinical_features: Dict[str, FeatureScore]
    clinical_interpretations: Dict[str, str] = Field(default_factory=dict)
    segmentation: SegmentationInfo
    explainability: ExplainabilityInfo = Field(default_factory=ExplainabilityInfo)
    grad_cam_path: Optional[str] = None
    preprocessing: PreprocessingInfo = Field(default_factory=PreprocessingInfo)
    scale_calibration: Optional[ScaleCalibrationInfo] = None
    pipeline: PipelineInfo = Field(default_factory=PipelineInfo)
    metadata: MetadataInfo
    processing_status: str = "completed"
    warnings: list[str] = Field(default_factory=list)

    def to_summary(self) -> str:
        """Human-readable summary for logging/debugging."""
        lines = [
            f"Diagnosis: {self.diagnosis.prediction} ({self.diagnosis.confidence:.1f}%)",
            f"Image: {self.metadata.image_path}",
        ]
        for name, feat in self.clinical_features.items():
            lines.append(f"  {name}: {feat.score_label} ({feat.score_numeric:.3f})")
        if self.explainability.attention_inside_lesion is not None:
            lines.append(f"  Attention inside lesion: {self.explainability.attention_inside_lesion:.3f}")
        return "\n".join(lines)
