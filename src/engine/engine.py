"""
Core Diagnosis Engine
======================

The single public interface for melanoma diagnosis. Initializes all
components from config, runs the inference pipeline, and packages
outputs into a standardized DiagnosisResult.

Downstream Phase 2 agents interact ONLY with this engine.
"""

import logging
from pathlib import Path

import yaml
import torch
import numpy as np
import cv2

logger = logging.getLogger(__name__)

from .diagnosis import (
    DiagnosisResult, FeatureScore, DiagnosisInfo,
    SegmentationInfo, MetadataInfo, ExplainabilityInfo, PreprocessingInfo,
    PipelineInfo, ScaleCalibrationInfo,
)
from .inference_pipeline import InferencePipeline
from .measurements import extract_lesion_measurements, refine_lesion_mask, refine_mask_with_scale
from .scale_detection import ScaleDetector, ScaleCalibration
from ..classification.inference import SwinV2Predictor
from ..segmentation.segformer_wrapper import SegFormerSegmenter
from ..features.extractor import ABCFeatureExtractor
from ..data.preprocessing import DermoscopyPreprocessor
from ..explainability.gradcam import SwinGradCAM


class CoreDiagnosisEngine:
    """
    Core Diagnosis Engine — single source of truth for melanoma diagnosis.

    Initializes all AI components from a config file and provides
    a simple `.diagnose(image_path)` interface that returns a
    DiagnosisResult consumed by Phase 2 agents.

    Args:
        config_path: Path to config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = str(Path(config_path).resolve())
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        device = self._resolve_device()
        self._init_components(device)

    def _resolve_device(self) -> torch.device:
        dev = self.config["project"]["device"]
        if dev == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev)

    def _init_components(self, device: torch.device):
        """Initialize all AI components."""
        cls_cfg = self.config["classification"]
        seg_cfg = self.config["segmentation"]
        feat_cfg = self.config.get("features", {})
        preproc_cfg = self.config.get("preprocessing", {})
        paths = self.config["paths"]

        # Preprocessor
        self.preprocessor = DermoscopyPreprocessor(
            hair_removal=preproc_cfg.get("hair_removal", True),
            illumination_normalization=preproc_cfg.get("illumination_normalization", True),
            preserve_scale_reference=preproc_cfg.get("preserve_scale_reference", True),
            max_processing_dimension=preproc_cfg.get("max_processing_dimension", 1600),
        )

        # Classifier — prefer explicit classification_checkpoint; fall back to
        # legacy best_swin_checkpoint.pth inside the checkpoints folder.
        project_root = Path(self.config_path).parent
        if paths.get("classification_checkpoint"):
            checkpoint = str(project_root / paths["classification_checkpoint"])
        else:
            checkpoint = str(project_root / "checkpoints" / "best_swin_checkpoint.pth")
        self.classifier = SwinV2Predictor(
            checkpoint_path=checkpoint,
            model_name=cls_cfg["model_name"],
            img_size=cls_cfg["img_size"],
            device=device,
            class_names=cls_cfg.get("class_names"),
            melanoma_class_name=cls_cfg.get("melanoma_class_name", "melanoma"),
            classification_threshold=cls_cfg.get("classification_threshold"),
        )

        # Segmenter
        self.segmenter = SegFormerSegmenter(
            checkpoint_path=str(project_root / seg_cfg["checkpoint"]),
            device=device,
            encoder_name=seg_cfg.get("encoder_name", "mit_b2"),
            input_size=seg_cfg.get("input_size", 512),
            closing_kernel=seg_cfg.get("morphological_closing_kernel", 5),
        )

        # Feature extractor
        self.feature_extractor = ABCFeatureExtractor(config=feat_cfg)

        # Grad-CAM (reuses the same Swin model weights)
        self.gradcam = SwinGradCAM(
            checkpoint_path=checkpoint,
            model_name=cls_cfg["model_name"],
            img_size=cls_cfg["img_size"],
            device=device,
        )

        # Inference pipeline
        self.pipeline = InferencePipeline(
            classifier=self.classifier,
            segmenter=self.segmenter,
            feature_extractor=self.feature_extractor,
            preprocessor=self.preprocessor,
        )

        # Scale detector for reference-object calibration
        self.scale_detector = ScaleDetector()

    def diagnose(
        self,
        image_path: str,
        save_mask: bool = False,
        request_id: str | None = None,
        scale_method: str = "auto",
        scale_reference_mm: float | None = None,
        scale_reference_key: str | None = None,
    ) -> DiagnosisResult:
        """
        Run full diagnosis on a single dermoscopic image.

        Args:
            image_path: Path to the input image.
            save_mask: If True, save the segmentation mask to outputs/.

        Returns:
            DiagnosisResult object.
        """
        # Validate input
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run inference pipeline
        raw = self.pipeline.run(image_path)
        raw_mask = np.asarray(raw["mask"])

        # Detect a known reference before choosing the lesion component.  If
        # the segmentation model has included the coin/sticker in its mask,
        # its detected bounding box can then be excluded from lesion
        # measurement. Unknown circles are deliberately not excluded because
        # an arbitrary circle may be the lesion itself.
        scale_cal = self.scale_detector.detect(
            raw["original_image"],
            method=scale_method,
            known_diameter_mm=scale_reference_mm,
            reference_key=scale_reference_key,
            lesion_mask=raw_mask,
            aruco_marker_size_mm=self.config.get("calibration", {}).get("aruco_marker_size_mm"),
            aruco_marker_id=self.config.get("calibration", {}).get("aruco_marker_id"),
            charuco_squares_x=self.config.get("calibration", {}).get("charuco_squares_x"),
            charuco_squares_y=self.config.get("calibration", {}).get("charuco_squares_y"),
            charuco_square_length_mm=self.config.get("calibration", {}).get("charuco_square_length_mm"),
            charuco_marker_length_mm=self.config.get("calibration", {}).get("charuco_marker_length_mm"),
            charuco_dictionary_id=self.config.get("calibration", {}).get("charuco_dictionary_id", 0),
            checkerboard_inner_corners=tuple(self.config.get("calibration", {}).get("checkerboard_inner_corners", [])) or None,
            checkerboard_square_size_mm=self.config.get("calibration", {}).get("checkerboard_square_size_mm"),
        )
        pixels_per_mm = (
            scale_cal.pixels_per_mm if scale_cal.calibration_valid else None
        )

        # Validate SegFormer output before any measurement is attempted.
        # Keep measurements tied to the interior lesion component.  This
        # prevents a border/background/reference-object segmentation from
        # being reported as the lesion's size.
        if raw_mask.ndim != 2:
            raise RuntimeError("segmentation_failed")
        try:
            # scale_cal.reference_bbox_px is in original-image pixel space.
            # raw_mask is in preprocessed (possibly resized) pixel space.
            # Scale the exclusion bbox to match the mask dimensions so the
            # scale reference is correctly excluded from lesion selection.
            reference_bbox = None
            if scale_cal.detected and scale_cal.calibration_valid and scale_cal.reference_bbox_px is not None:
                orig_h, orig_w = raw["original_image"].shape[:2]
                mask_h, mask_w = raw_mask.shape[:2]
                if orig_h > 0 and orig_w > 0 and (orig_h != mask_h or orig_w != mask_w):
                    sx = mask_w / orig_w
                    sy = mask_h / orig_h
                    bx, by, bw, bh = scale_cal.reference_bbox_px
                    reference_bbox = (
                        int(round(bx * sx)),
                        int(round(by * sy)),
                        max(1, int(round(bw * sx))),
                        max(1, int(round(bh * sy))),
                    )
                else:
                    reference_bbox = scale_cal.reference_bbox_px
            max_ratio = float(self.config.get("orchestrator", {}).get("segmentation_max_area_ratio", 0.90))
            mask = refine_lesion_mask(raw_mask, max_area_ratio=max_ratio, excluded_bbox=reference_bbox)
        except ValueError as exc:
            raise RuntimeError("segmentation_failed") from exc
        components_before_cleanup = int(cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1)
        # Segmentation refinement is independent of physical calibration.
        # Previously this block ran only when pixels_per_mm existed, so an
        # uncertain ruler silently caused the raw, inflated model mask to be
        # measured. Calibration controls only the px->mm conversion.
        cleaned_mask = refine_mask_with_scale(
            raw.get("hair_removed_image", raw["preprocessed_image"]),
            mask, pixels_per_mm or 1.0,
        )
        if int((cleaned_mask > 0).sum()) > 0:
            mask = cleaned_mask
        components_after_cleanup = int(cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1)
        mask_area = int((mask > 0).sum())
        image_area = int(mask.shape[0] * mask.shape[1])
        min_area = int(self.config.get("orchestrator", {}).get("segmentation_min_area_px", 100))
        max_ratio = float(self.config.get("orchestrator", {}).get("segmentation_max_area_ratio", 0.90))
        connected = cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1
        if (
            mask.ndim != 2
            or mask_area < min_area
            or mask_area / max(image_area, 1) >= max_ratio
            or connected < 1
        ):
            raise RuntimeError("segmentation_failed")

        # Optionally save mask and a dashboard-safe overlay.
        mask_path = None
        overlay_path = None
        if save_mask:
            output_dir = Path(self.config_path).parent / self.config["paths"]["outputs"] / "masks"
            output_dir.mkdir(parents=True, exist_ok=True)
            mask_file = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_file), mask.astype(np.uint8) * 255)
            mask_path = str(mask_file)
            original = cv2.cvtColor(np.asarray(raw["preprocessed_image"]), cv2.COLOR_RGB2BGR)
            overlay = original.copy()
            overlay[mask > 0] = (0, 180, 255)
            overlay = cv2.addWeighted(original, 0.72, overlay, 0.28, 0)
            overlay_file = output_dir / f"{img_path.stem}_overlay.png"
            cv2.imwrite(str(overlay_file), overlay)
            overlay_path = str(overlay_file)

        # Build clinical features and measurements from the same refined
        # lesion component used for area/diameter.  The pipeline extracts a
        # provisional feature set before this engine-level integrity check;
        # re-extracting here prevents ABC features from describing a border,
        # background, or reference-object component that was excluded from
        # the reported lesion measurements.
        cls = raw["classification"]
        features = self.feature_extractor.extract_all(
            raw["preprocessed_image"], mask
        )

        clinical_features = {}
        measurements = {}
        clinical_interpretations = {}

        for name, result in features.items():
            clinical_features[name] = FeatureScore(
                name=result.name,
                score_numeric=result.score_numeric,
                score_label=result.score_label,
            )
            measurements[name] = result.details
            clinical_interpretations[name] = result.score_label

        # Geometric measurements are evidence from the uploaded image.  Hair
        # removal is useful for model inference, but it can inpaint lesion
        # edges and printed reference marks; do not let it change the measured
        # diameter, perimeter, or physical size.
        measurement_image = raw["original_image"]
        measurements["lesion"] = extract_lesion_measurements(measurement_image, mask)

        # Re-compute lesion measurements with physical scale if available
        if pixels_per_mm:
            measurements["lesion"] = extract_lesion_measurements(
                measurement_image, mask, pixels_per_mm=pixels_per_mm,
                scale_confidence=scale_cal.confidence,
                pixel_to_mm_homography=scale_cal.homography,
            )

        # Keep the complete, machine-readable measurement contract together
        # with the lesion values.  The scale detector remains the source of
        # truth for the calibration metadata and region.
        measurements["lesion"].update({
            "pixels_per_mm": round(float(pixels_per_mm), 4) if pixels_per_mm else None,
            "scale_detected": bool(scale_cal.detected and scale_cal.calibration_valid),
            "scale_confidence": round(float(scale_cal.confidence), 4),
            "mask_cleanup": {
                "connected_components_before": components_before_cleanup,
                "connected_components_after": components_after_cleanup,
                "area_px": int((mask > 0).sum()),
                "bounding_box": measurements["lesion"].get("lesion_region", {}),
            },
            "lesion_region": measurements["lesion"].get("lesion_region", {}),
            "scale_region": (
                {
                    "x": int(scale_cal.reference_bbox_px[0]),
                    "y": int(scale_cal.reference_bbox_px[1]),
                    "width": int(scale_cal.reference_bbox_px[2]),
                    "height": int(scale_cal.reference_bbox_px[3]),
                }
                if scale_cal.reference_bbox_px is not None else {}
            ),
            "measurement_method": measurements["lesion"].get(
                "measurement_method", "maximum_feret_diameter"
            ),
        })

        # Always propagate calibration metadata, including invalid attempts,
        # so downstream reports can explain why physical units were withheld.
        scale_calibration_info = ScaleCalibrationInfo(
            pixels_per_mm=scale_cal.pixels_per_mm,
            method=scale_cal.method,
            confidence=scale_cal.confidence,
            reference_type=scale_cal.reference_type,
            reference_diameter_px=scale_cal.reference_diameter_px,
            reference_bbox_px=scale_cal.reference_bbox_px,
            reference_diameter_mm=scale_cal.reference_diameter_mm,
            detected=scale_cal.detected,
            calibration_confidence=scale_cal.confidence,
            calibration_valid=scale_cal.calibration_valid,
            calibration_method=scale_cal.method,
            reference_object_type=scale_cal.reference_type,
            reference_length_px=scale_cal.reference_length_px,
            reference_length_mm=scale_cal.reference_length_mm,
            calibration_reason=scale_cal.validation_reason,
            scale_region=(
                {
                    "x": int(scale_cal.reference_bbox_px[0]),
                    "y": int(scale_cal.reference_bbox_px[1]),
                    "width": int(scale_cal.reference_bbox_px[2]),
                    "height": int(scale_cal.reference_bbox_px[3]),
                }
                if scale_cal.reference_bbox_px is not None else None
            ),
            orientation=scale_cal.orientation,
            angle_degrees=scale_cal.angle_degrees,
            interval_mm=scale_cal.interval_mm,
            tick_positions_px=list(scale_cal.tick_positions_px) if scale_cal.tick_positions_px is not None else None,
            tick_spacing_px=scale_cal.tick_spacing_px,
            validated_tick_count=len(scale_cal.tick_positions_px) if scale_cal.tick_positions_px is not None else 0,
            reference_points_px=list(scale_cal.reference_points_px) if scale_cal.reference_points_px is not None else None,
            reprojection_error_px=scale_cal.reprojection_error_px,
            calibration_uncertainty=scale_cal.calibration_uncertainty,
            warnings=list(scale_cal.warnings),
            homography=[list(row) for row in scale_cal.homography] if scale_cal.homography is not None else None,
            axis_endpoints_px=scale_cal.axis_endpoints_px,
            tick_points_px=list(scale_cal.tick_points_px) if scale_cal.tick_points_px is not None else None,
            interval_residuals_px=list(scale_cal.interval_residuals_px) if scale_cal.interval_residuals_px is not None else None,
        )

        # Update diameter interpretation
        if scale_cal.calibration_valid and measurements["lesion"].get("diameter_mm") is not None:
            d_mm = measurements["lesion"]["diameter_mm"]
            if d_mm >= 6.0:
                clinical_interpretations["diameter"] = (
                    f"Estimated diameter {d_mm:.1f} mm exceeds the 6 mm ABCDE "
                    f"D-criterion threshold. Professional evaluation is recommended."
                )
            else:
                clinical_interpretations["diameter"] = (
                    f"Estimated diameter {d_mm:.1f} mm is below the 6 mm threshold "
                    f"for the D-criterion. Continue monitoring for changes."
                )
        else:
            clinical_interpretations["diameter"] = (
                "Pixel diameter is available; physical measurement is unavailable "
                f"because reference calibration is unreliable ({scale_cal.validation_reason})."
            )

        warnings = []
        if raw.get("segmentation_recovery_used", False):
            warnings.append(
                "SegFormer produced a dominant dermoscope-field mask; an "
                "image-based vignette recovery was used. Review the lesion "
                "outline before relying on measurements."
            )
        if scale_cal.method != "none" and not scale_cal.calibration_valid:
            warnings.append(f"Physical measurement unavailable: {scale_cal.validation_reason}")

        area_px = measurements["lesion"]["area_px"]
        perimeter_px = measurements["lesion"]["perimeter_px"]

        # Grad-CAM requires a second model pass with gradients and is the
        # dominant avoidable CPU cost on machines without CUDA.  It is
        # optional because diagnosis, segmentation, calibration, and the
        # report remain valid without it.
        grad_cam_path = None
        if self.config.get("explainability", {}).get("compute_gradcam", False):
            try:
                cam_result = self.gradcam.generate_from_array(raw["preprocessed_image"])
                cam_metrics = SwinGradCAM.compute_metrics(cam_result["heatmap"], mask)
                gradcam_dir = (
                    Path(self.config_path).parent
                    / self.config.get("paths", {}).get("outputs", "outputs")
                    / "gradcam_samples"
                )
                gradcam_dir.mkdir(parents=True, exist_ok=True)
                sample_name = request_id or img_path.stem
                gradcam_file = gradcam_dir / f"{sample_name}_gradcam.png"
                self.gradcam.save_cam_visual(cam_result, gradcam_file)
                grad_cam_path = str(gradcam_file)
                cam_metrics["grad_cam_path"] = grad_cam_path
                explainability = ExplainabilityInfo(**cam_metrics)
            except Exception as e:
                logger.warning("Grad-CAM generation or storage failed: %s", e)
                explainability = ExplainabilityInfo()
        else:
            explainability = ExplainabilityInfo()

        return DiagnosisResult(
            diagnosis=DiagnosisInfo(
                prediction=cls["prediction"],
                confidence=cls["confidence"]
            ),
            probabilities=cls["probabilities"],
            measurements=measurements,
            clinical_features=clinical_features,
            clinical_interpretations=clinical_interpretations,
            segmentation=SegmentationInfo(
                status="completed",
                mask_path=mask_path,
                overlay_path=overlay_path,
                area_px=area_px,
                perimeter_px=perimeter_px
            ),
            explainability=explainability,
            grad_cam_path=grad_cam_path,
            preprocessing=PreprocessingInfo(**raw["preprocessing_applied"]),
            scale_calibration=scale_calibration_info,
            pipeline=PipelineInfo(classification_threshold=cls.get("classification_threshold")),
            metadata=MetadataInfo(
                image_path=str(image_path), request_id=request_id
            ),
            classification_threshold=cls.get("classification_threshold"),
            processing_status="completed",
            warnings=warnings,
        )
