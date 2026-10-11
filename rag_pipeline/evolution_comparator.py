"""Mask-based longitudinal lesion comparison for the E in ABCDE."""

from datetime import date
from typing import Any, Iterable, Optional

import cv2
import numpy as np

try:
    from .evolution_models import EvolutionComparisonResult
except ImportError:
    from rag_pipeline.evolution_models import EvolutionComparisonResult


class EvolutionComparator:
    """Compare two segmented lesion images without full-image registration.

    The comparator normalizes each lesion-mask crop before shape comparison.
    This deliberately measures relative morphology, not pixel-perfect physical
    correspondence between independently captured photographs.
    """

    NORMALIZED_CANVAS = 256

    @staticmethod
    def _clean_labels(values: Optional[Iterable[str]]) -> list[str]:
        return sorted({str(value).strip().lower() for value in (values or []) if str(value).strip()})

    @staticmethod
    def _percent_change(baseline: float, followup: float) -> Optional[float]:
        if baseline <= 0:
            return None
        return round(((followup - baseline) / baseline) * 100.0, 2)

    @staticmethod
    def _normalized_mask(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        points = cv2.findNonZero(binary)
        if points is None:
            raise ValueError("Cannot compare an empty lesion mask.")

        x, y, width, height = cv2.boundingRect(points)
        crop = binary[y:y + height, x:x + width]
        target = EvolutionComparator.NORMALIZED_CANVAS
        scale = min((target - 32) / width, (target - 32) / height)
        resized = cv2.resize(
            crop,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = np.zeros((target, target), dtype=np.uint8)
        offset_y = (target - resized.shape[0]) // 2
        offset_x = (target - resized.shape[1]) // 2
        canvas[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
        return canvas

    @staticmethod
    def _shape_overlap(baseline_mask: np.ndarray, followup_mask: np.ndarray) -> float:
        first = EvolutionComparator._normalized_mask(baseline_mask).astype(bool)
        second = EvolutionComparator._normalized_mask(followup_mask).astype(bool)
        union = np.logical_or(first, second).sum()
        return round(float(np.logical_and(first, second).sum() / union), 4) if union else 0.0

    @staticmethod
    def _masked_lab_median(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image_rgb.shape[:2] != mask.shape[:2]:
            raise ValueError("Image and segmentation mask dimensions must match.")
        pixels = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[mask > 0]
        if len(pixels) == 0:
            raise ValueError("Cannot compare colour for an empty lesion mask.")
        return np.median(pixels.astype(np.float32), axis=0)

    def compare(
        self,
        baseline_image: np.ndarray,
        baseline_mask: np.ndarray,
        baseline_metrics: dict[str, Any],
        followup_image: np.ndarray,
        followup_mask: np.ndarray,
        followup_metrics: dict[str, Any],
        baseline_date: date,
        followup_date: date,
        reported_changes: Optional[Iterable[str]] = None,
        reported_symptoms: Optional[Iterable[str]] = None,
        lesion_site: Optional[str] = None,
        self_reported_prior_history: Optional[str] = None,
    ) -> EvolutionComparisonResult:
        if followup_date <= baseline_date:
            raise ValueError("followup_date must be later than baseline_date")

        first_area = float(np.count_nonzero(baseline_mask))
        second_area = float(np.count_nonzero(followup_mask))
        if first_area == 0 or second_area == 0:
            raise ValueError("Both visits require a non-empty lesion segmentation mask.")

        first_diameter = float(baseline_metrics.get("diameter_pixels", 0.0))
        second_diameter = float(followup_metrics.get("diameter_pixels", 0.0))
        first_border = float(baseline_metrics.get("border_irregularity_score", 0.0))
        second_border = float(followup_metrics.get("border_irregularity_score", 0.0))

        lab_delta = float(np.linalg.norm(
            self._masked_lab_median(followup_image, followup_mask)
            - self._masked_lab_median(baseline_image, baseline_mask)
        ))
        # OpenCV Lab's maximum Euclidean distance is approximately 375. The
        # normalized score is descriptive only and is not a clinical cutoff.
        color_score = min(1.0, lab_delta / 375.0)
        flags: list[str] = ["no_physical_scale_reference"]
        if baseline_image.shape[:2] != followup_image.shape[:2]:
            flags.append("different_image_dimensions")

        confidence = "medium"
        if "different_image_dimensions" in flags:
            confidence = "low"

        changes = self._clean_labels(reported_changes)
        symptoms = self._clean_labels(reported_symptoms)
        limitations = [
            "Size changes are relative image-space measurements because no shared physical scale was supplied.",
            "Colour differences can be affected by illumination, camera settings, and image processing.",
            "Normalized shape overlap compares lesion morphology after mask-crop normalization; it is not physical image registration.",
        ]

        return EvolutionComparisonResult(
            baseline_date=baseline_date,
            followup_date=followup_date,
            interval_days=(followup_date - baseline_date).days,
            area_change_percent=self._percent_change(first_area, second_area),
            diameter_change_percent=self._percent_change(first_diameter, second_diameter),
            border_change_score=round(abs(second_border - first_border), 4),
            color_change_score=round(color_score, 4),
            color_lab_delta=round(lab_delta, 4),
            normalized_shape_overlap=self._shape_overlap(baseline_mask, followup_mask),
            reported_changes=changes,
            reported_symptoms=symptoms,
            reported_change=bool(changes or symptoms),
            lesion_site=lesion_site,
            self_reported_prior_history=self_reported_prior_history,
            comparison_confidence=confidence,
            quality_flags=flags,
            limitations=limitations,
        )
