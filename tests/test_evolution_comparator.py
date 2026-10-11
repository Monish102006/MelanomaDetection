"""Regression tests for the mask-based E criterion comparator."""

from datetime import date
import unittest

import cv2
import numpy as np

from rag_pipeline.evolution_comparator import EvolutionComparator


def visit(radius: int, colour: tuple[int, int, int]):
    image = np.full((200, 200, 3), 190, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), radius, 255, thickness=-1)
    image[mask > 0] = colour
    metrics = {
        "diameter_pixels": float(radius * 2),
        "border_irregularity_score": 0.2,
    }
    return image, mask, metrics


class EvolutionComparatorTests(unittest.TestCase):
    def test_returns_relative_longitudinal_measurements(self):
        baseline_image, baseline_mask, baseline_metrics = visit(30, (90, 70, 50))
        followup_image, followup_mask, followup_metrics = visit(36, (130, 70, 50))

        result = EvolutionComparator().compare(
            baseline_image, baseline_mask, baseline_metrics,
            followup_image, followup_mask, followup_metrics,
            date(2026, 1, 1), date(2026, 4, 1),
            reported_changes=["darkening"],
            reported_symptoms=["itching"],
            lesion_site="left forearm",
            self_reported_prior_history="User reports a prior dermatology review.",
        )

        self.assertEqual(result.interval_days, 90)
        self.assertGreater(result.area_change_percent, 0)
        self.assertEqual(result.diameter_change_percent, 20.0)
        self.assertGreater(result.color_change_score, 0)
        self.assertTrue(result.reported_change)
        self.assertFalse(result.scale_available)
        self.assertIn("no_physical_scale_reference", result.quality_flags)
        self.assertEqual(result.lesion_site, "left forearm")
        self.assertIn("unverified", result.to_rag_evolution()["notes"])
        self.assertEqual(result.to_rag_evolution()["status"], "longitudinal_image_comparison")

    def test_rejects_non_chronological_visits(self):
        image, mask, metrics = visit(30, (90, 70, 50))
        with self.assertRaises(ValueError):
            EvolutionComparator().compare(
                image, mask, metrics, image, mask, metrics,
                date(2026, 4, 1), date(2026, 4, 1),
            )


if __name__ == "__main__":
    unittest.main()
