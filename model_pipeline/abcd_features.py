import cv2
import numpy as np


class ABCDEFeatureExtractor:

    def __init__(self):
        print("ABCDE feature extraction module initialized.")


    # ========================================================
    # A — Asymmetry
    # ========================================================

    def calculate_asymmetry(self, mask):

        height, width = mask.shape

        center_x = width // 2
        center_y = height // 2

        # Compare left/right
        left = mask[:, :center_x]
        right = mask[:, center_x:]

        right = cv2.flip(
            right,
            1
        )

        min_width = min(
            left.shape[1],
            right.shape[1]
        )

        left = left[:, :min_width]
        right = right[:, :min_width]

        difference = cv2.absdiff(
            left,
            right
        )

        asymmetric_pixels = np.sum(
            difference > 0
        )

        total_pixels = np.sum(
            mask > 0
        )

        if total_pixels == 0:
            return 0.0

        asymmetry_index = (
            asymmetric_pixels /
            total_pixels
        ) * 100

        return float(
            asymmetry_index
        )


    # ========================================================
    # B — Border irregularity
    # ========================================================

    def calculate_border_irregularity(
        self,
        contour
    ):

        perimeter = cv2.arcLength(
            contour,
            True
        )

        area = cv2.contourArea(
            contour
        )

        if area == 0:
            return 0.0

        # Circularity
        circularity = (
            4 *
            np.pi *
            area
        ) / (
            perimeter *
            perimeter
        )

        # Lower circularity = more irregular
        irregularity = max(
            0.0,
            1.0 - circularity
        )

        return float(
            irregularity
        )


    # ========================================================
    # C — Color variation
    # ========================================================

    def calculate_color_variation(
        self,
        image,
        mask
    ):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2HSV
        )

        pixels = hsv[
            mask > 0
        ]

        if len(pixels) == 0:
            return 0.0

        hue_std = np.std(
            pixels[:, 0]
        )

        saturation_std = np.std(
            pixels[:, 1]
        )

        value_std = np.std(
            pixels[:, 2]
        )

        color_variation = np.mean([
            hue_std,
            saturation_std,
            value_std
        ])

        return float(
            color_variation
        )


    # ========================================================
    # D — Diameter
    # ========================================================

    def calculate_diameter(
        self,
        contour
    ):

        x, y, width, height = cv2.boundingRect(
            contour
        )

        diameter_pixels = max(
            width,
            height
        )

        return float(
            diameter_pixels
        )


    # ========================================================
    # E — Evolution (Temporal change & clinical history)
    # ========================================================

    def format_evolution_status(
        self,
        evolution_history: dict = None
    ) -> dict:
        """
        Processes patient or clinical reported evolution (E in ABCDE).
        Since static 2D images capture a single point in time,
        the E criterion incorporates longitudinal change metadata.
        """
        if not evolution_history:
            return {
                "reported_change": False,
                "status": "single_timepoint_capture",
                "notes": "No prior baseline or longitudinal change reported. Clinical history required."
            }

        return {
            "reported_change": bool(evolution_history.get("reported_change", False)),
            "change_types": evolution_history.get("change_types", []),
            "timeframe_months": evolution_history.get("timeframe_months", None),
            "symptoms": evolution_history.get("symptoms", []),
            "notes": evolution_history.get("notes", "Longitudinal change reported by patient/clinician.")
        }


    # ========================================================
    # Extract all ABCDE features
    # ========================================================

    def extract(
        self,
        image,
        mask,
        contour,
        evolution_history: dict = None
    ):

        asymmetry = self.calculate_asymmetry(
            mask
        )

        border = self.calculate_border_irregularity(
            contour
        )

        color = self.calculate_color_variation(
            image,
            mask
        )

        diameter = self.calculate_diameter(
            contour
        )

        evolution = self.format_evolution_status(
            evolution_history
        )

        return {
            "asymmetry_index": asymmetry,
            "border_irregularity_score": border,
            "color_variation_score": color,
            "diameter_pixels": diameter,
            "evolution": evolution
        }


# Backward compatibility alias
ABCDFeatureExtractor = ABCDEFeatureExtractor


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    extractor = ABCDEFeatureExtractor()

    print(
        "ABCDE feature extraction module ready."
    )