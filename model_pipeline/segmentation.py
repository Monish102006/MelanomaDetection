import os
import cv2
import numpy as np


class LesionSegmenter:

    def __init__(self):
        print("Lesion segmentation module initialized.")


    def load_image(self, image_path: str):

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image


    def segment(self, image_path: str):

        image = self.load_image(
            image_path
        )

        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Convert to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Otsu threshold
        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological cleanup
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:

            raise ValueError(
                "No lesion region detected."
            )

        # Select largest contour
        lesion_contour = max(
            contours,
            key=cv2.contourArea
        )

        # Create clean binary mask
        clean_mask = np.zeros_like(
            mask
        )

        cv2.drawContours(
            clean_mask,
            [lesion_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # Extract lesion
        lesion = cv2.bitwise_and(
            rgb,
            rgb,
            mask=clean_mask
        )

        return {
            "image": rgb,
            "mask": clean_mask,
            "lesion": lesion,
            "contour": lesion_contour
        }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    segmenter = LesionSegmenter()

    print(
        "Segmentation module ready."
    )