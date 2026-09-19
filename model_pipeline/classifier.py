import os
import cv2
import numpy as np


class MelanomaClassifier:

    def __init__(self):
        print("Classifier module initialized.")

        self.model_loaded = False

        # We will connect the trained Swin Transformer here later.
        # For now this module provides the required interface.


    def preprocess(self, image):
        """
        Prepare image for the classification model.
        """

        image = cv2.resize(
            image,
            (224, 224)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        return image


    def predict(self, image):
        """
        Return classification result.

        This is currently a development placeholder.
        The actual Swin Transformer will replace this later.
        """

        processed_image = self.preprocess(
            image
        )

        # Temporary development output.
        prediction = "Melanoma"
        confidence = 0.91

        return {
            "label": prediction,
            "confidence": confidence
        }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    classifier = MelanomaClassifier()

    print(
        "Classifier module ready."
    )