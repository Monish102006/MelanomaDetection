import os
import json

from model_pipeline.segmentation import LesionSegmenter
from model_pipeline.abcd_features import ABCDFeatureExtractor
from model_pipeline.classifier import MelanomaClassifier


class MelanomaPipeline:

    def __init__(self):
        print("Initializing melanoma pipeline...")

        self.segmenter = LesionSegmenter()
        self.feature_extractor = ABCDFeatureExtractor()
        self.classifier = MelanomaClassifier()

        print("Melanoma pipeline initialized.")


    def analyze(self, image_path):

        print("\n" + "=" * 70)
        print("MELANOMA IMAGE ANALYSIS PIPELINE")
        print("=" * 70)

        # ----------------------------------------------------
        # 1. SEGMENTATION
        # ----------------------------------------------------

        print("\n[1/3] Segmenting lesion...")

        segmentation_result = self.segmenter.segment(
            image_path
        )

        image = segmentation_result["image"]
        mask = segmentation_result["mask"]
        contour = segmentation_result["contour"]

        print("Segmentation completed.")


        # ----------------------------------------------------
        # 2. ABCD FEATURES
        # ----------------------------------------------------

        print("\n[2/3] Extracting ABCD features...")

        features = self.feature_extractor.extract(
            image,
            mask,
            contour
        )

        print("ABCD features:")
        print(
            json.dumps(
                features,
                indent=2
            )
        )


        # ----------------------------------------------------
        # 3. CLASSIFICATION
        # ----------------------------------------------------

        print("\n[3/3] Running classifier...")

        prediction = self.classifier.predict(
            image
        )

        print("Classification:")
        print(
            json.dumps(
                prediction,
                indent=2
            )
        )


        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        result = {

            "prediction": prediction,

            "abcd_metrics": features,

            "image_path": image_path
        }

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED")
        print("=" * 70)

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pipeline = MelanomaPipeline()

    print("\nPipeline is ready.")

    print(
        "Usage:"
    )

    print(
        'pipeline.analyze("path/to/image.jpg")'
    )