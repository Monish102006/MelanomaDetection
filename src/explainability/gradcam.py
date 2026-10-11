"""
Grad-CAM for Swin Transformer V2
==================================

Development and debugging utility that visualizes where the Swin
Transformer focuses during classification. Generates heatmaps
overlaid on the original dermoscopic image.

This is NOT the LLM-based explainability of Phase 2. Its purpose
is solely to validate that the classifier attends to the lesion
region rather than background artifacts (hair, rulers, ink marks).

Usage:
    gradcam = SwinGradCAM("checkpoints/best_swin_checkpoint.pth")
    result = gradcam.generate("image.jpg")
    gradcam.save_visualization("image.jpg", "outputs/gradcam_samples/")
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ..classification.model import SwinV2Classifier
from ..data.transforms import get_classification_transforms, IMAGENET_MEAN, IMAGENET_STD


class SwinGradCAM:
    """
    Grad-CAM visualization for the Swin Transformer V2 classifier.

    Generates attention heatmaps showing which image regions most
    influenced the classification decision.

    Args:
        checkpoint_path: Path to trained best_swin_checkpoint.pth.
        model_name: timm model identifier.
        img_size: Input image size.
        device: Torch device (auto-detected if None).
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "swinv2_base_window12to16_192to256",
        img_size: int = 256,
        device: torch.device = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.img_size = img_size

        # Load checkpoint (support both raw state_dict and full checkpoint)
        raw = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        if isinstance(raw, dict) and "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
            self.class_names = list(raw.get("class_names", ["melanoma", "non_melanoma"]))
            ckpt_cfg = raw.get("model_config", {})
            model_name = ckpt_cfg.get("model_name", model_name)
        else:
            state_dict = raw
            self.class_names = ["melanoma", "non_melanoma"]

        # Load model
        self.model = SwinV2Classifier(
            model_name=model_name, num_classes=len(self.class_names), pretrained=False
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Target layer: last normalization layer of the Swin backbone
        # This is the standard target for Swin Transformers
        self.target_layer = self._find_target_layer()

        # Preprocessing
        self.transform = get_classification_transforms("val", img_size)

    def _find_target_layer(self):
        """Find the appropriate target layer for Grad-CAM in Swin V2."""
        # For Swin Transformers in timm, the last norm layer works best
        if hasattr(self.model.backbone, "norm"):
            return self.model.backbone.norm
        # Fallback: last layer of the backbone
        layers = list(self.model.backbone.children())
        return layers[-1]

    def generate(self, image_path: str, target_class: Optional[int] = None) -> dict:
        """
        Generate Grad-CAM visualization for a single image.

        Args:
            image_path: Path to the dermoscopic image.
            target_class: Class index to visualize (None = predicted class).

        Returns:
            Dict with keys:
                - original: RGB image array (H, W, 3), float [0, 1]
                - heatmap: Grad-CAM heatmap (H, W), float [0, 1]
                - overlay: Heatmap blended on original (H, W, 3), float [0, 1]
                - prediction: predicted class name
                - confidence: prediction confidence
        """
        # Load original image (for visualization)
        pil_img = Image.open(image_path).convert("RGB")
        original = np.array(pil_img.resize((self.img_size, self.img_size)))
        original_float = original.astype(np.float32) / 255.0

        # Preprocess for model
        transformed = self.transform(image=np.array(pil_img))
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx]) * 100.0

        # Grad-CAM
        target_idx = target_class if target_class is not None else pred_idx
        targets = [ClassifierOutputTarget(target_idx)]

        # Reshape transform for Swin Transformer compatibility
        def reshape_transform(tensor, height=8, width=8):
            if tensor.ndim == 3:
                # Swin outputs (B, H*W, C) — reshape to (B, C, H, W)
                B, HW, C = tensor.shape
                h = w = int(HW ** 0.5)
                return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
            elif tensor.ndim == 4:
                # Timms' SwinV2 often outputs (B, H, W, C)
                return tensor.permute(0, 3, 1, 2)
            return tensor

        cam = GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=reshape_transform,
        )

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # (H, W)

        # Create overlay
        overlay = show_cam_on_image(original_float, grayscale_cam, use_rgb=True)

        return {
            "original": original_float,
            "heatmap": grayscale_cam,
            "overlay": overlay / 255.0 if overlay.max() > 1 else overlay,
            "prediction": self.class_names[pred_idx].capitalize(),
            "confidence": round(confidence, 2),
        }

    def save_visualization(
        self,
        image_path: str,
        output_dir: str,
        target_class: Optional[int] = None,
    ):
        """
        Generate and save Grad-CAM visualization as PNG files.

        Saves three images:
            - {stem}_original.png
            - {stem}_heatmap.png
            - {stem}_overlay.png

        Args:
            image_path: Path to the input image.
            output_dir: Directory to save visualizations.
            target_class: Class to visualize (None = predicted).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        result = self.generate(image_path, target_class)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(result["original"])
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(result["heatmap"], cmap="jet")
        axes[1].set_title("Attention Heatmap")
        axes[1].axis("off")

        axes[2].imshow(result["overlay"])
        axes[2].set_title(
            f"Overlay — {result['prediction']} ({result['confidence']:.1f}%)"
        )
        axes[2].axis("off")

        plt.suptitle(f"Grad-CAM Analysis: {stem}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{stem}_gradcam.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"Grad-CAM saved: {output_dir / f'{stem}_gradcam.png'}")

    def save_cam_visual(
        self,
        cam_result: dict,
        output_path: str | Path,
        stem: str = "gradcam",
    ) -> str:
        """
        Save the Grad-CAM visual (overlay heatmap on original image) as a PNG file.

        Args:
            cam_result: Dict returned by generate() or generate_from_array().
            output_path: Destination file path or directory.
            stem: File stem if output_path is a directory.

        Returns:
            The saved file path as a string.
        """
        out_path = Path(output_path)
        if out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{stem}_gradcam.png"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_path

        overlay = cam_result.get("overlay")
        if overlay is None:
            heatmap = cam_result.get("heatmap")
            if heatmap is not None:
                orig = cam_result.get("original", np.ones_like(heatmap)[:, :, None])
                overlay = show_cam_on_image(orig, heatmap, use_rgb=True)
            else:
                return ""

        if isinstance(overlay, np.ndarray):
            if overlay.max() <= 1.0:
                overlay_uint8 = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
            else:
                overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
            Image.fromarray(overlay_uint8).save(str(out_file), format="PNG")
            return str(out_file)
        return ""

    def generate_from_array(
        self, image: np.ndarray, target_class: Optional[int] = None
    ) -> dict:
        """
        Generate Grad-CAM heatmap from a preprocessed image array.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.
            target_class: Class index to visualize (None = predicted class).

        Returns:
            Same dict format as generate().
        """
        original = np.array(
            Image.fromarray(image).resize((self.img_size, self.img_size))
        )
        original_float = original.astype(np.float32) / 255.0

        # Preprocess for model
        transformed = self.transform(image=image)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx]) * 100.0

        # Grad-CAM
        target_idx = target_class if target_class is not None else pred_idx
        targets = [ClassifierOutputTarget(target_idx)]

        def reshape_transform(tensor, height=8, width=8):
            if tensor.ndim == 3:
                B, HW, C = tensor.shape
                h = w = int(HW ** 0.5)
                return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
            elif tensor.ndim == 4:
                return tensor.permute(0, 3, 1, 2)
            return tensor

        cam = GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=reshape_transform,
        )

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # (H, W)

        overlay = show_cam_on_image(original_float, grayscale_cam, use_rgb=True)

        return {
            "original": original_float,
            "heatmap": grayscale_cam,
            "overlay": overlay / 255.0 if overlay.max() > 1 else overlay,
            "prediction": self.class_names[pred_idx].capitalize(),
            "confidence": round(confidence, 2),
        }

    @staticmethod
    def compute_metrics(heatmap: np.ndarray, mask: np.ndarray) -> dict:
        """
        Compute quantitative explainability metrics by comparing the
        Grad-CAM heatmap against the segmentation mask.

        Args:
            heatmap: Grad-CAM heatmap (H_cam, W_cam), float [0, 1].
            mask: Binary lesion mask (H_mask, W_mask), uint8, values 0 or 1.

        Returns:
            Dict with keys:
                - attention_inside_lesion (float): AIL — fraction of total
                  Grad-CAM activation inside the mask. Range [0, 1].
                - attention_outside_lesion (float): AOL — 1 - AIL.
                - centroid_distance (float): Normalized Euclidean distance
                  between Grad-CAM centroid and mask centroid.
                - mask_cam_iou (float): IoU between thresholded Grad-CAM
                  (>0.5) and the segmentation mask.
        """
        import cv2 as _cv2

        # Resize mask to match heatmap dimensions (not the other way around).
        # The heatmap is typically small (e.g. 256×256 from the model input),
        # while the mask can be very large (e.g. 4288×2848). Upsampling the
        # heatmap to the mask size dilutes activation values and produces
        # misleadingly low AIL scores.
        H_cam, W_cam = heatmap.shape[:2]
        mask_resized = _cv2.resize(
            mask.astype(np.uint8), (W_cam, H_cam),
            interpolation=_cv2.INTER_NEAREST,
        )

        mask_binary = (mask_resized > 0).astype(np.float64)
        total_activation = heatmap.sum()

        # AIL / AOL
        if total_activation > 0:
            ail = float((heatmap * mask_binary).sum() / total_activation)
        else:
            ail = 0.0
        aol = 1.0 - ail

        # Centroid distance (normalized by sqrt of mask area in heatmap space)
        mask_area = mask_binary.sum()
        if mask_area > 0 and total_activation > 0:
            # Mask centroid
            ys, xs = np.where(mask_binary > 0)
            mask_cx, mask_cy = xs.mean(), ys.mean()

            # Heatmap weighted centroid
            y_coords, x_coords = np.mgrid[0:H_cam, 0:W_cam]
            cam_cx = float((x_coords * heatmap).sum() / total_activation)
            cam_cy = float((y_coords * heatmap).sum() / total_activation)

            dist = np.sqrt((cam_cx - mask_cx) ** 2 + (cam_cy - mask_cy) ** 2)
            centroid_distance = float(dist / np.sqrt(mask_area))
        else:
            centroid_distance = 1.0  # worst case

        # Mask-CAM IoU
        cam_binary = (heatmap > 0.5).astype(np.float64)
        intersection = (cam_binary * mask_binary).sum()
        union = ((cam_binary + mask_binary) > 0).sum()
        mask_cam_iou = float(intersection / max(union, 1))

        return {
            "attention_inside_lesion": round(ail, 4),
            "attention_outside_lesion": round(aol, 4),
            "centroid_distance": round(centroid_distance, 4),
            "mask_cam_iou": round(mask_cam_iou, 4),
        }

