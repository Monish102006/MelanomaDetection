"""
SegFormer Segmentation Wrapper
==============================

Wraps the pretrained SegFormer model for lesion segmentation on
2D dermoscopic images. Does not require a bounding box prompt.
"""

import logging
import re
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from skimage import transform as sk_transform

logger = logging.getLogger(__name__)


class SegFormerSegmenter:
    """
    SegFormer-based lesion segmentation for dermoscopic images.

    Loads the pretrained checkpoint and generates binary
    segmentation masks given an input image.

    Args:
        checkpoint_path: Path to the .pth or .pt checkpoint.
        device: Torch device (auto-detected if None).
        encoder_name: Encoder name (e.g. "mit_b2" or "nvidia/mit-b2").
        input_size: Expected input resolution (default: 512).
        closing_kernel: Morphological closing kernel size for post-processing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = None,
        encoder_name: str = "mit_b2",
        input_size: int = 512,
        closing_kernel: int = 5,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.input_size = input_size
        self.closing_kernel = closing_kernel
        self.last_probability_map = None
        self.last_used_recovery = False

        # Load weights
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else (ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        )

        # Respect training resolution if present in checkpoint and default input_size passed
        if isinstance(ckpt, dict) and "image_size" in ckpt and input_size == 512:
            self.input_size = int(ckpt["image_size"])

        # Detect whether checkpoint is HuggingFace SegFormer or SMP SegFormer
        has_stages = any("segformer.stages" in k for k in state_dict.keys())
        has_encoder = any("segformer.encoder" in k for k in state_dict.keys())
        has_hf_decode = any("decode_head" in k for k in state_dict.keys())

        if has_stages or has_encoder or has_hf_decode:
            self._init_hf_model(ckpt, state_dict, encoder_name)
        else:
            self._init_smp_model(state_dict, encoder_name)

    def _init_hf_model(self, ckpt, state_dict: dict, encoder_name: str) -> None:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        clean_sd = {}
        for k, v in state_dict.items():
            new_k = k
            new_k = new_k.replace("decode_head.linear_projections.", "decode_head.linear_c.")
            m = re.match(r"segformer\.stages\.(\d+)\.patch_embeddings\.(.*)", new_k)
            if m:
                new_k = f"segformer.encoder.patch_embeddings.{m.group(1)}.{m.group(2)}"
            m = re.match(r"segformer\.stages\.(\d+)\.layer_norm\.(.*)", new_k)
            if m:
                new_k = f"segformer.encoder.layer_norm.{m.group(1)}.{m.group(2)}"
            m = re.match(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.(.*)", new_k)
            if m:
                s, b, r = m.groups()
                r = r.replace("layernorm_before", "layer_norm_1").replace("layernorm_after", "layer_norm_2")
                r = r.replace("attention.q_proj", "attention.self.query").replace("attention.k_proj", "attention.self.key")
                r = r.replace("attention.v_proj", "attention.self.value").replace("attention.o_proj", "attention.output.dense")
                r = r.replace("attention.sequence_reduction.sequence_reduction", "attention.self.sr")
                r = r.replace("attention.sequence_reduction.layer_norm", "attention.self.layer_norm")
                r = r.replace("mlp.fc1", "mlp.dense1").replace("mlp.fc2", "mlp.dense2")
                new_k = f"segformer.encoder.block.{s}.{b}.{r}"
            clean_sd[new_k] = v

        num_labels = 2
        for k in clean_sd:
            if "classifier.weight" in k:
                num_labels = clean_sd[k].shape[0]
                break

        model_name = (
            ckpt.get("model_name", encoder_name)
            if isinstance(ckpt, dict)
            else encoder_name
        )
        if not model_name.startswith("nvidia/") and "mit" in model_name:
            model_name = "nvidia/" + model_name.replace("_", "-")

        try:
            cfg = SegformerConfig.from_pretrained(
                model_name, num_labels=num_labels, local_files_only=True
            )
        except Exception:
            try:
                cfg = SegformerConfig.from_pretrained(model_name, num_labels=num_labels)
            except Exception:
                cfg = SegformerConfig(num_labels=num_labels)

        self.model = SegformerForSemanticSegmentation(cfg)
        self.model.load_state_dict(clean_sd, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_hf = True

    def _init_smp_model(self, state_dict: dict, encoder_name: str) -> None:
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean_state_dict[k.replace("model.", "", 1)] = v
            else:
                clean_state_dict[k] = v

        self.model.load_state_dict(clean_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_hf = False

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the lesion region given an image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            Binary mask as numpy array (H, W), uint8, values 0 or 1.
        """
        H, W = image.shape[:2]
        self.last_used_recovery = False

        # 1. Preprocess image to input_size
        img_resized = sk_transform.resize(
            image, (self.input_size, self.input_size),
            order=3, preserve_range=True, anti_aliasing=True,
        ).astype(np.float32)

        # Normalize to [0, 1] using standard ImageNet mean and std as smp often expects
        img_norm = img_resized / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std

        # Convert to tensor (1, 3, H, W)
        img_tensor = (
            torch.tensor(img_norm, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        # 2. Generate mask
        if getattr(self, "is_hf", False):
            outputs = self.model(img_tensor)
            logits = outputs.logits  # (1, num_labels, H/4, W/4)
            if logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)
                low_res_mask = probs[0, 1].cpu().numpy()
            else:
                low_res_mask = torch.sigmoid(logits[0, 0]).cpu().numpy()
        else:
            logits = self.model(img_tensor)
            # SegFormer outputs logits at 1/4 resolution, smp automatically upsamples to input_size
            low_res_mask = torch.sigmoid(logits[0, 0]).cpu().numpy()

        mask_full = sk_transform.resize(
            low_res_mask, (H, W),
            order=1, preserve_range=True, anti_aliasing=False,
        )
        self.last_probability_map = mask_full.astype(np.float32, copy=False)

        # 3. Binarize
        binary_mask = (mask_full > 0.5).astype(np.uint8)

        # Morphological closing to fill small holes
        if self.closing_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.closing_kernel, self.closing_kernel),
            )
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Dermatoscope images often have a black circular vignette.  On some
        # low-contrast lesions the network mistakes the complete illuminated
        # field for foreground.  Passing that mask downstream makes the
        # border/background look like a lesion, so recover from this specific
        # failure mode using an image-only, centre-biased mask.  The recovery
        # is deliberately conservative and is only used for a dominant,
        # border-touching prediction; ordinary model masks are unchanged.
        if self._is_dominant_border_mask(binary_mask):
            recovered = self._recover_vignetted_mask(image)
            if recovered is not None:
                binary_mask = recovered
                self.last_used_recovery = True

        return binary_mask

    @staticmethod
    def _is_dominant_border_mask(mask: np.ndarray) -> bool:
        """Return True for the common full dermoscope-field false positive."""
        area_ratio = float(np.count_nonzero(mask)) / max(mask.size, 1)
        if area_ratio < 0.75:
            return False
        height, width = mask.shape
        border_width = max(1, int(round(min(height, width) * 0.015)))
        border = np.concatenate((
            mask[:border_width, :].ravel(), mask[-border_width:, :].ravel(),
            mask[:, :border_width].ravel(), mask[:, -border_width:].ravel(),
        ))
        return float(np.mean(border > 0)) > 0.25

    @staticmethod
    def _recover_vignetted_mask(image: np.ndarray) -> np.ndarray | None:
        """Estimate a bounded lesion mask when SegFormer captures the field.

        This is a safety fallback, not a replacement segmentation model.  It
        uses a circular field-of-view prior and robust darkness/chroma scores,
        then keeps only an interior component near the image centre.  Returning
        ``None`` leaves the normal engine validation in charge of rejecting the
        image when no plausible lesion can be found.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            return None
        height, width = image.shape[:2]
        yy, xx = np.ogrid[:height, :width]
        cx, cy = width / 2.0, height / 2.0
        radius = min(height, width) * 0.46
        field = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        pixels = np.column_stack((gray[field], lab[:, :, 1][field], lab[:, :, 2][field]))
        centre = np.median(pixels, axis=0)
        spread = np.maximum(np.percentile(pixels, 75, axis=0) - np.percentile(pixels, 25, axis=0), 1.0)
        # Darkness is the strongest cue; chroma distance helps separate brown
        # pigment from the pink skin background under uneven illumination.
        score = ((centre[0] - gray) / spread[0])
        score += 0.35 * np.abs(lab[:, :, 1] - centre[1]) / spread[1]
        score += 0.35 * np.abs(lab[:, :, 2] - centre[2]) / spread[2]
        score[~field] = -np.inf

        cutoff = float(np.percentile(score[field], 72.0))
        candidate = (score >= cutoff).astype(np.uint8)
        candidate[~field] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
        if count <= 1:
            return None
        candidates = []
        for component in range(1, count):
            x, y, w, h, area = stats[component]
            if area < max(100, int(height * width * 0.002)):
                continue
            if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
                continue
            distance = np.hypot(centroids[component][0] - cx, centroids[component][1] - cy)
            centrality = 1.0 - min(distance / max(radius, 1.0), 1.0)
            candidates.append((float(area) * (0.55 + 0.45 * centrality), component))
        if not candidates:
            return None
        _, selected = max(candidates, key=lambda item: item[0])
        recovered = (labels == selected).astype(np.uint8)
        ratio = float(recovered.sum()) / max(height * width, 1)
        if ratio < 0.002 or ratio > 0.75:
            return None
        return recovered
