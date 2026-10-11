"""Lesion measurements computed from the validated segmentation mask."""

import cv2
import numpy as np
from skimage.measure import regionprops, label
from typing import Optional


def _feret_diameters(contour: np.ndarray) -> tuple[float, float, tuple[float, float, float, float]]:
    """Return maximum/minimum Feret diameters and max-diameter endpoints."""
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
    if len(hull) > 2500:
        epsilon = max(0.5, 0.002 * cv2.arcLength(hull.astype(np.float32), True))
        hull = cv2.approxPolyDP(hull.astype(np.float32), epsilon, True).reshape(-1, 2).astype(np.float64)

    max_distance = -1.0
    max_pair = (hull[0], hull[1])
    for start in range(0, len(hull), 512):
        points = hull[start:start + 512]
        distances = np.sum((points[:, None, :] - hull[None, :, :]) ** 2, axis=2)
        local_index = np.unravel_index(int(np.argmax(distances)), distances.shape)
        local_distance = float(distances[local_index])
        if local_distance > max_distance:
            max_distance = local_distance
            max_pair = (points[local_index[0]], hull[local_index[1]])

    # Minimum width of a convex polygon is attained on a normal to an edge.
    min_width = float("inf")
    for index, point in enumerate(hull):
        edge = hull[(index + 1) % len(hull)] - point
        length = float(np.linalg.norm(edge))
        if length <= 1e-9:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        projections = hull @ normal
        min_width = min(min_width, float(projections.max() - projections.min()))
    if not np.isfinite(min_width):
        min_width = 0.0
    return (
        float(np.sqrt(max(max_distance, 0.0))),
        float(min_width),
        (float(max_pair[0][0]), float(max_pair[0][1]),
         float(max_pair[1][0]), float(max_pair[1][1])),
    )


def _project_contour_to_mm(
    contour: np.ndarray, pixel_to_mm_homography: np.ndarray,
) -> np.ndarray:
    """Map an image-space contour to its calibrated planar coordinates."""
    homography = np.asarray(pixel_to_mm_homography, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("pixel_to_mm_homography must be a finite 3x3 matrix")
    points_mm = cv2.perspectiveTransform(
        np.asarray(contour, dtype=np.float32).reshape(1, -1, 2), homography
    )[0]
    if not np.isfinite(points_mm).all() or len(points_mm) < 3:
        raise ValueError("homography produced invalid physical contour coordinates")
    return points_mm.reshape(-1, 1, 2).astype(np.float32)


def refine_lesion_mask(
    mask: np.ndarray,
    max_area_ratio: float = 0.90,
    excluded_bbox: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Keep the most plausible interior lesion component.

    Segmentation models can occasionally include a ruler, coin, or the image
    border. Measuring that mask would return the scale/background instead of
    the lesion. Border-touching components and implausibly large components
    are rejected; the remaining component closest to the image centre is
    preferred.  When a calibrated reference object's bounding box is known,
    its component is excluded explicitly before lesion selection.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or binary.sum() == 0:
        raise ValueError("empty lesion mask")

    height, width = binary.shape
    image_area = float(height * width)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    image_cx, image_cy = width / 2.0, height / 2.0
    max_distance = float(np.hypot(image_cx, image_cy))

    for component in range(1, count):
        x, y, w, h, area = stats[component]
        if area < max(25, image_area * 0.0002) or area / image_area >= max_area_ratio:
            continue
        if excluded_bbox is not None:
            ref_x, ref_y, ref_w, ref_h = excluded_bbox
            overlap_x = max(0, min(x + w, ref_x + ref_w) - max(x, ref_x))
            overlap_y = max(0, min(y + h, ref_y + ref_h) - max(y, ref_y))
            overlap_area = overlap_x * overlap_y
            centroid_x, centroid_y = centroids[component]
            if (
                overlap_area / max(float(area), 1.0) >= 0.20
                or (ref_x <= centroid_x <= ref_x + ref_w
                    and ref_y <= centroid_y <= ref_y + ref_h)
            ):
                continue
        touches_border = x <= 0 or y <= 0 or x + w >= width or y + h >= height
        if touches_border:
            continue
        cx, cy = centroids[component]
        centre_score = 1.0 - min(float(np.hypot(cx - image_cx, cy - image_cy)) / max_distance, 1.0)
        compactness = min(float(area) / max(float(w * h), 1.0), 1.0)
        score = np.log1p(float(area)) * (0.55 + 0.30 * centre_score + 0.15 * compactness)
        candidates.append((score, component))

    if not candidates:
        # Fallback: if all components touch the image boundary, allow the largest/most-centered non-excluded component
        for component in range(1, count):
            x, y, w, h, area = stats[component]
            if area < max(25, image_area * 0.0002) or area / image_area >= max_area_ratio:
                continue
            if excluded_bbox is not None:
                ref_x, ref_y, ref_w, ref_h = excluded_bbox
                overlap_x = max(0, min(x + w, ref_x + ref_w) - max(x, ref_x))
                overlap_y = max(0, min(y + h, ref_y + ref_h) - max(y, ref_y))
                overlap_area = overlap_x * overlap_y
                centroid_x, centroid_y = centroids[component]
                if (
                    overlap_area / max(float(area), 1.0) >= 0.20
                    or (ref_x <= centroid_x <= ref_x + ref_w and ref_y <= centroid_y <= ref_y + ref_h)
                ):
                    continue
            cx, cy = centroids[component]
            centre_score = 1.0 - min(float(np.hypot(cx - image_cx, cy - image_cy)) / max_distance, 1.0)
            compactness = min(float(area) / max(float(w * h), 1.0), 1.0)
            score = np.log1p(float(area)) * (0.55 + 0.30 * centre_score + 0.15 * compactness)
            candidates.append((score, component))

    if not candidates:
        raise ValueError("segmentation mask has no plausible interior lesion component")
    _, selected = max(candidates, key=lambda item: item[0])
    return (labels == selected).astype(np.uint8)


def refine_mask_with_scale(
    image: np.ndarray,
    mask: np.ndarray,
    pixels_per_mm: float,
    max_lesion_diameter_mm: float = 30.0,
) -> np.ndarray:
    """Remove a low-contrast halo/artifact shell from a calibrated mask.

    The segmentation model can connect a dark lesion to a broad brown halo or
    hair.  Candidate cores are derived from adaptive intensity percentiles in
    the supplied (hair-cleaned) image, then selected using relative geometry,
    centrality, compactness, and contrast against the remaining mask.  The
    original mask is retained unless the image provides clear evidence of a
    separate darker core, so irregular but legitimate lesion boundaries are
    not routinely eroded.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = np.asarray(gray, dtype=np.uint8)
    mask_pixels = gray[binary > 0]
    if len(mask_pixels) == 0:
        return mask

    # Robust spread prevents a fixed intensity threshold from being sensitive
    # to camera exposure or skin tone.
    median = float(np.median(mask_pixels))
    intensity_iqr = max(
        1.0,
        float(np.percentile(mask_pixels, 75) - np.percentile(mask_pixels, 25)),
    )
    raw_area = float(binary.sum())
    raw_centroid = np.mean(np.argwhere(binary > 0), axis=0)[::-1]
    raw_diagonal = max(float(np.hypot(w, h)), 1.0)
    candidates = []

    # Percentiles are relative to this lesion candidate, not image-wide
    # constants.  The low tail finds a pigmented core; higher tails preserve
    # larger boundaries when there is no convincing halo separation.
    for pct in (3, 5, 7, 10, 15, 20, 30, 40):
        cutoff = float(np.percentile(mask_pixels, pct))
        dark_mask = ((gray <= cutoff) & (binary > 0)).astype(np.uint8)

        # A small closing reconnects naturally ragged lesion pixels.  It is
        # intentionally relative to the candidate geometry rather than a
        # large fixed pixel radius.
        kernel_size = max(3, int(round(min(w, h) / 150.0)) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, 8)
        best_component = None
        best_component_score = -1.0
        for comp in range(1, count):
            cx_c, cy_c, cw, ch, area = stats[comp]
            area_fraction = float(area) / max(raw_area, 1.0)
            if area_fraction < 0.002:
                continue
            if cx_c <= 0 or cy_c <= 0 or cx_c + cw >= gray.shape[1] or cy_c + ch >= gray.shape[0]:
                continue
            cx_cent, cy_cent = centroids[comp]
            distance_score = 1.0 - min(
                float(np.hypot(cx_cent - raw_centroid[0], cy_cent - raw_centroid[1])) / raw_diagonal,
                1.0,
            )
            compactness = min(float(area) / max(float(cw * ch), 1.0), 1.0)
            component_score = float(area) * (0.55 + 0.25 * distance_score + 0.20 * compactness)
            if component_score > best_component_score:
                best_component_score = component_score
                best_component = (labels == comp).astype(np.uint8)

        if best_component is None:
            continue

        candidate_area = float(best_component.sum())
        candidate_fraction = candidate_area / max(raw_area, 1.0)
        if candidate_fraction >= 0.80:
            continue
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ring = cv2.dilate(best_component, ring_kernel, iterations=2).astype(bool)
        ring = (ring & (binary > 0) & (best_component == 0))
        if not np.any(ring):
            continue
        contrast = float(np.median(mask_pixels[ring[binary > 0]])) - float(np.median(gray[best_component > 0]))
        # Compare the core/shell separation with this candidate's own
        # intensity spread.  The broad halo can make the raw-mask MAD large,
        # so using it alone would incorrectly reject a real dark core.
        if contrast <= max(2.0, intensity_iqr * 0.20):
            continue
        component_area = int(best_component.sum())
        component_ys, component_xs = np.where(best_component > 0)
        if len(component_xs) == 0:
            continue
        component_height = int(component_ys.max() - component_ys.min() + 1)
        component_width = int(component_xs.max() - component_xs.min() + 1)
        compactness = min(component_area / max(float(component_width * component_height), 1.0), 1.0)
        candidate_median = float(np.median(gray[best_component > 0]))
        darkness = max(0.0, median - candidate_median)
        # Prefer a compact candidate that is both separated from its shell
        # and genuinely dark, instead of allowing a broad low-contrast tail
        # to win merely because it contains more pixels.
        score = (
            (contrast / intensity_iqr)
            * (1.0 + darkness / intensity_iqr)
            * (0.65 + 0.35 * compactness)
            * (candidate_fraction ** 0.15)
        )
        candidates.append((score, pct, best_component))

    if not candidates:
        return mask

    best_score, _, best_result = max(candidates, key=lambda item: item[0])
    # A very weak contrast gain is not enough evidence to replace the model
    # boundary.  This also protects uniformly dark or heavily pigmented
    # lesions from percentile-driven erosion.
    if best_score <= 0.20:
        return mask
    return best_result.astype(np.uint8)


def extract_lesion_measurements(
    image: np.ndarray,
    mask: np.ndarray,
    pixels_per_mm: Optional[float] = None,
    scale_confidence: Optional[float] = None,
    pixel_to_mm_homography: Optional[np.ndarray] = None,
) -> dict:
    """Return raw measurements; no clinical interpretation is performed here.

    Args:
        image:         RGB image array (H, W, 3).
        mask:          Binary segmentation mask (H, W).
        pixels_per_mm: Fallback local scale for physical measurements.
        pixel_to_mm_homography: Optional image-pixel-to-mm planar transform.
            When supplied, physical Feret diameter, perimeter, and area are
            computed after perspective correction rather than by dividing all
            image pixels by one global scale.

    Returns:
        Dict of measurement values.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    area_px = int(binary.sum())
    if area_px <= 0:
        raise ValueError("empty lesion mask")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("lesion contour unavailable")
    contour = max(contours, key=cv2.contourArea)
    perimeter_px = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    diameter_px, minimum_feret_px, feret_endpoints = _feret_diameters(contour)
    equivalent_diameter_px = float(np.sqrt(4.0 * area_px / np.pi))

    props = regionprops(label(binary))
    major_axis_px = None
    minor_axis_px = None
    if props:
        major_value = getattr(props[0], "axis_major_length", None)
        minor_value = getattr(props[0], "axis_minor_length", None)
        if major_value is None:
            major_value = props[0].major_axis_length
        if minor_value is None:
            minor_value = props[0].minor_axis_length
        major_axis_px = float(major_value)
        minor_axis_px = float(minor_value)
    circularity = float((4.0 * np.pi * area_px) / (perimeter_px ** 2)) if perimeter_px else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = float(np.clip(area_px / hull_area, 0.0, 1.0)) if hull_area > 0 else 0.0
    lesion_confidence = float(np.clip(
        0.55 + 0.25 * min(solidity, 1.0) + 0.20 * min(area_px / 1000.0, 1.0),
        0.0, 1.0,
    ))

    result = {
        "area_px": area_px,
        "perimeter_px": round(perimeter_px, 3),
        "diameter_px": round(diameter_px, 3),
        "lesion_diameter_pixels": round(diameter_px, 3),
        "max_feret_diameter_px": round(diameter_px, 3),
        "minimum_feret_diameter_px": round(minimum_feret_px, 3),
        "equivalent_diameter_px": round(equivalent_diameter_px, 3),
        "feret_endpoints_px": tuple(round(value, 3) for value in feret_endpoints),
        "bounding_box_width_px": int(width),
        "bounding_box_height_px": int(height),
        "lesion_region": {
            "x": int(x), "y": int(y), "width": int(width), "height": int(height),
        },
        "major_axis_length_px": round(major_axis_px, 3) if major_axis_px is not None else None,
        "minor_axis_length_px": round(minor_axis_px, 3) if minor_axis_px is not None else None,
        "circularity": round(min(circularity, 1.0), 5),
        "solidity": round(solidity, 5),
        "lesion_confidence": round(lesion_confidence, 4),
        "pixels_per_mm": None,
        "lesion_diameter_mm": None,
        "max_feret_diameter_mm": None,
        "minimum_feret_diameter_mm": None,
        "equivalent_diameter_mm": None,
        "measurement_method": "maximum_feret_diameter",
    }

    # A planar board/ruler calibration corrects perspective before measuring
    # the contour. This is more accurate than applying one pixels/mm value
    # across a lesion and a reference captured at different image positions.
    if pixel_to_mm_homography is not None:
        try:
            contour_mm = _project_contour_to_mm(contour, pixel_to_mm_homography)
            diameter_mm, minimum_feret_mm, feret_endpoints_mm = _feret_diameters(contour_mm)
            area_mm2 = float(cv2.contourArea(contour_mm))
            perimeter_mm = float(cv2.arcLength(contour_mm, True))
            equivalent_mm = float(np.sqrt(4.0 * area_mm2 / np.pi)) if area_mm2 > 0 else 0.0
            if not all(np.isfinite(value) and value >= 0 for value in (diameter_mm, minimum_feret_mm, area_mm2, perimeter_mm)):
                raise ValueError("non-finite physical contour measurement")
            result.update({
                "physical_scale_available": True,
                "diameter_mm": round(diameter_mm, 3),
                "lesion_diameter_mm": round(diameter_mm, 3),
                "max_feret_diameter_mm": round(diameter_mm, 3),
                "minimum_feret_diameter_mm": round(minimum_feret_mm, 3),
                "equivalent_diameter_mm": round(equivalent_mm, 3),
                "area_mm2": round(area_mm2, 2),
                "perimeter_mm": round(perimeter_mm, 2),
                "feret_endpoints_mm": tuple(round(float(value), 3) for value in feret_endpoints_mm),
                "measurement_method": "projective_maximum_feret_diameter",
                "physical_units_note": "Physical contour measurements use a planar-reference homography to correct perspective.",
            })
            if pixels_per_mm and pixels_per_mm > 0:
                result["pixels_per_mm"] = round(float(pixels_per_mm), 4)
            result["measurement_confidence"] = round(
                min(lesion_confidence, float(scale_confidence))
                if scale_confidence is not None else lesion_confidence, 4
            )
            return result
        except (ValueError, cv2.error):
            # A valid scalar calibration is still preferable to withholding a
            # measurement if a malformed transform was passed by a caller.
            pass

    # Physical measurements from a validated local scale when no planar
    # transform is available (e.g., a circular sticker or coin).
    if pixels_per_mm and pixels_per_mm > 0:
        ppm = pixels_per_mm
        result["physical_scale_available"] = True
        result["pixels_per_mm"] = round(float(ppm), 4)
        result["diameter_mm"] = round(diameter_px / ppm, 3)
        result["lesion_diameter_mm"] = round(diameter_px / ppm, 3)
        result["max_feret_diameter_mm"] = round(diameter_px / ppm, 3)
        result["minimum_feret_diameter_mm"] = round(minimum_feret_px / ppm, 3)
        result["equivalent_diameter_mm"] = round(equivalent_diameter_px / ppm, 3)
        result["area_mm2"] = round(area_px / (ppm ** 2), 2)
        result["perimeter_mm"] = round(perimeter_px / ppm, 2)
        if major_axis_px is not None:
            result["major_axis_length_mm"] = round(major_axis_px / ppm, 2)
        if minor_axis_px is not None:
            result["minor_axis_length_mm"] = round(minor_axis_px / ppm, 2)
        result["measurement_confidence"] = round(
            min(lesion_confidence, float(scale_confidence))
            if scale_confidence is not None else lesion_confidence, 4
        )
        result["physical_units_note"] = (
            "Physical measurements were computed using a reference-object "
            "calibration. Accuracy depends on the calibration quality."
        )
    else:
        result["physical_scale_available"] = False
        result["measurement_confidence"] = round(lesion_confidence, 4)
        result["physical_units_note"] = (
            "Physical size is unavailable because no calibrated reference "
            "scale was supplied."
        )

    return result
