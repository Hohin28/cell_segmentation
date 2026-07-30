"""
Cell Analysis Module for Cellpose
==================================
Provides 4 enhanced capabilities on top of Cellpose segmentation masks:
  1. Cell classification  - morphology-based (WBC, RBC, Pap smear types, Tumour)
  2. Boundary extraction  - rich contour polygons with thickness
  3. Cell count           - simple integer count from masks
  4. Overlap separation   - watershed-based splitting of merged masks

All functions operate purely on the output of CellposeModel.eval()
so no additional model weights are required.
"""

import logging
import numpy as np
import cv2
from scipy.ndimage import find_objects, distance_transform_edt, label as scipy_label
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------
# Each mode maps cell-feature-vector → class label + colour (R,G,B)
# Features per cell: [area_px, compactness, mean_intensity_norm, aspect_ratio]
#
# Colours are used to tint the overlay in the GUI.

_CLASS_COLOURS = {
    # Blood smear
    "RBC":          (220, 80,  80),
    "WBC":          (80,  140, 220),
    "Platelet":     (220, 200, 60),
    # Pap smear
    "Superficial":  (80,  200, 120),
    "Intermediate": (200, 160, 80),
    "Parabasal":    (200, 80,  200),
    "Endocervical": (60,  180, 200),
    "Abnormal":     (240, 60,  60),
    # Histopathology
    "Tumour":       (220, 80,  80),
    "Lymphocyte":   (80,  140, 220),
    "Stromal":      (140, 200, 100),
    "Epithelial":   (220, 180, 60),
    # Generic fallback
    "Cell":         (160, 160, 160),
    "Unknown":      (100, 100, 100),
}


def _cell_features(masks, image, cell_id, slc):
    """
    Return (area_px, compactness, mean_intensity_norm, aspect_ratio, std_intensity)
    for a single cell given its label and bounding-box slice.
    """
    roi_mask  = masks[slc] == cell_id           # bool patch
    area      = int(roi_mask.sum())
    if area == 0:
        return None

    # contour-based perimeter
    roi_u8   = roi_mask.astype(np.uint8)
    contours, _ = cv2.findContours(roi_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    perimeter = cv2.arcLength(contours[0], closed=True)
    compactness = (4 * np.pi * area / (perimeter**2 + 1e-6))

    # bounding-box aspect ratio
    y0, x0 = slc[0].start, slc[1].start
    y1, x1 = slc[0].stop,  slc[1].stop
    h, w   = y1 - y0,  x1 - x0
    aspect  = min(h, w) / (max(h, w) + 1e-6)

    # intensity statistics (works with gray or RGB)
    if image is not None:
        patch = image[slc]
        if patch.ndim == 3:               # RGB → gray
            patch = patch.mean(axis=-1)
        masked_vals = patch[roi_mask]
        mean_int    = masked_vals.mean() / 255.0
        std_int     = masked_vals.std()  / 255.0
    else:
        mean_int = 0.5
        std_int  = 0.0

    return area, compactness, mean_int, aspect, std_int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_cells(masks):
    """Return the total number of detected cells.

    Args:
        masks (np.ndarray): 2-D labelled mask array (0 = background, 1…N = cells).

    Returns:
        int: Number of distinct cells.
    """
    return int(masks.max())


def extract_boundaries(masks, thickness=2):
    """Extract per-cell boundary contours.

    Args:
        masks (np.ndarray): 2-D labelled mask array.
        thickness (int): Desired boundary line thickness in pixels.

    Returns:
        list[dict]: One dict per cell with keys:
            - ``cell_id``   (int)
            - ``contour``   (np.ndarray, shape Nx2, XY coordinates)
            - ``centroid``  (tuple[float, float])  Y, X
            - ``area``      (int)
            - ``perimeter`` (float)
    """
    results = []
    slices  = find_objects(masks)
    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        cell_id = idx + 1
        roi     = (masks[slc] == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c    = max(contours, key=lambda x: x.shape[0])
        pts  = c.squeeze(axis=1)   # (N, 2)  XY
        # offset to full-image coords
        pts[:, 0] += slc[1].start
        pts[:, 1] += slc[0].start

        area      = int(roi.sum())
        perimeter = cv2.arcLength(c, closed=True)
        ys, xs    = np.nonzero(roi)
        centroid  = (ys.mean() + slc[0].start, xs.mean() + slc[1].start)

        results.append({
            "cell_id":   cell_id,
            "contour":   pts,
            "centroid":  centroid,
            "area":      area,
            "perimeter": float(perimeter),
        })
    return results


def classify_cells(masks, image, mode="auto"):
    """
    Classify every cell in *masks* using morphological features.

    Args:
        masks (np.ndarray): 2-D labelled mask array (uint16).
        image (np.ndarray): Original image (H x W x C) or (H x W), uint8.
            Used for intensity-based features.
        mode  (str):
            ``"auto"``         — heuristic auto-detection of image domain.
            ``"blood"``        — WBC / RBC / Platelet.
            ``"papsmear"``     — Pap smear cell types.
            ``"histology"``    — Tumour / Lymphocyte / Stromal / Epithelial.
            ``"generic"``      — just labels everything "Cell".

    Returns:
        dict: Mapping ``cell_id (int)`` → classification dict with keys:
            - ``label``   (str)   class name
            - ``colour``  (tuple) RGB colour for overlay
            - ``area``    (int)   cell area in pixels
            - ``compactness`` (float) 0–1
            - ``mean_intensity`` (float) 0–1
    """
    n_cells = int(masks.max())
    if n_cells == 0:
        return {}

    if image is not None and image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # collect raw features
    feature_list = []
    slices = find_objects(masks)
    for idx, slc in enumerate(slices):
        if slc is None:
            feature_list.append(None)
            continue
        feat = _cell_features(masks, image, idx + 1, slc)
        feature_list.append(feat)

    # auto-detect mode from feature statistics
    if mode == "auto":
        mode = _auto_detect_mode(feature_list)
        logger.info(f"cell_analysis: auto-detected image mode = '{mode}'")

    # classify
    results = {}
    for idx, feat in enumerate(feature_list):
        cell_id = idx + 1
        if feat is None:
            results[cell_id] = {
                "label": "Unknown", "colour": _CLASS_COLOURS["Unknown"],
                "area": 0, "compactness": 0.0, "mean_intensity": 0.0
            }
            continue
        area, compactness, mean_int, aspect, std_int = feat
        label = _classify_one(area, compactness, mean_int, aspect, std_int, mode,
                               median_area=_median_area(feature_list))
        results[cell_id] = {
            "label":          label,
            "colour":         _CLASS_COLOURS.get(label, _CLASS_COLOURS["Cell"]),
            "area":           area,
            "compactness":    round(float(compactness), 3),
            "mean_intensity": round(float(mean_int), 3),
        }
    return results


def _median_area(feature_list):
    areas = [f[0] for f in feature_list if f is not None]
    return float(np.median(areas)) if areas else 1.0


def _auto_detect_mode(feature_list):
    """
    Guess image domain from shape / size distribution.

    Heuristic:
      - Very bimodal size (ratio > 3 between large & small) → blood smear
        (RBC + WBC separation is large).
      - High compactness overall (>0.8 mean) → papsmear (round cells).
      - Otherwise → histology.
    """
    valid = [f for f in feature_list if f is not None]
    if not valid:
        return "generic"
    areas        = np.array([f[0] for f in valid])
    compactness  = np.array([f[1] for f in valid])
    ratio = areas.max() / (areas.min() + 1)
    if ratio > 5 and compactness.mean() > 0.65:
        return "blood"
    if compactness.mean() > 0.80:
        return "papsmear"
    return "histology"


def _classify_one(area, compactness, mean_int, aspect, std_int, mode, median_area=100):
    """Apply rule-based classification for a single cell."""

    if mode == "generic":
        return "Cell"

    if mode == "blood":
        # Platelets: very small
        if area < median_area * 0.15:
            return "Platelet"
        # RBC: small, high compactness (biconcave disc), lighter centre
        if compactness > 0.72 and area < median_area * 1.4:
            return "RBC"
        # WBC: larger, lower compactness (irregular nucleus)
        return "WBC"

    if mode == "papsmear":
        # Superficial: large, very round, bright cytoplasm
        if area > median_area * 1.3 and compactness > 0.80:
            return "Superficial"
        # Intermediate: medium, round
        if median_area * 0.6 < area <= median_area * 1.3 and compactness > 0.70:
            return "Intermediate"
        # Parabasal: small, round
        if area <= median_area * 0.6 and compactness > 0.72:
            return "Parabasal"
        # Endocervical: elongated (low aspect ratio)
        if aspect < 0.55:
            return "Endocervical"
        # Abnormal: irregular (low compactness), or very dark
        if compactness < 0.60 or mean_int < 0.25:
            return "Abnormal"
        return "Intermediate"

    if mode == "histology":
        # Lymphocytes: small, round, dark nucleus
        if area < median_area * 0.5 and compactness > 0.75:
            return "Lymphocyte"
        # Tumour: large, irregular, variable intensity
        if area > median_area * 1.5 and compactness < 0.72:
            return "Tumour"
        # Stromal: medium, elongated
        if aspect < 0.55:
            return "Stromal"
        # Epithelial: medium, round
        return "Epithelial"

    return "Cell"


# ---------------------------------------------------------------------------
# Overlap / merge separation
# ---------------------------------------------------------------------------

def separate_overlapping_cells(masks, min_size=15, area_ratio_threshold=2.2,
                                compactness_threshold=0.62):
    """
    Detect and split likely merged (overlapping) cell masks using watershed.

    A mask is a candidate for splitting when it is:
      - larger than ``area_ratio_threshold`` × median cell area, AND
      - has compactness below ``compactness_threshold``.

    Args:
        masks (np.ndarray): 2-D labelled uint16 mask array.
        min_size (int): Minimum pixels for a resulting sub-mask to be kept.
        area_ratio_threshold (float): Size multiplier above median to flag candidates.
        compactness_threshold (float): Compactness below this triggers splitting.

    Returns:
        np.ndarray: Updated mask array with split masks relabelled.
    """
    import fastremap

    masks = masks.copy()
    slices = find_objects(masks)
    n_cells = int(masks.max())
    if n_cells == 0:
        return masks

    areas = np.array([
        int((masks[slc] == (i + 1)).sum()) if slc is not None else 0
        for i, slc in enumerate(slices)
    ])
    median_area = float(np.median(areas[areas > 0])) if (areas > 0).any() else 1.0

    next_label = n_cells + 1
    changed = False

    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        cell_id = idx + 1
        roi = (masks[slc] == cell_id).astype(np.uint8)
        area = int(roi.sum())

        if area < area_ratio_threshold * median_area:
            continue

        # compute compactness
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], closed=True)
        compactness = 4 * np.pi * area / (perimeter**2 + 1e-6)

        if compactness >= compactness_threshold:
            continue  # round enough → single cell, skip

        # --- watershed split ---
        dist  = distance_transform_edt(roi)
        dist  = gaussian_filter(dist, sigma=1.5)
        # find local maxima as seeds
        from scipy.ndimage import maximum_filter, label as sci_label
        local_max = (dist == maximum_filter(dist, size=7)) & (roi > 0)
        seeds, n_seeds = sci_label(local_max)
        if n_seeds < 2:
            continue  # cannot split

        # use opencv watershed
        roi_rgb = cv2.cvtColor(roi * 255, cv2.COLOR_GRAY2BGR)
        markers = seeds.astype(np.int32)
        cv2.watershed(roi_rgb, markers)

        # remap sub-masks back to full image
        for sub_id in range(1, n_seeds + 1):
            sub_mask = markers == sub_id
            if sub_mask.sum() < min_size:
                continue
            # first sub-mask reuses current cell_id, rest get new labels
            if sub_id == 1:
                masks[slc][sub_mask] = cell_id
                # clear pixels not in this sub-mask within the original roi
                masks[slc][(roi > 0) & ~sub_mask & (masks[slc] == cell_id)] = 0
            else:
                masks[slc][sub_mask] = next_label
                next_label += 1
                changed = True

    if changed:
        fastremap.renumber(masks, in_place=True)
        logger.info(f"separate_overlapping_cells: relabelled masks, new count = {int(masks.max())}")

    return masks


# ---------------------------------------------------------------------------
# Overlay drawing helper  (for export / saving annotated images)
# ---------------------------------------------------------------------------

def draw_analysis_overlay(image, masks, classification, boundaries,
                          show_labels=True, alpha=0.35):
    """
    Render classification results onto the image.

    Args:
        image (np.ndarray): H x W x 3 uint8 image.
        masks (np.ndarray): 2-D labelled mask array.
        classification (dict): Output of :func:`classify_cells`.
        boundaries (list): Output of :func:`extract_boundaries`.
        show_labels (bool): Draw text labels at cell centroids.
        alpha (float): Mask fill opacity.

    Returns:
        np.ndarray: Annotated H x W x 3 uint8 image.
    """
    overlay = image.copy().astype(np.float32)
    result  = image.copy()

    for cell_id, info in classification.items():
        colour = np.array(info["colour"], dtype=np.float32)
        cell_mask = masks == cell_id
        overlay[cell_mask] = colour

    # blend
    result = (alpha * overlay + (1 - alpha) * result.astype(np.float32)).astype(np.uint8)

    # draw contours
    bnd_map = {b["cell_id"]: b for b in boundaries}
    for cell_id, info in classification.items():
        colour_bgr = info["colour"][::-1]   # RGB → BGR for cv2
        if cell_id in bnd_map:
            pts = bnd_map[cell_id]["contour"]
            pts_reshaped = pts[:, np.newaxis, :]
            # swap XY → YX for cv2 drawContours (it expects X in dim0, Y in dim1?)
            # pts from extract_boundaries are XY, cv2 wants (x,y) → already correct
            cv2.polylines(result, [pts_reshaped], isClosed=True,
                          color=colour_bgr, thickness=2)

        if show_labels and cell_id in bnd_map:
            cy, cx = bnd_map[cell_id]["centroid"]
            label  = info["label"]
            cv2.putText(result, label[:3],
                        (int(cx) - 8, int(cy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        colour_bgr, 1, cv2.LINE_AA)

    return result
