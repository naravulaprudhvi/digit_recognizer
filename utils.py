"""
utils.py
--------
Enhanced computer vision preprocessing and digit segmentation pipeline.

Key Capabilities:
1. Robust Polarity & Background Filtering (edge sampling, Otsu thresholding, bilateral denoising)
2. Multi-Stroke Component Merging (merges top bar and hook of 5, crossbars of 4 and 7)
3. Connected Digit Splitting (vertical projection profile valley analysis for touching digits)
4. MNIST Tight-Foreground Normalization (scales tight active pixels to 20x20 box inside 28x28)
5. Bounded Center-of-Mass Alignment (MNIST standard alignment without clipping)
6. Multi-Scale Tensor Generation (for Test-Time Augmentation in app.py)

Author: Naravula Prudhvi Sri Bhanu Vivek
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def _get_best_shift(img: np.ndarray):
    """
    Compute how far the digit's center of mass is from the image center (14, 14),
    bounded to [-4, 4] to prevent pushing strokes out of bounds.
    """
    cy, cx = ndimage.center_of_mass(img)
    if np.isnan(cy) or np.isnan(cx):
        return 0, 0
    rows, cols = img.shape
    shift_x = int(np.round(cols / 2.0 - cx))
    shift_y = int(np.round(rows / 2.0 - cy))
    return int(np.clip(shift_x, -4, 4)), int(np.clip(shift_y, -4, 4))


def _shift(img: np.ndarray, sx: int, sy: int) -> np.ndarray:
    """Shift numpy array by (sx, sy) using OpenCV affine transform."""
    if sx == 0 and sy == 0:
        return img
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (cols, rows),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return shifted


def preprocess_single_crop(cropped: np.ndarray, target_dim: float = 20.0) -> np.ndarray:
    """
    Turn a cropped bounding box of a single digit into a (1, 28, 28, 1) float32 array.
    Finds tight foreground pixels, scales preserving aspect ratio so the longer side
    fits into `target_dim` pixels, symmetrically pads into 28x28, and aligns center-of-mass.
    """
    # 1. Extract tight bounding box of active foreground pixels (> 25)
    coords = np.argwhere(cropped > 25)
    if coords.size == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    tight = cropped[y0:y1, x0:x1]

    h, w = tight.shape
    if h == 0 or w == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    # 2. Resize so longer side is target_dim (MNIST standard is 20px)
    if h > w:
        new_h = int(round(target_dim))
        new_w = max(1, int(round(w * (target_dim / float(h)))))
    else:
        new_w = int(round(target_dim))
        new_h = max(1, int(round(h * (target_dim / float(w)))))

    cropped_img = Image.fromarray(tight.astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    resized = np.array(cropped_img).astype(np.float32)

    # 3. Pad into 28x28 (centered)
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    # 4. Center of mass shift for precise MNIST alignment
    if canvas.max() > 0:
        shift_x, shift_y = _get_best_shift(canvas)
        canvas = _shift(canvas, shift_x, shift_y)

    # Clamp values & normalize to [0, 1]
    canvas = np.clip(canvas, 0.0, 255.0) / 255.0
    return canvas.reshape(1, 28, 28, 1).astype(np.float32)


def split_connected_box(box, binary_img):
    """
    Inspect a wide bounding box for touching digits using vertical projection profile.
    If a significant valley exists in the middle region, split into two boxes.
    """
    x, y, w, h = box
    # Only split if box is significantly wider than a single digit (w/h >= 1.15)
    # and wide enough to contain 2 valid digits (w >= 36)
    if (w / float(h) < 1.15) or w < 36:
        return [box]

    crop_bin = binary_img[y : y + h, x : x + w]
    profile = np.sum(crop_bin > 0, axis=0)

    # Search between 35% and 65% of width, preserving at least 18px on both sides
    mid_start = max(18, int(w * 0.35))
    mid_end = min(w - 18, int(w * 0.65))
    if mid_end <= mid_start:
        return [box]

    sub_profile = profile[mid_start:mid_end]
    min_idx_rel = int(np.argmin(sub_profile))
    min_val = sub_profile[min_idx_rel]
    split_x = mid_start + min_idx_rel

    peak_left = np.max(profile[:split_x]) if split_x > 0 else 1
    peak_right = np.max(profile[split_x:]) if split_x < w else 1
    min_peak = min(peak_left, peak_right)

    # Strict valley ratio: must be a real bridge (< 0.40 of peak)
    if min_peak > 0 and (min_val / float(min_peak) <= 0.40):
        box1 = [x, y, split_x, h]
        box2 = [x + split_x, y, w - split_x, h]
        return [box1, box2]

    return [box]


def segment_and_preprocess_image(pil_image: Image.Image):
    """
    Segment an image into individual digit bounding boxes ordered Top-to-Bottom
    (line by line) and Left-to-Right within each line.
    """
    # 1. Convert to grayscale
    img = pil_image.convert("L")
    arr = np.array(img).astype(np.float32)
    H, W = arr.shape

    if H < 10 or W < 10:
        return []

    # 2. Robust polarity detection (sample borders to determine true background)
    edge_pixels = np.concatenate([
        arr[:5, :].ravel(),
        arr[-5:, :].ravel(),
        arr[:, :5].ravel(),
        arr[:, -5:].ravel(),
    ])
    if np.median(edge_pixels) > 120:
        arr = 255.0 - arr

    # 3. Clean background noise with Gaussian blur and Otsu thresholding
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    if arr_uint8.max() < 25:
        return []

    blurred = cv2.GaussianBlur(arr_uint8, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.max() == 0:
        return []

    # Cleaned image: zero out background artifacts
    cleaned_arr = np.where(binary > 0, arr_uint8, 0).astype(np.float32)

    # 4. Light dilation for contour finding
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out tiny specks
        if w >= 3 and h >= 6 and (w * h) >= 16:
            raw_boxes.append([x, y, w, h])

    if not raw_boxes:
        return []

    # 5. Connected digit splitting for touching numbers (e.g. "2026", "42")
    split_boxes = []
    for b in raw_boxes:
        split_boxes.extend(split_connected_box(b, binary))

    # 6. Multi-stroke component merging:
    # Merges disconnected strokes of the SAME digit (e.g., top bar of 5, crossbar of 4 or 7)
    merged = True
    boxes = split_boxes
    while merged:
        merged = False
        new_boxes = []
        skip = set()
        for i in range(len(boxes)):
            if i in skip:
                continue
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in skip:
                    continue
                x2, y2, w2, h2 = boxes[j]

                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                min_w = min(w1, w2)
                min_h = min(h1, h2)
                max_h = max(h1, h2)

                # Vertical gap between two components
                y_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                center_x_dist = abs((x1 + w1 / 2.0) - (x2 + w2 / 2.0))

                # Case A: 2D spatial overlap
                is_overlap_2d = (
                    min_w > 0
                    and min_h > 0
                    and (overlap_x / float(min_w) > 0.35)
                    and (overlap_y / float(min_h) > 0.25)
                )

                # Case B: Vertical stacked strokes (e.g. top bar of 5, or dot/crossbar)
                is_vertical_part = (
                    min_w > 0
                    and (overlap_x / float(min_w) > 0.45 or center_x_dist < min_w * 0.5)
                    and y_gap <= (max_h * 0.35)
                )

                # Case C: Complete inclusion
                is_contained = (
                    x1 <= x2 and y1 <= y2 and (x1 + w1) >= (x2 + w2) and (y1 + h1) >= (y2 + h2)
                ) or (
                    x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1)
                )

                if is_overlap_2d or is_vertical_part or is_contained:
                    x1 = min(x1, x2)
                    y1 = min(y1, y2)
                    w1 = max(boxes[i][0] + w1, x2 + w2) - x1
                    h1 = max(boxes[i][1] + h1, y2 + h2) - y1
                    skip.add(j)
                    merged = True

            new_boxes.append([x1, y1, w1, h1])
        boxes = new_boxes

    # 7. Group into distinct horizontal lines (Top-to-Bottom)
    boxes.sort(key=lambda b: b[1])
    lines = []
    for b in boxes:
        x, y, w, h = b
        cy = y + h / 2.0
        matched_line = None

        for line in lines:
            ly_min = line["y_min"]
            ly_max = line["y_max"]
            overlap_y = max(0, min(y + h, ly_max) - max(y, ly_min))
            line_avg_h = line["avg_h"]
            min_h = min(h, line_avg_h)

            if (min_h > 0 and overlap_y / float(min_h) > 0.35) or abs(cy - line["cy_avg"]) < (0.45 * line_avg_h):
                matched_line = line
                break

        if matched_line is not None:
            matched_line["boxes"].append(b)
            matched_line["y_min"] = min(matched_line["y_min"], y)
            matched_line["y_max"] = max(matched_line["y_max"], y + h)
            matched_line["cy_avg"] = np.mean([box[1] + box[3] / 2.0 for box in matched_line["boxes"]])
            matched_line["avg_h"] = np.mean([box[3] for box in matched_line["boxes"]])
        else:
            lines.append({
                "y_min": y,
                "y_max": y + h,
                "cy_avg": cy,
                "avg_h": float(h),
                "boxes": [b],
            })

    # Sort lines Top-to-Bottom by cy_avg
    lines.sort(key=lambda l: l["cy_avg"])

    # Sort digits Left-to-Right within each line
    sorted_boxes = []
    for line_idx, line in enumerate(lines):
        line["boxes"].sort(key=lambda b: b[0])
        for b in line["boxes"]:
            sorted_boxes.append((b, line_idx))

    results = []
    for (x, y, w, h), line_idx in sorted_boxes:
        # Tightly cropped segment directly from cleaned array
        cropped = cleaned_arr[y : y + h, x : x + w]
        if cropped.size == 0 or cropped.max() == 0:
            continue

        # Standard scale (20px)
        tensor_std = preprocess_single_crop(cropped, target_dim=20.0)
        # Scaled variation (18px) for Test-Time Augmentation
        tensor_scaled = preprocess_single_crop(cropped, target_dim=18.0)

        preview_arr = (tensor_std[0, :, :, 0] * 255).astype(np.uint8)
        preview_pil = Image.fromarray(preview_arr)

        results.append({
            "tensor": tensor_std,
            "tensor_scaled": tensor_scaled,
            "preview_pil": preview_pil,
            "bbox": (int(x), int(y), int(w), int(h)),
            "line_idx": int(line_idx),
        })

    return results


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Backwards-compatible single-digit preprocessor."""
    segments = segment_and_preprocess_image(pil_image)
    if segments:
        return segments[0]["tensor"]
    return np.zeros((1, 28, 28, 1), dtype=np.float32)


def pil_from_canvas_rgba(rgba_array: np.ndarray) -> Image.Image:
    """Flatten canvas RGBA array onto a black background and return PIL Image."""
    rgba = rgba_array.astype(np.uint8)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0
    black_bg = np.zeros_like(rgb)
    composited = (rgb * alpha + black_bg * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(composited)


def pil_from_data_url(data_url: str) -> Image.Image:
    """Convert base64 data URL (e.g. from HTML5 canvas) to PIL Image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    return Image.open(io.BytesIO(image_bytes))


def pil_to_data_url(pil_img: Image.Image) -> str:
    """Convert a PIL Image to a base64 PNG data URL string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
