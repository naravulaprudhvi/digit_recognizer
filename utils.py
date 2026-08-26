"""
utils.py
--------
Image preprocessing and segmentation utilities that extract individual digits
from a drawing or uploaded image (number sequence), center each digit into
the exact format the CNN was trained on (28x28 grayscale, white digit on black
background, centered by center-of-mass — same style as MNIST).

Author: Naravula Prudhvi Sri Bhanu Vivek
"""

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def _get_best_shift(img: np.ndarray):
    """Compute how far the digit's center of mass is from the image center."""
    cy, cx = ndimage.center_of_mass(img)
    if np.isnan(cy) or np.isnan(cx):
        return 0, 0
    rows, cols = img.shape
    shift_x = int(np.round(cols / 2.0 - cx))
    shift_y = int(np.round(rows / 2.0 - cy))
    return shift_x, shift_y


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


def preprocess_single_crop(cropped: np.ndarray) -> np.ndarray:
    """
    Turn a cropped bounding box of a single digit into a (1, 28, 28, 1) float32 array.
    """
    h, w = cropped.shape
    if h == 0 or w == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    # 1. Resize so the longer side is 20px (MNIST standard style)
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))

    cropped_img = Image.fromarray(cropped.astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    resized = np.array(cropped_img).astype(np.float32)

    # 2. Pad to 28x28 (centered)
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    # 3. Center of mass shift for precise MNIST alignment
    if canvas.max() > 0:
        shift_x, shift_y = _get_best_shift(canvas)
        canvas = _shift(canvas, shift_x, shift_y)

    # Clamp values & normalize to [0, 1]
    canvas = np.clip(canvas, 0.0, 255.0) / 255.0
    return canvas.reshape(1, 28, 28, 1).astype(np.float32)


def segment_and_preprocess_image(pil_image: Image.Image):
    """
    Segment an image into individual digit bounding boxes ordered Top-to-Bottom
    (line by line) and Left-to-Right within each line.

    Returns a list of dicts:
        [
            {
                "tensor": np.ndarray of shape (1, 28, 28, 1),
                "preview_pil": PIL.Image of shape (28, 28),
                "bbox": (x, y, w, h),
                "line_idx": int (0, 1, 2...)
            },
            ...
        ]
    """
    # 1. Convert to grayscale
    img = pil_image.convert("L")
    arr = np.array(img).astype(np.float32)

    # 2. Auto-detect polarity: MNIST is white digit on black background
    if arr.mean() > 127:
        arr = 255.0 - arr

    # 3. Threshold background noise with Otsu or soft fallback
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    if arr_uint8.max() < 20:
        return []

    _, binary = cv2.threshold(arr_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.max() == 0:
        return []

    H, W = binary.shape

    # 4. Light dilation to connect broken strokes within the SAME digit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out tiny noise specks
        if w >= 3 and h >= 6 and (w * h) >= 18:
            boxes.append([x, y, w, h])

    if not boxes:
        return []

    # 5. Merge boxes ONLY if they overlap significantly in BOTH X and Y (2D spatial overlap),
    # or if one box is completely contained inside another (e.g. multi-stroke digit parts).
    # THIS PREVENTS DIGITS ON DIFFERENT LINES FROM BEING MERGED TOGETHER!
    merged = True
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

                # Merge if significant 2D overlap or near complete vertical+horizontal inclusion
                is_overlap_2d = (
                    min_w > 0
                    and min_h > 0
                    and (overlap_x / float(min_w) > 0.3)
                    and (overlap_y / float(min_h) > 0.3)
                )
                is_contained = (
                    x1 <= x2 and y1 <= y2 and (x1 + w1) >= (x2 + w2) and (y1 + h1) >= (y2 + h2)
                ) or (
                    x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1)
                )

                if is_overlap_2d or is_contained:
                    x1 = min(x1, x2)
                    y1 = min(y1, y2)
                    w1 = max(boxes[i][0] + w1, x2 + w2) - x1
                    h1 = max(boxes[i][1] + h1, y2 + h2) - y1
                    skip.add(j)
                    merged = True
            new_boxes.append([x1, y1, w1, h1])
        boxes = new_boxes

    # 6. Group into distinct horizontal lines (Top-to-Bottom)
    # Sort boxes by top y coordinate initially
    boxes.sort(key=lambda b: b[1])

    lines = []
    for b in boxes:
        x, y, w, h = b
        cy = y + h / 2.0
        matched_line = None

        for line in lines:
            # Check vertical overlap with line bounds or vertical center distance
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
    padding = 3
    for (x, y, w, h), line_idx in sorted_boxes:
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(W, x + w + padding)
        y_max = min(H, y + h + padding)

        cropped = arr[y_min:y_max, x_min:x_max]
        tensor = preprocess_single_crop(cropped)
        preview_arr = (tensor[0, :, :, 0] * 255).astype(np.uint8)
        preview_pil = Image.fromarray(preview_arr)

        results.append({
            "tensor": tensor,
            "preview_pil": preview_pil,
            "bbox": (x, y, w, h),
            "line_idx": line_idx,
        })

    return results


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Backwards-compatible single-digit preprocessor.
    If the image contains digits, takes the first digit segment or fallback crop.
    """
    segments = segment_and_preprocess_image(pil_image)
    if segments:
        return segments[0]["tensor"]

    return np.zeros((1, 28, 28, 1), dtype=np.float32)


def pil_from_canvas_rgba(rgba_array: np.ndarray) -> Image.Image:
    """
    Flatten Streamlit canvas RGBA array onto a black background and return PIL Image.
    """
    rgba = rgba_array.astype(np.uint8)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0
    black_bg = np.zeros_like(rgb)
    composited = (rgb * alpha + black_bg * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(composited)

