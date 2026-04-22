from pathlib import Path

import cv2
import numpy as np


# Tunable HSV range for yellow lane markers.
LOWER_YELLOW = np.array([12, 35, 75], dtype=np.uint8)
UPPER_YELLOW = np.array([45, 255, 255], dtype=np.uint8)


def _build_center_roi(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [[
            (int(0.36 * width), height - 1),   # bottom-left
            (int(0.64 * width), height - 1),   # bottom-right
            (int(0.53 * width), int(0.30 * height)),  # top-right
            (int(0.47 * width), int(0.30 * height)),  # top-left
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    return mask


def _threshold_yellow(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)


def _select_center_components(
    binary_mask: np.ndarray, image_shape: tuple[int, int, int]
) -> np.ndarray:
    height, width = image_shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    selected = np.zeros_like(binary_mask)
    image_center_x = width / 2.0

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]

        if area < 5 or area > 12000:
            continue
        if cy < 0.18 * height:
            continue
        if abs(cx - image_center_x) > 0.16 * width:
            continue
        if w < 2 or h < 2:
            continue
        if w < 0.45 * h:
            continue

        selected[labels == label] = 255

    return selected


def _expand_and_clean(mask: np.ndarray) -> np.ndarray:
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    shrunk = cv2.erode(mask, erode_kernel, iterations=1)
    cleaned = cv2.dilate(shrunk, dilate_kernel, iterations=1)

    return cleaned


def detect_lane_markers(image: np.ndarray, fill_overlay: bool = True) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    output = image.copy()
    height, width = image.shape[:2]

    roi_mask = _build_center_roi(height, width)
    yellow_mask = _threshold_yellow(image)
    masked = cv2.bitwise_and(yellow_mask, roi_mask)
    selected = _select_center_components(masked, image.shape)
    expanded = _expand_and_clean(selected)

    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if fill_overlay:
        overlay = output.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
        output = cv2.addWeighted(overlay, 0.30, output, 0.70, 0)

    cv2.drawContours(output, contours, -1, (0, 255, 0), thickness=1)
    return output


def process_image(output_name: str = "lane_output.png") -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / "lane.png"
    output_path = script_dir / output_name

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    result = detect_lane_markers(image, fill_overlay=True)
    cv2.imwrite(str(output_path), result)
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    process_image()
