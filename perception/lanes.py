"""Classical CV lane detection: perspective warp -> threshold -> polynomial fit.

No ground truth exists for lanes in the KITTI 2D object detection subset
Phase 1 downloaded -- KITTI's lane-marking labels live in a separate "Road"
benchmark, out of scope for this project. So this module is validated
visually (rendered overlays on sample frames via the CLI below), not
numerically, unlike perception/distance.py.

Sanity-check gating: a sliding-window polynomial fit will happily fit a
curve to a handful of stray thresholded pixels (curb shadow, a parked car's
edge, sidewalk texture) and report it as a lane with no indication anything
is wrong -- confirmed empirically: fit_lane_polynomial found *some* fit for
both boundaries in 100% of a 400-frame sample before this gate existed, which
is not the same as finding a *correct* lane. detect_lanes() now rejects a fit
(falls back to None, which callers already treat as "lane not found") if the
two boundaries aren't roughly parallel or imply an implausible lane width --
the standard technique from the reference "Advanced Lane Finding" lineage
this pipeline descends from. This directly fixes LDWS triggering on nearly
every frame regardless of scene, which is what an ungated fit produces.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

# Assumed standard lane width, used only to convert the bird's-eye pixel
# scale to meters for the center-offset number. An approximation, like the
# rest of this classical pipeline -- there's no ground truth to calibrate
# against (see module docstring).
ASSUMED_LANE_WIDTH_M = 3.5

N_WINDOWS = 9
WINDOW_MARGIN_PX = 80
WINDOW_MIN_PIXELS = 40

# Minimum total pixels backing a fit before it's trusted at all. The old bar
# (>=3, enough to run np.polyfit on a 2nd-order curve at all) accepted fits
# built from a handful of noise pixels -- confirmed to be exactly why
# fit_lane_polynomial "succeeded" on 100% of a 400-frame sample regardless of
# whether a real lane marking was visible.
MIN_LANE_PIXELS = 50

# Sanity-check bounds on the fitted lane width, evaluated at two rows of the
# warped image (see _lanes_are_plausible). Width is checked at the bottom
# (nearest the vehicle) and mid-height; ratio bounds catch boundaries that
# aren't roughly parallel -- the standard "sanity check" step from the
# Advanced Lane Finding reference technique this pipeline is based on.
MIN_LANE_WIDTH_FRACTION = 0.15  # of image width
MAX_LANE_WIDTH_FRACTION = 0.9
MAX_WIDTH_RATIO = 2.0  # width at one row vs the other, either direction


def _roi_source_points(img_w: int, img_h: int) -> np.ndarray:
    """Trapezoid ROI tuned for KITTI's fixed ~1242x375 forward-facing camera framing."""
    return np.float32(
        [
            [img_w * 0.45, img_h * 0.58],  # top-left, near horizon
            [img_w * 0.58, img_h * 0.58],  # top-right, near horizon
            [img_w * 0.90, img_h * 0.97],  # bottom-right, near vehicle
            [img_w * 0.15, img_h * 0.97],  # bottom-left, near vehicle
        ]
    )


def _warp_destination_points(img_w: int, img_h: int) -> np.ndarray:
    return np.float32([[img_w * 0.25, 0], [img_w * 0.75, 0], [img_w * 0.75, img_h], [img_w * 0.25, img_h]])


def warp_to_birdseye(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perspective-warp the ROI trapezoid to a bird's-eye rectangle. Returns (warped, M, M_inv)."""
    h, w = img.shape[:2]
    src = _roi_source_points(w, h)
    dst = _warp_destination_points(w, h)
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, m, (w, h))
    return warped, m, m_inv


def threshold_lane_pixels(img: np.ndarray) -> np.ndarray:
    """Binary mask combining an HLS S-channel (color) and Sobel-x gradient (edge) threshold."""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = (s_channel > 100) & (s_channel <= 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel_x = np.abs(sobel_x)
    scaled = np.uint8(255 * abs_sobel_x / (np.max(abs_sobel_x) + 1e-6))
    gradient_binary = (scaled > 20) & (scaled <= 100)

    combined = np.zeros_like(s_channel, dtype=np.uint8)
    combined[s_binary | gradient_binary] = 1
    return combined


def fit_lane_polynomial(binary_warped: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sliding-window search + 2nd-order polynomial fit (x as a function of y) per lane boundary."""
    h, w = binary_warped.shape[:2]
    histogram = np.sum(binary_warped[h // 2 :, :], axis=0)
    midpoint = w // 2
    left_x_current = int(np.argmax(histogram[:midpoint]))
    right_x_current = int(np.argmax(histogram[midpoint:]) + midpoint)

    window_height = h // N_WINDOWS
    nonzero_y, nonzero_x = binary_warped.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    for window in range(N_WINDOWS):
        y_low = h - (window + 1) * window_height
        y_high = h - window * window_height

        left_low, left_high = left_x_current - WINDOW_MARGIN_PX, left_x_current + WINDOW_MARGIN_PX
        right_low, right_high = right_x_current - WINDOW_MARGIN_PX, right_x_current + WINDOW_MARGIN_PX

        in_row = (nonzero_y >= y_low) & (nonzero_y < y_high)
        good_left = (in_row & (nonzero_x >= left_low) & (nonzero_x < left_high)).nonzero()[0]
        good_right = (in_row & (nonzero_x >= right_low) & (nonzero_x < right_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > WINDOW_MIN_PIXELS:
            left_x_current = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > WINDOW_MIN_PIXELS:
            right_x_current = int(np.mean(nonzero_x[good_right]))

    left_idx = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_idx = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    left_fit = None
    if len(left_idx) >= MIN_LANE_PIXELS:
        left_fit = np.polyfit(nonzero_y[left_idx], nonzero_x[left_idx], 2)
    right_fit = None
    if len(right_idx) >= MIN_LANE_PIXELS:
        right_fit = np.polyfit(nonzero_y[right_idx], nonzero_x[right_idx], 2)
    return left_fit, right_fit


def _lanes_are_plausible(left_fit: np.ndarray, right_fit: np.ndarray, img_h: int, img_w: int) -> bool:
    """Sanity check: are these two boundaries roughly parallel and a plausible width apart?

    Evaluated at the bottom of the warped image (nearest the vehicle) and at
    mid-height, rather than comparing raw polynomial coefficients directly --
    a width comparison is a more direct, numerically stable proxy for "are
    these actually parallel" than differencing curve-fit coefficients.
    """
    y_bottom = img_h - 1
    y_mid = img_h * 0.5

    width_bottom = np.polyval(right_fit, y_bottom) - np.polyval(left_fit, y_bottom)
    width_mid = np.polyval(right_fit, y_mid) - np.polyval(left_fit, y_mid)

    if width_bottom <= 0 or width_mid <= 0:
        return False  # crossed or inverted boundaries

    if not (MIN_LANE_WIDTH_FRACTION * img_w <= width_bottom <= MAX_LANE_WIDTH_FRACTION * img_w):
        return False  # implausibly narrow or wide relative to the frame

    ratio = width_mid / width_bottom
    return bool((1 / MAX_WIDTH_RATIO) <= ratio <= MAX_WIDTH_RATIO)


def lane_center_offset(
    left_fit: np.ndarray | None,
    right_fit: np.ndarray | None,
    img_shape: tuple[int, ...],
    assumed_lane_width_m: float = ASSUMED_LANE_WIDTH_M,
) -> float | None:
    """Signed offset in meters, evaluated nearest the vehicle (bottom of the warped image).

    Positive = lane center falls left of image center, i.e. the vehicle is
    drifted right of lane center. Returns None if either lane boundary
    wasn't found, or the fitted boundaries don't form a sane (positive-width)
    lane.
    """
    if left_fit is None or right_fit is None:
        return None

    h, w = img_shape[:2]
    y_eval = h - 1
    x_left = np.polyval(left_fit, y_eval)
    x_right = np.polyval(right_fit, y_eval)
    lane_width_px = x_right - x_left
    if lane_width_px <= 0:
        return None

    meters_per_px = assumed_lane_width_m / lane_width_px
    lane_center_px = (x_left + x_right) / 2
    image_center_px = w / 2
    return (image_center_px - lane_center_px) * meters_per_px


@dataclass
class LaneResult:
    left_fit: np.ndarray | None
    right_fit: np.ndarray | None
    center_offset_m: float | None
    warp_matrix_inv: np.ndarray


def detect_lanes(img: np.ndarray) -> LaneResult:
    warped, _, m_inv = warp_to_birdseye(img)
    binary = threshold_lane_pixels(warped)
    left_fit, right_fit = fit_lane_polynomial(binary)

    if left_fit is not None and right_fit is not None:
        h, w = img.shape[:2]
        if not _lanes_are_plausible(left_fit, right_fit, h, w):
            left_fit, right_fit = None, None

    offset = lane_center_offset(left_fit, right_fit, img.shape)
    return LaneResult(left_fit=left_fit, right_fit=right_fit, center_offset_m=offset, warp_matrix_inv=m_inv)


def draw_lane_overlay(img: np.ndarray, result: LaneResult) -> np.ndarray:
    """Render the fitted lane region back onto the original (unwarped) image, for visual spot-checking."""
    if result.left_fit is None or result.right_fit is None:
        return img.copy()

    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, h)
    left_xs = np.polyval(result.left_fit, ys)
    right_xs = np.polyval(result.right_fit, ys)

    overlay = np.zeros_like(img)
    left_pts = np.array([np.transpose(np.vstack([left_xs, ys]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_xs, ys])))])
    pts = np.hstack((left_pts, right_pts))
    cv2.fillPoly(overlay, np.int32([pts]), (0, 255, 0))

    unwarped_overlay = cv2.warpPerspective(overlay, result.warp_matrix_inv, (w, h))
    combined = cv2.addWeighted(img, 1.0, unwarped_overlay, 0.4, 0)

    if result.center_offset_m is not None:
        cv2.putText(
            combined,
            f"offset: {result.center_offset_m:+.2f} m",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="path to an input frame")
    parser.add_argument("--out", required=True, help="path to write the rendered overlay")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"could not read image: {args.image}")

    result = detect_lanes(img)
    overlay = draw_lane_overlay(img, result)
    cv2.imwrite(args.out, overlay)

    if result.center_offset_m is not None:
        offset_str = f"{result.center_offset_m:+.2f}m"
    else:
        offset_str = "N/A (lane not found)"
    print(f"center offset: {offset_str}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
