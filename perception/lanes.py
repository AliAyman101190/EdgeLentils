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
# Half-width of each sliding window, as a fraction of image width -- 0.064
# reproduces the previous absolute 80px on KITTI's 1242px frames exactly,
# but scales with resolution so nearby distinct features (e.g. a guardrail
# right next to dashed paint) don't get merged into one corridor on
# differently-sized inputs.
WINDOW_MARGIN_FRACTION = 0.064
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

# --- Candidate search scoring ------------------------------------------------
# The search no longer picks each side's start column by raw pixel count
# (argmax of the histogram): raw count structurally favors continuous features
# -- fences, guardrails, curbs -- over dashed lane paint, because dashes can
# never accumulate the "on"-pixel total of a solid edge even when they sit
# exactly where the lane is (confirmed on real frames: a guardrail's histogram
# peak drowns out the adjacent dashed line's). Instead, several candidate
# columns are generated per side and scored on lane-like properties.

# Candidate generation: up to this many local histogram peaks per side, forced
# apart by a minimum separation so they represent distinct features rather
# than adjacent columns of one dominant edge. k=4 (tuned up from 3): with only
# 3 slots, a cluttered side (guardrail + road edge + vegetation) could exhaust
# candidacy before the real dashed line's weaker peak got a slot.
CANDIDATES_PER_SIDE = 4
CANDIDATE_MIN_SEPARATION_FRACTION = 0.05  # of image width

# Curve-coherence: fraction of warped rows with >=1 pixel within a tight band
# around the *fitted curve* (not the whole walk corridor). Real lane
# boundaries -- dashed or solid -- put most rows' pixels right on the curve;
# scattered clutter fitted by force leaves the curve passing through mostly
# empty rows. Measured on real frames: genuine boundaries 0.63-0.99, clutter
# 0.19-0.53.
#
# Honest design note: this REPLACED the originally-specified dash-likelihood
# signal (penalize continuous coverage, favor intermittent). Dash
# intermittency proved unmeasurable in this pipeline when tested on real
# frames: measured over the walk corridor, road-texture threshold noise fills
# the dash gaps (real dashed lines read as ~98% covered while off-road
# clutter corridors read as ~83% -- the signal actually *inverted*); measured
# in a tight band, warped dashes still read dense (0.86+). Coherence is the
# version of "does this look like one real line" that the data supports.
COHERENCE_BAND_PX = 8
COHERENCE_FULL_SCORE_AT = 0.8  # coherence >= this scores 1.0; linear below

# Center-proximity: soft Gaussian prior on the candidate's start column.
# Initially centered on the warp-destination anchors (0.25w/0.75w); tuned
# inboard to 0.32w/0.68w after the tuning set showed the anchors sit closer
# to the *road edges* than to the ego lane's own boundaries -- observed real
# ego-lane boundaries on demonstrably-correct frames: ~0.35w (left) and
# 0.56w-0.79w (right). At 0.25w the prior actively preferred a road-edge
# line over the true ego-lane line sitting at 0.36w. Sigma stays wide by
# design: a genuinely departing lane must remain findable far off-center --
# this signal nudges ranking, it cannot exclude.
CENTER_PRIOR_LEFT_FRACTION = 0.32
CENTER_PRIOR_RIGHT_FRACTION = 0.68
CENTER_PRIOR_SIGMA_FRACTION = 0.15

# Pixel-support saturation: a small term so a barely-above-minimum wisp does
# not tie a well-supported line; saturates quickly so bulk count stops
# mattering beyond "clearly enough" (the whole point is NOT to reward bulk).
SUPPORT_SATURATION_PIXELS = 500

# Pair width: graded score peaking for a bottom-row lane width inside the band
# observed on frames where the pipeline demonstrably tracked real paint
# (~0.20w-0.45w), falling off toward implausibly narrow/wide pairs (e.g. one
# side latched onto a guardrail far off to the side). Upper zero tuned down
# from 0.75 to 0.65 after the tuning set showed road-edge-to-road-edge pairs
# (~0.55w) were surviving with too much width score.
PAIR_WIDTH_PLATEAU_FRACTION = (0.20, 0.45)
PAIR_WIDTH_ZERO_FRACTION = (0.08, 0.65)

# Parallelism: scored from the coefficient of variation of the pair's width
# sampled along the curves -- parallel boundaries keep near-constant width.
# cv >= this scores 0; cv == 0 scores 1.
PARALLELISM_CV_CEILING = 0.5

# Signal weights. Initial values chosen as even-handed starting points, then
# tuned against the visual tuning set (known-failure + known-good frames)
# rather than guessed -- see EdgeADAS_Project_Plan.md Phase 3 notes for the
# tuning outcome.
LINE_WEIGHT_COHERENCE = 0.4
LINE_WEIGHT_CENTER = 0.4
LINE_WEIGHT_SUPPORT = 0.2
PAIR_WEIGHT_LINES = 0.4
PAIR_WEIGHT_WIDTH = 0.3
PAIR_WEIGHT_PARALLEL = 0.3


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


def _find_candidate_columns(
    histogram: np.ndarray, k: int = CANDIDATES_PER_SIDE, min_separation_px: int = 1
) -> list[int]:
    """Up to k local histogram peaks, each at least min_separation_px apart.

    Iteratively takes the highest remaining column and suppresses its
    neighborhood, so the returned columns represent distinct image features
    rather than adjacent columns of one dominant edge.
    """
    remaining = histogram.astype(np.float64).copy()
    columns: list[int] = []
    for _ in range(k):
        peak = int(np.argmax(remaining))
        if remaining[peak] <= 0:
            break
        columns.append(peak)
        lo = max(0, peak - min_separation_px)
        remaining[lo : peak + min_separation_px + 1] = 0
    return columns


@dataclass
class _LineCandidate:
    start_col: int
    fit: np.ndarray
    coherence: float
    n_pixels: int
    line_score: float


def _collect_lane_pixels(
    nonzero_y: np.ndarray, nonzero_x: np.ndarray, img_h: int, start_col: int, margin_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Single-boundary sliding-window walk: the existing 9-band search, for one start column.

    Returns the (ys, xs) of collected pixels. Same mechanics as before the
    scoring redesign -- 9 bands bottom-to-top, each window recentering on the
    mean x of what it caught when it caught enough.
    """
    window_height = img_h // N_WINDOWS
    x_current = start_col
    collected = []

    for window in range(N_WINDOWS):
        y_low = img_h - (window + 1) * window_height
        y_high = img_h - window * window_height
        in_window = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= x_current - margin_px)
            & (nonzero_x < x_current + margin_px)
        )
        good = in_window.nonzero()[0]
        collected.append(good)
        if len(good) > WINDOW_MIN_PIXELS:
            x_current = int(np.mean(nonzero_x[good]))

    idx = np.concatenate(collected) if collected else np.array([], dtype=int)
    return nonzero_y[idx], nonzero_x[idx]


def _curve_coherence(nonzero_y: np.ndarray, nonzero_x: np.ndarray, fit: np.ndarray, img_h: int) -> float:
    """Fraction of warped rows with >=1 pixel within COHERENCE_BAND_PX of the fitted curve."""
    residual = np.abs(nonzero_x - np.polyval(fit, nonzero_y))
    near = residual <= COHERENCE_BAND_PX
    return len(np.unique(nonzero_y[near])) / img_h


def _coherence_score(coherence: float) -> float:
    """Signal 1: favor candidates whose pixels lie consistently along one curve.

    Monotonic: 1.0 at/above COHERENCE_FULL_SCORE_AT, linear below. See the
    constants block for why this replaced the dash-intermittency design.
    """
    return min(1.0, coherence / COHERENCE_FULL_SCORE_AT)


def _center_proximity_score(start_col: int, img_w: int, side: str) -> float:
    """Signal 2: soft Gaussian prior on where this side's boundary typically sits.

    Never zero in practice -- nudges the ranking toward typical positions
    without excluding a genuinely departing lane found far off-center.
    """
    center_fraction = CENTER_PRIOR_LEFT_FRACTION if side == "left" else CENTER_PRIOR_RIGHT_FRACTION
    sigma = CENTER_PRIOR_SIGMA_FRACTION * img_w
    return float(np.exp(-0.5 * ((start_col - center_fraction * img_w) / sigma) ** 2))


def _pair_width_score(left_fit: np.ndarray, right_fit: np.ndarray, img_h: int, img_w: int) -> float:
    """Signal 3: graded preference for pairs whose bottom-row width looks like a real lane.

    1.0 inside the plateau observed on demonstrably-correct frames, linear
    falloff to 0.0 at the implausible extremes (crossed/too-narrow pairs, or
    one side latched onto a feature far off to the side).
    """
    y_bottom = img_h - 1
    width = (np.polyval(right_fit, y_bottom) - np.polyval(left_fit, y_bottom)) / img_w
    plateau_lo, plateau_hi = PAIR_WIDTH_PLATEAU_FRACTION
    zero_lo, zero_hi = PAIR_WIDTH_ZERO_FRACTION
    if plateau_lo <= width <= plateau_hi:
        return 1.0
    if width <= zero_lo or width >= zero_hi:
        return 0.0
    if width < plateau_lo:
        return (width - zero_lo) / (plateau_lo - zero_lo)
    return (zero_hi - width) / (zero_hi - plateau_hi)


def _pair_parallelism_score(left_fit: np.ndarray, right_fit: np.ndarray, img_h: int) -> float:
    """Signal 4: parallel boundaries keep near-constant width along their length.

    Scored from the coefficient of variation of the width sampled at several
    rows; any non-positive width (crossed boundaries) scores 0.
    """
    ys = np.array([img_h * 0.25, img_h * 0.5, img_h * 0.75, img_h - 1.0])
    widths = np.polyval(right_fit, ys) - np.polyval(left_fit, ys)
    if np.any(widths <= 0):
        return 0.0
    cv = float(np.std(widths) / np.mean(widths))
    return max(0.0, 1.0 - cv / PARALLELISM_CV_CEILING)


def _score_side_candidates(
    nonzero_y: np.ndarray,
    nonzero_x: np.ndarray,
    histogram_half: np.ndarray,
    column_offset: int,
    img_h: int,
    img_w: int,
    side: str,
) -> list[_LineCandidate]:
    """Stage 1: generate, collect, fit, and per-line-score one side's candidates."""
    min_sep = max(1, int(CANDIDATE_MIN_SEPARATION_FRACTION * img_w))
    margin_px = max(1, int(WINDOW_MARGIN_FRACTION * img_w))
    candidates = []
    for col in _find_candidate_columns(histogram_half, min_separation_px=min_sep):
        start_col = col + column_offset
        ys, xs = _collect_lane_pixels(nonzero_y, nonzero_x, img_h, start_col, margin_px)
        if len(ys) < MIN_LANE_PIXELS:
            continue
        fit = np.polyfit(ys, xs, 2)
        coherence = _curve_coherence(nonzero_y, nonzero_x, fit, img_h)
        line_score = (
            LINE_WEIGHT_COHERENCE * _coherence_score(coherence)
            + LINE_WEIGHT_CENTER * _center_proximity_score(start_col, img_w, side)
            + LINE_WEIGHT_SUPPORT * min(1.0, len(ys) / SUPPORT_SATURATION_PIXELS)
        )
        candidates.append(
            _LineCandidate(
                start_col=start_col,
                fit=fit,
                coherence=coherence,
                n_pixels=len(ys),
                line_score=line_score,
            )
        )
    return candidates


def fit_lane_polynomial(binary_warped: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Two-stage candidate search + 2nd-order polynomial fit per lane boundary.

    Stage 1 (per side): several local histogram peaks become candidate start
    columns; each is walked with the sliding-window search, fitted, and scored
    on dash-likelihood + center-proximity + pixel support. Stage 2 (pairs):
    every (left, right) combination is scored on the per-line scores plus
    width-plausibility and parallelism, and the best pair wins. See the
    scoring-constants block at the top of this module for the rationale.
    """
    h, w = binary_warped.shape[:2]
    histogram = np.sum(binary_warped[h // 2 :, :], axis=0)
    midpoint = w // 2
    nonzero_y, nonzero_x = binary_warped.nonzero()

    left_candidates = _score_side_candidates(nonzero_y, nonzero_x, histogram[:midpoint], 0, h, w, "left")
    right_candidates = _score_side_candidates(
        nonzero_y, nonzero_x, histogram[midpoint:], midpoint, h, w, "right"
    )

    if not left_candidates or not right_candidates:
        # One-sided (or empty) result: return the best available fit per side,
        # same contract as before -- downstream treats either-None as no lane.
        best_left = max(left_candidates, key=lambda c: c.line_score, default=None)
        best_right = max(right_candidates, key=lambda c: c.line_score, default=None)
        return (
            best_left.fit if best_left is not None else None,
            best_right.fit if best_right is not None else None,
        )

    best_pair = max(
        ((lc, rc) for lc in left_candidates for rc in right_candidates),
        key=lambda pair: (
            PAIR_WEIGHT_LINES * (pair[0].line_score + pair[1].line_score) / 2
            + PAIR_WEIGHT_WIDTH * _pair_width_score(pair[0].fit, pair[1].fit, h, w)
            + PAIR_WEIGHT_PARALLEL * _pair_parallelism_score(pair[0].fit, pair[1].fit, h)
        ),
    )
    return best_pair[0].fit, best_pair[1].fit


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
