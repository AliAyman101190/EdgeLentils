import cv2
import numpy as np
import pytest

from perception.lanes import (
    CENTER_PRIOR_LEFT_FRACTION,
    CENTER_PRIOR_RIGHT_FRACTION,
    MIN_LANE_PIXELS,
    _center_proximity_score,
    _coherence_score,
    _curve_coherence,
    _find_candidate_columns,
    _lanes_are_plausible,
    _pair_parallelism_score,
    _pair_width_score,
    detect_lanes,
    draw_lane_overlay,
    fit_lane_polynomial,
    lane_center_offset,
    threshold_lane_pixels,
    warp_to_birdseye,
)


def _blank_image(h=300, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _image_with_two_vertical_lines(h=300, w=200, left_x=50, right_x=150, thickness=6):
    img = _blank_image(h, w)
    cv2.line(img, (left_x, 0), (left_x, h - 1), (255, 255, 255), thickness)
    cv2.line(img, (right_x, 0), (right_x, h - 1), (255, 255, 255), thickness)
    return img


def test_warp_to_birdseye_preserves_shape_and_inverts():
    img = _blank_image()
    warped, m, m_inv = warp_to_birdseye(img)

    assert warped.shape == img.shape
    assert m.shape == (3, 3)
    assert m_inv.shape == (3, 3)

    point = np.array([[[100.0, 150.0]]], dtype=np.float32)
    forward = cv2.perspectiveTransform(point, m)
    back = cv2.perspectiveTransform(forward, m_inv)
    assert back[0, 0] == pytest.approx(point[0, 0], abs=1e-3)


def test_threshold_lane_pixels_is_binary_and_finds_colored_lines():
    # Pure white-on-black has zero HLS saturation (achromatic) and a step
    # edge sharp enough to saturate the Sobel scaling -- neither triggers a
    # threshold tuned for real photos. Yellow (real lane-marking color) has
    # genuine saturation and exercises the color half of the OR meaningfully.
    img = _blank_image()
    cv2.line(img, (50, 0), (50, img.shape[0] - 1), (0, 255, 255), 6)  # BGR yellow
    cv2.line(img, (150, 0), (150, img.shape[0] - 1), (0, 255, 255), 6)

    mask = threshold_lane_pixels(img)

    assert mask.shape == img.shape[:2]
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.sum() > 0


def test_fit_lane_polynomial_recovers_known_line_positions():
    binary = np.zeros((300, 200), dtype=np.uint8)
    binary[:, 48:53] = 1  # left line around x=50
    binary[:, 148:153] = 1  # right line around x=150

    left_fit, right_fit = fit_lane_polynomial(binary)

    assert left_fit is not None
    assert right_fit is not None
    assert np.polyval(left_fit, 299) == pytest.approx(50, abs=5)
    assert np.polyval(right_fit, 299) == pytest.approx(150, abs=5)


def test_fit_lane_polynomial_returns_none_for_blank_input():
    binary = np.zeros((300, 200), dtype=np.uint8)

    left_fit, right_fit = fit_lane_polynomial(binary)

    assert left_fit is None
    assert right_fit is None


def test_fit_lane_polynomial_rejects_sparse_noise_pixels():
    # A handful of scattered pixels is enough to run polyfit on (the old bar
    # was >=3), but not enough to trust as a real lane marking -- this is
    # exactly the failure mode that made LDWS trigger on nearly every frame
    # before MIN_LANE_PIXELS existed.
    assert MIN_LANE_PIXELS > 10, "test assumes the sparse case below stays under the real threshold"
    binary = np.zeros((300, 200), dtype=np.uint8)
    binary[100:105, 50] = 1  # 5 stray pixels, well under MIN_LANE_PIXELS
    binary[100:105, 150] = 1

    left_fit, right_fit = fit_lane_polynomial(binary)

    assert left_fit is None
    assert right_fit is None


def test_find_candidate_columns_returns_separated_peaks():
    histogram = np.zeros(200)
    histogram[178:192] = 100.0  # one wide dominant feature (fence-like plateau)
    histogram[50] = 30.0  # smaller, distinct feature (dash-like)

    columns = _find_candidate_columns(histogram, k=3, min_separation_px=20)

    # Both features must be represented; the plateau must not consume
    # multiple candidate slots with adjacent columns of itself.
    assert any(abs(c - 50) <= 2 for c in columns)
    plateau_hits = [c for c in columns if 178 - 20 <= c <= 192 + 20]
    assert len(plateau_hits) == 1


def test_find_candidate_columns_empty_histogram_returns_nothing():
    assert _find_candidate_columns(np.zeros(200), k=3, min_separation_px=10) == []


def test_coherence_score_favors_curve_consistent_candidates():
    # Measured bands on real frames: genuine boundaries 0.63-0.99, scattered
    # clutter 0.19-0.53 -- the score must order those correctly and saturate
    # for clearly-coherent lines.
    solid_line = _coherence_score(0.95)
    dashed_line = _coherence_score(0.65)
    clutter = _coherence_score(0.3)

    assert solid_line == pytest.approx(1.0)  # saturates at/above COHERENCE_FULL_SCORE_AT
    assert solid_line >= dashed_line > clutter
    assert clutter > 0.0  # graded, not a disqualifier


def test_curve_coherence_measures_tight_band_only():
    h = 100
    ys = np.arange(h)
    # A perfect vertical line at x=50: every row within band -> coherence 1.0
    xs_on_line = np.full(h, 50.0)
    fit = np.array([0.0, 0.0, 50.0])
    assert _curve_coherence(ys, xs_on_line, fit, h) == pytest.approx(1.0)

    # Same rows but pixels far from the fitted curve -> coherence 0.0
    xs_far = np.full(h, 120.0)
    assert _curve_coherence(ys, xs_far, fit, h) == pytest.approx(0.0)


def test_center_proximity_is_soft_and_monotonic():
    img_w = 200
    peak = int(CENTER_PRIOR_LEFT_FRACTION * img_w)
    at_prior = _center_proximity_score(peak, img_w, "left")
    nearby = _center_proximity_score(peak + 25, img_w, "left")
    far = _center_proximity_score(4, img_w, "left")  # extreme edge (guardrail territory)

    assert at_prior == pytest.approx(1.0)
    assert at_prior > nearby > far
    assert far > 0.0  # soft: far candidates penalized, never excluded


def test_center_proximity_right_side_uses_right_prior():
    img_w = 200
    right_peak = int(CENTER_PRIOR_RIGHT_FRACTION * img_w)
    assert _center_proximity_score(right_peak, img_w, "right") == pytest.approx(1.0)
    assert _center_proximity_score(right_peak, img_w, "left") < _center_proximity_score(
        right_peak, img_w, "right"
    )


def test_pair_width_score_prefers_plausible_lane_widths():
    img_h, img_w = 300, 200

    def pair_at(width_px):
        left = np.array([0.0, 0.0, 60.0])
        right = np.array([0.0, 0.0, 60.0 + width_px])
        return _pair_width_score(left, right, img_h, img_w)

    plausible = pair_at(0.30 * img_w)
    too_wide = pair_at(0.85 * img_w)  # e.g. one side latched onto a guardrail
    too_narrow = pair_at(0.05 * img_w)

    assert plausible == pytest.approx(1.0)
    assert too_wide == 0.0
    assert too_narrow == 0.0
    assert 0.0 < pair_at(0.55 * img_w) < 1.0  # graded between plateau and zero


def test_pair_parallelism_prefers_constant_width():
    img_h = 300
    left = np.array([0.0, 0.0, 50.0])
    parallel_right = np.array([0.0, 0.0, 150.0])
    diverging_right = np.array([0.0, 0.4, 90.0])  # width grows ~120px over the image height
    crossed_right = np.array([0.0, 0.0, 30.0])

    parallel = _pair_parallelism_score(left, parallel_right, img_h)
    diverging = _pair_parallelism_score(left, diverging_right, img_h)
    crossed = _pair_parallelism_score(left, crossed_right, img_h)

    assert parallel == pytest.approx(1.0)
    assert parallel > diverging
    assert crossed == 0.0


def test_fit_lane_polynomial_prefers_dashed_line_over_continuous_edge():
    # The headline regression for the scoring redesign: a continuous
    # fence-like edge with MORE total pixels than the dashed lane paint,
    # in the same half. Raw pixel count (the old argmax search) picks the
    # fence; the scored search must pick the dashes.
    h, w = 300, 200
    binary = np.zeros((h, w), dtype=np.uint8)
    binary[:, 18:22] = 1  # continuous fence at x~20: 300 rows x 4 cols = 1200 px
    for y0 in range(0, h, 60):
        binary[y0 : y0 + 30, 58:62] = 1  # dashed line at x~60: 150 rows x 4 cols = 600 px
    binary[:, 148:152] = 1  # solid right boundary at x~150 (width 90px = 0.45w from dashes)

    left_fit, right_fit = fit_lane_polynomial(binary)

    assert left_fit is not None
    assert right_fit is not None
    assert np.polyval(left_fit, h - 1) == pytest.approx(60, abs=6)  # dashes, not the fence at 20
    assert np.polyval(right_fit, h - 1) == pytest.approx(150, abs=6)


def test_lanes_are_plausible_accepts_parallel_lines():
    left_fit = np.array([0.0, 0.0, 50.0])
    right_fit = np.array([0.0, 0.0, 150.0])  # constant 100px width, well within bounds

    assert _lanes_are_plausible(left_fit, right_fit, img_h=300, img_w=200) is True


def test_lanes_are_plausible_rejects_crossed_lines():
    left_fit = np.array([0.0, 0.0, 150.0])
    right_fit = np.array([0.0, 0.0, 50.0])  # "right" boundary left of "left"

    assert _lanes_are_plausible(left_fit, right_fit, img_h=300, img_w=200) is False


def test_lanes_are_plausible_rejects_implausibly_narrow_width():
    left_fit = np.array([0.0, 0.0, 95.0])
    right_fit = np.array([0.0, 0.0, 105.0])  # 10px width = 5% of a 200px-wide frame

    assert _lanes_are_plausible(left_fit, right_fit, img_h=300, img_w=200) is False


def test_lanes_are_plausible_rejects_implausibly_wide_width():
    left_fit = np.array([0.0, 0.0, 5.0])
    right_fit = np.array([0.0, 0.0, 195.0])  # 190px width = 95% of a 200px-wide frame

    assert _lanes_are_plausible(left_fit, right_fit, img_h=300, img_w=200) is False


def test_lanes_are_plausible_rejects_non_parallel_lines():
    left_fit = np.array([0.0, 0.0, 90.0])  # flat at x=90
    right_fit = np.array([0.0, 0.5, 60.0])  # drifts steadily wider with y -- not parallel

    assert _lanes_are_plausible(left_fit, right_fit, img_h=300, img_w=200) is False


def test_lane_center_offset_known_value():
    left_fit = np.array([0.0, 0.0, 40.0])
    right_fit = np.array([0.0, 0.0, 140.0])

    offset = lane_center_offset(left_fit, right_fit, img_shape=(300, 200), assumed_lane_width_m=3.5)

    # lane_center_px=90, image_center_px=100 -> (100-90)*3.5/100 = 0.35
    assert offset == pytest.approx(0.35, abs=1e-6)


def test_lane_center_offset_none_when_fit_missing():
    assert lane_center_offset(None, np.array([0.0, 0.0, 140.0]), (300, 200)) is None
    assert lane_center_offset(np.array([0.0, 0.0, 40.0]), None, (300, 200)) is None


def test_lane_center_offset_none_for_nonpositive_width():
    left_fit = np.array([0.0, 0.0, 140.0])
    right_fit = np.array([0.0, 0.0, 40.0])  # crossed: right is left of left

    assert lane_center_offset(left_fit, right_fit, (300, 200)) is None


def test_detect_lanes_and_draw_overlay_smoke():
    img = _image_with_two_vertical_lines(h=300, w=200, left_x=60, right_x=140, thickness=10)

    result = detect_lanes(img)
    overlay = draw_lane_overlay(img, result)

    assert overlay.shape == img.shape
    assert overlay.dtype == img.dtype


def test_draw_lane_overlay_returns_copy_when_lane_not_found():
    img = _blank_image()
    result = detect_lanes(img)  # no lane markings present -> fits should be None

    overlay = draw_lane_overlay(img, result)

    assert overlay.shape == img.shape
    np.testing.assert_array_equal(overlay, img)
