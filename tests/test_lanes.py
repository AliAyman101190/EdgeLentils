import cv2
import numpy as np
import pytest

from perception.lanes import (
    MIN_LANE_PIXELS,
    _lanes_are_plausible,
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
