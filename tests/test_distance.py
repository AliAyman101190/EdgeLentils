import math
from pathlib import Path

import numpy as np
import pytest

from perception.calibration import CameraIntrinsics
from perception.distance import (
    _aggregate,
    _distance_bucket,
    _fit_camera_height,
    estimate_distance,
    estimate_distance_for_box,
    estimate_position_for_box,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_estimate_distance_known_value():
    # D = fy * H / (v - cy) = 700 * 1.65 / 70 = 16.5
    assert estimate_distance(v_bottom=250.0, cy=180.0, fy=700.0, camera_height=1.65) == pytest.approx(16.5)


def test_estimate_distance_at_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        estimate_distance(v_bottom=180.0, cy=180.0, fy=700.0)


def test_estimate_distance_above_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        estimate_distance(v_bottom=100.0, cy=180.0, fy=700.0)


def test_estimate_distance_for_box_reduces_to_plain_formula_at_zero_roll():
    intrinsics = CameraIntrinsics(fx=700.0, fy=700.0, cx=600.0, cy=180.0, baseline_offset=0.0)
    bbox = (400.0, 150.0, 500.0, 250.0)  # y2 = 250, u_center = 450 (off-center)

    result = estimate_distance_for_box(bbox, intrinsics, roll_rad=0.0)

    assert result == pytest.approx(estimate_distance(250.0, 180.0, 700.0))


def test_estimate_distance_for_box_dead_ahead_unaffected_by_roll():
    # u_center == cx -> the roll term's coefficient (u_center-cx) is zero,
    # so a dead-ahead box gives the same distance regardless of roll_rad.
    intrinsics = CameraIntrinsics(fx=700.0, fy=700.0, cx=600.0, cy=180.0, baseline_offset=0.0)
    bbox = (550.0, 150.0, 650.0, 250.0)  # u_center = 600 == cx

    no_roll = estimate_distance_for_box(bbox, intrinsics, roll_rad=0.0)
    with_roll = estimate_distance_for_box(bbox, intrinsics, roll_rad=math.radians(-1.4))

    assert with_roll == pytest.approx(no_roll)


def test_estimate_distance_for_box_roll_shifts_off_center_boxes_oppositely():
    intrinsics = CameraIntrinsics(fx=700.0, fy=700.0, cx=600.0, cy=180.0, baseline_offset=0.0)
    bbox_right = (650.0, 150.0, 750.0, 250.0)  # u_center = 700, right of cx
    bbox_left = (450.0, 150.0, 550.0, 250.0)  # u_center = 500, left of cx
    roll = math.radians(-1.4)

    baseline_right = estimate_distance_for_box(bbox_right, intrinsics, roll_rad=0.0)
    baseline_left = estimate_distance_for_box(bbox_left, intrinsics, roll_rad=0.0)
    right_with_roll = estimate_distance_for_box(bbox_right, intrinsics, roll_rad=roll)
    left_with_roll = estimate_distance_for_box(bbox_left, intrinsics, roll_rad=roll)

    # symmetric offsets from cx -> roll shifts distance in opposite directions
    # (not necessarily equal magnitude: Z=H/denom is nonlinear in the roll term)
    assert right_with_roll != pytest.approx(baseline_right)
    assert (right_with_roll - baseline_right) < 0
    assert (left_with_roll - baseline_left) > 0


@pytest.mark.parametrize(
    "z,expected_bucket",
    [
        (0.0, "<10m"),
        (9.9, "<10m"),
        (10.0, "10-25m"),
        (24.9, "10-25m"),
        (25.0, "25-50m"),
        (49.9, "25-50m"),
        (50.0, "50m+"),
        (103.6, "50m+"),
    ],
)
def test_distance_bucket_boundaries(z, expected_bucket):
    assert _distance_bucket(z) == expected_bucket


def test_estimate_position_for_box_lateral_and_distance():
    intrinsics = CameraIntrinsics(fx=700.0, fy=700.0, cx=600.0, cy=180.0, baseline_offset=0.0)
    bbox = (550.0, 150.0, 650.0, 250.0)  # u_center=600 (== cx), y2=250

    lateral_m, distance_m = estimate_position_for_box(bbox, intrinsics)

    assert distance_m == pytest.approx(estimate_distance(250.0, 180.0, 700.0))
    assert lateral_m == pytest.approx(0.0, abs=1e-9)  # u_center == cx -> dead ahead


def test_estimate_position_for_box_lateral_offset_sign():
    intrinsics = CameraIntrinsics(fx=700.0, fy=700.0, cx=600.0, cy=180.0, baseline_offset=0.0)
    bbox_right = (650.0, 150.0, 750.0, 250.0)  # u_center=700, right of cx
    bbox_left = (450.0, 150.0, 550.0, 250.0)  # u_center=500, left of cx

    # roll_rad=0 for a clean symmetry check -- roll breaks exact left/right
    # symmetry by design (see the dedicated roll tests), so isolate it here.
    lateral_right, _ = estimate_position_for_box(bbox_right, intrinsics, roll_rad=0.0)
    lateral_left, _ = estimate_position_for_box(bbox_left, intrinsics, roll_rad=0.0)

    assert lateral_right > 0
    assert lateral_left < 0
    assert lateral_right == pytest.approx(-lateral_left)  # symmetric offsets from cx


def test_fit_camera_height_recovers_known_h():
    rng = np.random.default_rng(0)
    true_h = 1.65
    denom = rng.uniform(0.01, 0.3, size=200)
    z_true = true_h / denom  # noise-free, generated from true_h

    fitted = _fit_camera_height(denom, z_true)

    assert fitted == pytest.approx(true_h, abs=1e-9)


def test_aggregate_mean_and_median():
    records = [{"abs_error": e} for e in (1.0, 2.0, 3.0, 4.0)]

    result = _aggregate(records)

    assert result["count"] == 4
    assert result["mean_abs_error_m"] == pytest.approx(2.5)
    assert result["median_abs_error_m"] == pytest.approx(2.5)


def test_aggregate_empty_records():
    result = _aggregate([])

    assert result["count"] == 0
    assert result["mean_abs_error_m"] is None
    assert result["median_abs_error_m"] is None


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed" / "splits" / "chen_val.txt").exists(),
    reason="processed KITTI dataset not present locally",
)
def test_validate_against_kitti_report_shape():
    from perception.distance import validate_against_kitti

    report = validate_against_kitti(split="val")

    assert report["overall"]["count"] > 0
    assert set(report["per_class"]) == {"Car", "Pedestrian", "Cyclist"}
    assert set(report["per_distance_bucket"]) == {"<10m", "10-25m", "25-50m", "50m+"}
    assert report["fitted_camera_height_m_diagnostic"] > 0
