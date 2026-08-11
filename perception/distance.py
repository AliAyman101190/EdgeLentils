"""Monocular ground-plane distance estimation, using KITTI's own calibration.

Flat-road assumption: D = fy * H / (v - cy), where v is the pixel row of the
object's ground-contact point (box bottom edge) and H is the camera height
above the road. See EdgeADAS_Project_Plan.md Phase 3 for the full write-up.

Known failure mode, confirmed empirically on the Chen val split: as an
object's ground-contact point approaches the horizon row (v -> cy), the
denominator approaches zero and predicted distance blows up -- a handful of
KITTI objects have their box bottom within ~1 pixel of cy and produce
predictions in the thousands of meters for objects that are actually 25-75m
away. This is a real, textbook limitation of the single-camera flat-ground
method, not a bug: reported error therefore always includes median/percentile
breakdowns alongside the mean, since a raw mean is tail-dominated by this
handful of near-horizon points and is not representative of typical accuracy.

Camera roll correction: the plain formula above assumes the ground is at a
constant height directly below the camera's vertical (image-row) axis,
regardless of lateral position -- true only if the camera's image-row axis is
exactly aligned with true vertical (gravity). Checked empirically on the
Chen val split (clean subset: non-truncated, z<50m, away from the horizon
singularity): predicted-vs-true distance error correlates with signed
lateral position (r~0.27) in an antisymmetric pattern -- objects far to one
side underestimate, far to the other side overestimate by a similar amount
-- which is exactly the signature of a small camera roll, not a formula bug.
A search over roll angles found theta ~ -1.4 deg minimizes val-set error,
cutting error in the far-lateral tails (|X|>10m) by ~30%; the effect is
negligible near X=0, where most objects sit, so overall median error moves
only slightly. Unlike CAMERA_HEIGHT_M, this angle is a calibration fit
against this project's own val set (defensible for a fixed physical rig
property, but not an independently-cited literature constant) -- flagged
here explicitly rather than presented as equally well-sourced.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perception.calibration import CameraIntrinsics, load_camera_intrinsics
from perception.kitti_labels import CLASS_MAP, YOLO_CLASSES, parse_label_file

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_TRAINING = REPO_ROOT / "data" / "raw" / "training"
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "splits"

# Standard literature constant for the KITTI recording rig, not a per-frame
# measured value: KITTI's calib files encode camera-to-camera and
# camera-to-LiDAR transforms, not camera-to-road-surface height. Traces back
# to Chen et al.'s Mono3D (the same "Chen split" used for Phase 1's train/val
# split) and used consistently since across monocular 3D KITTI literature.
# A documented simplification -- real error from it (pitch, suspension,
# non-flat road) is exactly what validate_against_kitti() below measures.
CAMERA_HEIGHT_M = 1.65

# Calibration fit, not a literature constant -- see module docstring.
CAMERA_ROLL_RAD = math.radians(-1.4)

DISTANCE_BUCKETS = (
    ("<10m", 0.0, 10.0),
    ("10-25m", 10.0, 25.0),  # straddles the 15m FCWS threshold -- the range that matters most for it
    ("25-50m", 25.0, 50.0),
    ("50m+", 50.0, float("inf")),
)

# Below this many pixels from the horizon, a 1-pixel annotation/detection
# wobble swings predicted distance by many meters -- predictions are kept in
# the report (not silently dropped), but flagged as low-confidence so the
# near-horizon tail is visible without having to go hunting for it.
NEAR_HORIZON_PX = 2.0


def estimate_distance(v_bottom: float, cy: float, fy: float, camera_height: float = CAMERA_HEIGHT_M) -> float:
    """Flat-ground monocular distance, ignoring camera roll: D = fy * H / (v_bottom - cy).

    Kept as the simple building block (matches the textbook derivation
    exactly). estimate_distance_for_box/estimate_position_for_box apply the
    roll-corrected version below, since that needs horizontal position too.
    """
    denom = v_bottom - cy
    if denom <= 0:
        raise ValueError(
            f"v_bottom ({v_bottom}) is at or above the horizon (cy={cy}) -- not a valid ground-plane point"
        )
    return fy * camera_height / denom


def estimate_distance_for_box(
    bbox: tuple[float, float, float, float],
    intrinsics: CameraIntrinsics,
    camera_height: float = CAMERA_HEIGHT_M,
    roll_rad: float = CAMERA_ROLL_RAD,
) -> float:
    """Roll-corrected distance to a box's ground-contact point (bottom edge, horizontal center).

    D = H / [(y2-cy)/fy - (u_center-cx)/fx * roll_rad]. Reduces exactly to
    estimate_distance's plain formula when roll_rad=0. See CAMERA_ROLL_RAD.
    """
    x1, _, x2, y2 = bbox
    u_center = (x1 + x2) / 2
    v_term = (y2 - intrinsics.cy) / intrinsics.fy
    u_term = (u_center - intrinsics.cx) / intrinsics.fx
    denom = v_term - u_term * roll_rad
    if denom <= 0:
        raise ValueError(
            f"box {bbox} ground-contact point is at or above the (roll-corrected) horizon -- "
            "not a valid ground-plane point"
        )
    return camera_height / denom


def estimate_position_for_box(
    bbox: tuple[float, float, float, float],
    intrinsics: CameraIntrinsics,
    camera_height: float = CAMERA_HEIGHT_M,
    roll_rad: float = CAMERA_ROLL_RAD,
) -> tuple[float, float]:
    """Ground-plane (lateral_m, distance_m) for a box's ground-contact point.

    lateral_m uses the same (roll-corrected) distance_m: X = (u_center - cx) *
    Z / fx. Positive lateral_m is to the right of the camera center.
    """
    x1, _, x2, _ = bbox
    z = estimate_distance_for_box(bbox, intrinsics, camera_height, roll_rad)
    u_center = (x1 + x2) / 2
    x = (u_center - intrinsics.cx) * z / intrinsics.fx
    return x, z


@dataclass
class Detection:
    obj_class: str
    bbox: tuple[float, float, float, float]
    distance_m: float
    lateral_m: float


def _distance_bucket(z: float) -> str:
    for name, lo, hi in DISTANCE_BUCKETS:
        if lo <= z < hi:
            return name
    raise AssertionError(f"z={z} did not match any bucket")  # buckets cover [0, inf)


def _fit_camera_height(denom: np.ndarray, z_true: np.ndarray) -> float:
    """Robust (median) H implied by the observed ground truth, as a diagnostic only.

    denom is the same (roll-corrected) denominator estimate_distance_for_box
    uses, so this checks CAMERA_HEIGHT_M against the model actually shipped,
    not a simplified one. Z = H/denom rearranges to H = Z*denom -- solved
    per-sample and take the median, not a least-squares fit: objects near the
    horizon (denom -> 0) are genuinely singular under this formula, and an
    ordinary least-squares fit weights those points by 1/denom^2, letting a
    handful of near-horizon samples dominate and produce a meaningless
    result. The median is resistant to that tail by construction.
    """
    per_sample_h = z_true * denom
    return float(np.median(per_sample_h))


def _aggregate(records: list[dict]) -> dict:
    errors = [r["abs_error"] for r in records]
    result = {
        "count": len(errors),
        "mean_abs_error_m": float(np.mean(errors)) if errors else None,
        "median_abs_error_m": float(np.median(errors)) if errors else None,
    }
    if errors:
        p90, p95, p99 = np.percentile(errors, [90, 95, 99])
        result["p90_abs_error_m"] = float(p90)
        result["p95_abs_error_m"] = float(p95)
        result["p99_abs_error_m"] = float(p99)
    else:
        result["p90_abs_error_m"] = result["p95_abs_error_m"] = result["p99_abs_error_m"] = None
    return result


def validate_against_kitti(split: str = "val", roll_rad: float = CAMERA_ROLL_RAD) -> dict:
    """Score the flat-ground formula against KITTI's own ground-truth 3D locations.

    Uses ground-truth 2D boxes (not detector predictions), so the reported
    error isolates the flat-ground geometry/camera-height/roll assumptions
    from detector localization error.
    """
    split_file = SPLITS_DIR / f"chen_{split}.txt"
    frame_ids = [Path(line).stem for line in split_file.read_text().splitlines() if line.strip()]

    records = []
    n_behind_camera = 0
    n_above_horizon = 0
    n_near_horizon_low_confidence = 0

    for frame_id in frame_ids:
        intrinsics = load_camera_intrinsics(RAW_TRAINING / "calib" / f"{frame_id}.txt")

        for obj in parse_label_file(RAW_TRAINING / "label_2" / f"{frame_id}.txt"):
            mapped_class = CLASS_MAP.get(obj.raw_class)
            if mapped_class is None:
                continue

            z_true = obj.location[2]
            if z_true <= 0:
                n_behind_camera += 1
                continue

            x1, _, x2, y2 = obj.bbox
            u_center = (x1 + x2) / 2
            v_minus_cy = y2 - intrinsics.cy
            v_term = v_minus_cy / intrinsics.fy
            u_term = (u_center - intrinsics.cx) / intrinsics.fx
            denom = v_term - u_term * roll_rad
            if denom <= 0:
                n_above_horizon += 1
                continue

            if 0 < v_minus_cy < NEAR_HORIZON_PX:
                n_near_horizon_low_confidence += 1

            z_pred = CAMERA_HEIGHT_M / denom
            records.append(
                {
                    "class": mapped_class,
                    "z_true": z_true,
                    "abs_error": abs(z_pred - z_true),
                    "bucket": _distance_bucket(z_true),
                    "denom": denom,
                }
            )

    fitted_h = None
    if records:
        fitted_h = _fit_camera_height(
            np.array([r["denom"] for r in records]),
            np.array([r["z_true"] for r in records]),
        )

    return {
        "split": split,
        "camera_height_m": CAMERA_HEIGHT_M,
        "camera_roll_rad": roll_rad,
        "fitted_camera_height_m_diagnostic": fitted_h,
        "n_filtered_behind_camera": n_behind_camera,
        "n_filtered_above_horizon": n_above_horizon,
        "n_near_horizon_low_confidence": n_near_horizon_low_confidence,
        "overall": _aggregate(records),
        "per_class": {cls: _aggregate([r for r in records if r["class"] == cls]) for cls in YOLO_CLASSES},
        "per_distance_bucket": {
            name: _aggregate([r for r in records if r["bucket"] == name]) for name, _, _ in DISTANCE_BUCKETS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="distance_validation.json")
    args = parser.parse_args()

    report = validate_against_kitti(args.split)
    Path(args.out).write_text(json.dumps(report, indent=2))

    def _line(label, stats):
        if not stats["count"]:
            return
        print(
            f"  {label}: n={stats['count']} mean={stats['mean_abs_error_m']:.2f}m "
            f"median={stats['median_abs_error_m']:.2f}m p95={stats['p95_abs_error_m']:.2f}m"
        )

    _line("overall", report["overall"])
    for cls, stats in report["per_class"].items():
        _line(cls, stats)
    for bucket, stats in report["per_distance_bucket"].items():
        _line(bucket, stats)
    print(
        f"robust (median) implied H: {report['fitted_camera_height_m_diagnostic']:.3f}m "
        f"vs shipped {CAMERA_HEIGHT_M}m (roll={math.degrees(report['camera_roll_rad']):.2f} deg)"
    )
    print(
        f"filtered: {report['n_filtered_behind_camera']} behind-camera, "
        f"{report['n_filtered_above_horizon']} above-horizon "
        f"({report['n_near_horizon_low_confidence']} kept but within {NEAR_HORIZON_PX}px of horizon -- "
        "low-confidence, see module docstring)"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
