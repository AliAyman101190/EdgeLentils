"""Parse KITTI calibration files into named matrices and derive camera intrinsics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MATRIX_SHAPES = {
    "P0": (3, 4),
    "P1": (3, 4),
    "P2": (3, 4),
    "P3": (3, 4),
    "R0_rect": (3, 3),
    "Tr_velo_to_cam": (3, 4),
    "Tr_imu_to_velo": (3, 4),
}


def parse_calib_file(path) -> dict[str, np.ndarray]:
    """Parse a KITTI calib .txt file into a dict of reshaped matrices."""
    matrices: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, values = line.partition(":")
        key = key.strip()
        if key not in _MATRIX_SHAPES:
            continue
        nums = np.array([float(v) for v in values.split()], dtype=np.float64)
        matrices[key] = nums.reshape(_MATRIX_SHAPES[key])

    missing = _MATRIX_SHAPES.keys() - matrices.keys()
    if missing:
        raise ValueError(f"calib file {path} missing matrices: {sorted(missing)}")
    return matrices


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_offset: float  # P2[0,3] / -fx; the horizontal shift of cam2 from the rectified reference frame


def left_color_intrinsics(matrices: dict[str, np.ndarray]) -> CameraIntrinsics:
    """Intrinsics for the left color camera (P2), which is what KITTI's image_2/label_2 correspond to."""
    p2 = matrices["P2"]
    fx, fy = p2[0, 0], p2[1, 1]
    cx, cy = p2[0, 2], p2[1, 2]
    baseline_offset = p2[0, 3] / -fx
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, baseline_offset=baseline_offset)


def load_camera_intrinsics(path) -> CameraIntrinsics:
    return left_color_intrinsics(parse_calib_file(path))
