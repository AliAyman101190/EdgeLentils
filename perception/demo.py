"""Full-pipeline visual demo: detect -> distance/lateral -> lanes -> warnings, all on one frame.

Local dev/demo tool only (needs torch/ultralytics for real detections) -- not
part of the CI-tested surface, same status as perception/train.py and
perception/eval.py.

Usage:
    python -m perception.demo --image data/processed/images/000010.png \
        --weights runs/detect/rect1024_finetune/weights/best.pt --out overlay.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from perception.calibration import CameraIntrinsics, load_camera_intrinsics
from perception.distance import CAMERA_HEIGHT_M, Detection, estimate_position_for_box
from perception.kitti_labels import CLASS_MAP, YOLO_CLASSES, parse_label_file
from perception.lanes import LaneResult, detect_lanes
from perception.warnings import CORRIDOR_HALF_WIDTH_M, WarningState, evaluate_warnings

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_on_image(
    image_path: str, weights_path: str, calib_path: str | None = None, conf: float = 0.4
) -> tuple[np.ndarray, CameraIntrinsics, list[Detection], LaneResult, WarningState]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    if calib_path is None:
        calib_path = REPO_ROOT / "data" / "raw" / "training" / "calib" / f"{Path(image_path).stem}.txt"
    intrinsics = load_camera_intrinsics(calib_path)

    model = YOLO(weights_path)
    result = model.predict(img, conf=conf, verbose=False)[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        bbox = tuple(box.xyxy[0].tolist())
        lateral_m, distance_m = estimate_position_for_box(bbox, intrinsics)
        detections.append(Detection(YOLO_CLASSES[cls_id], bbox, distance_m, lateral_m))

    lane_result = detect_lanes(img)
    warning_state = evaluate_warnings(detections, lane_result)

    return img, intrinsics, detections, lane_result, warning_state


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ground_truth_distances(image_path: str) -> list[tuple[tuple[float, float, float, float], float]]:
    """[(bbox, z_true), ...] for this KITTI frame's ground-truth objects (trained classes only).

    Diagnostic-only: reads the raw label file matching image_path's frame id,
    so this only works for KITTI frames with a local raw label (gitignored
    dataset) -- same scope restriction as the rest of this demo tool.
    """
    label_path = REPO_ROOT / "data" / "raw" / "training" / "label_2" / f"{Path(image_path).stem}.txt"
    if not label_path.exists():
        return []
    out = []
    for obj in parse_label_file(label_path):
        if CLASS_MAP.get(obj.raw_class) is None:
            continue
        if obj.location[2] <= 0:
            continue
        out.append((obj.bbox, obj.location[2]))
    return out


def match_ground_truth(
    bbox: tuple[float, float, float, float],
    ground_truths: list[tuple[tuple[float, float, float, float], float]],
    iou_threshold: float = 0.3,
) -> float | None:
    """Best-IoU-matched ground-truth distance for a box, or None if nothing clears the threshold."""
    best_iou, best_z = iou_threshold, None
    for gt_bbox, z_true in ground_truths:
        iou = _iou(bbox, gt_bbox)
        if iou > best_iou:
            best_iou, best_z = iou, z_true
    return best_z


def draw_ego_corridor_overlay(
    img: np.ndarray,
    intrinsics: CameraIntrinsics,
    half_width_m: float = CORRIDOR_HALF_WIDTH_M,
    z_near: float = 3.0,
    z_far: float = 60.0,
    n: int = 30,
) -> np.ndarray:
    """Draw the fixed ego-path corridor as two straight converging lines.

    Projects X = +/-half_width_m at a range of depths back through the same
    pinhole intrinsics used everywhere else in this pipeline. Deliberately
    straight (not curved to the road) -- makes the corridor's straight-line
    assumption visually honest: on a curved-road frame these lines will
    visibly diverge from the actual lane, which is the point, not a bug.
    """
    h, w = img.shape[:2]
    overlay = img.copy()

    left_line, right_line = [], []
    for z in np.linspace(z_near, z_far, n):
        v = intrinsics.cy + intrinsics.fy * CAMERA_HEIGHT_M / z
        if not (0 <= v < h):
            continue
        left_line.append((intrinsics.cx - intrinsics.fx * half_width_m / z, v))
        right_line.append((intrinsics.cx + intrinsics.fx * half_width_m / z, v))

    for line in (left_line, right_line):
        if len(line) >= 2:
            cv2.polylines(overlay, [np.int32(line)], isClosed=False, color=(0, 255, 255), thickness=2)

    return overlay


def draw_lane_lines_overlay(
    img: np.ndarray, lane_result: LaneResult, color: tuple[int, int, int] = (255, 0, 255), thickness: int = 2
) -> np.ndarray:
    """Draw the fitted lane boundaries as two curves (not a filled region).

    Each fitted curve is sampled in bird's-eye (warped) space, then mapped
    back through warp_matrix_inv point-by-point -- so what's drawn is the
    true fitted curvature, not an approximation, and it reads distinctly
    from the straight ego-corridor lines (different color) rather than
    competing with them as a solid fill.
    """
    if lane_result.left_fit is None or lane_result.right_fit is None:
        return img.copy()

    h = img.shape[0]
    overlay = img.copy()
    ys = np.linspace(0, h - 1, 60, dtype=np.float32)

    for fit in (lane_result.left_fit, lane_result.right_fit):
        xs = np.polyval(fit, ys).astype(np.float32)
        pts_warped = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        pts_unwarped = cv2.perspectiveTransform(pts_warped, lane_result.warp_matrix_inv)
        cv2.polylines(overlay, [np.int32(pts_unwarped)], isClosed=False, color=color, thickness=thickness)

    return overlay


def draw_full_overlay(
    img: np.ndarray,
    intrinsics: CameraIntrinsics,
    detections: list[Detection],
    lane_result: LaneResult,
    warning_state: WarningState,
    ground_truths: list[tuple[tuple[float, float, float, float], float]] | None = None,
) -> np.ndarray:
    h = img.shape[0]
    overlay = draw_lane_lines_overlay(img, lane_result)
    overlay = draw_ego_corridor_overlay(overlay, intrinsics)

    triggered_bboxes = {det.bbox for det in warning_state.fcws_objects}

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = (0, 0, 255) if det.bbox in triggered_bboxes else (0, 165, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        gt_z = match_ground_truth(det.bbox, ground_truths) if ground_truths else None
        gt_str = f"{gt_z:.1f}m" if gt_z is not None else "N/A"
        label = f"D:{det.distance_m:.1f}m (GT:{gt_str}) X:{det.lateral_m:+.1f}m"
        cv2.putText(overlay, label, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if lane_result.center_offset_m is not None:
        offset_text = f"Lane offset: {lane_result.center_offset_m:+.2f}m"
    else:
        offset_text = "Lane offset: N/A"
    cv2.putText(overlay, offset_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    banner_y = 30
    if warning_state.fcws_triggered:
        cv2.putText(overlay, "FCWS WARNING", (20, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        banner_y += 35
    if warning_state.ldws_triggered:
        cv2.putText(overlay, "LDWS WARNING", (20, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="path to an input frame")
    parser.add_argument("--weights", required=True, help="path to a trained .pt checkpoint")
    parser.add_argument("--calib", default=None, help="calib file; defaults to the matching KITTI raw calib")
    parser.add_argument("--conf", type=float, default=0.4, help="detector confidence threshold")
    parser.add_argument("--out", required=True, help="path to write the rendered overlay")
    args = parser.parse_args()

    img, intrinsics, detections, lane_result, warning_state = run_on_image(
        args.image, args.weights, calib_path=args.calib, conf=args.conf
    )
    ground_truths = ground_truth_distances(args.image)
    overlay = draw_full_overlay(img, intrinsics, detections, lane_result, warning_state, ground_truths)
    cv2.imwrite(args.out, overlay)

    triggered_bboxes = {det.bbox for det in warning_state.fcws_objects}
    print(f"{len(detections)} detections")
    for det in detections:
        flag = " [FCWS]" if det.bbox in triggered_bboxes else ""
        gt_z = match_ground_truth(det.bbox, ground_truths)
        gt_str = f"{gt_z:.1f}m" if gt_z is not None else "N/A"
        print(f"  {det.obj_class}: D={det.distance_m:.1f}m (GT={gt_str}) X={det.lateral_m:+.1f}m{flag}")
    print(f"FCWS: {'WARNING' if warning_state.fcws_triggered else 'clear'}")
    ldws_status = "WARNING" if warning_state.ldws_triggered else "clear"
    print(f"LDWS: {ldws_status} (offset={warning_state.lane_offset_m})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
