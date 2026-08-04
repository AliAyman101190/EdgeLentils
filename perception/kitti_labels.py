"""Parse KITTI 2D-detection labels and convert them to YOLO format."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Car/Van/Truck all read as "vehicle obstacle" for an FCWS; Person_sitting is a
# pedestrian in an unusual pose, not a distinct hazard class.
CLASS_MAP: dict[str, str] = {
    "Car": "Car",
    "Van": "Car",
    "Truck": "Car",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
    "Cyclist": "Cyclist",
}

DROPPED_CLASSES = frozenset({"Tram", "Misc", "DontCare"})

YOLO_CLASSES = ("Car", "Pedestrian", "Cyclist")
CLASS_TO_ID = {name: i for i, name in enumerate(YOLO_CLASSES)}


@dataclass(frozen=True)
class KittiObject:
    raw_class: str
    truncated: float
    occluded: int
    alpha: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    dimensions: tuple[float, float, float]  # h, w, l in meters
    location: tuple[float, float, float]  # x, y, z in meters, camera frame
    rotation_y: float


def parse_label_line(line: str) -> KittiObject:
    parts = line.split()
    if len(parts) != 15:
        raise ValueError(f"expected 15 fields in KITTI label line, got {len(parts)}: {line!r}")
    return KittiObject(
        raw_class=parts[0],
        truncated=float(parts[1]),
        occluded=int(parts[2]),
        alpha=float(parts[3]),
        bbox=(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
        dimensions=(float(parts[8]), float(parts[9]), float(parts[10])),
        location=(float(parts[11]), float(parts[12]), float(parts[13])),
        rotation_y=float(parts[14]),
    )


def parse_label_file(path) -> list[KittiObject]:
    return [parse_label_line(line) for line in Path(path).read_text().splitlines() if line.strip()]


def box_to_yolo(
    bbox: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert an absolute pixel box (x1, y1, x2, y2) to normalized YOLO (cx, cy, w, h)."""
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        raise ValueError(f"box {bbox} out of bounds for image {img_w}x{img_h}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box {bbox} is degenerate (non-positive width or height)")
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def convert_objects(
    objects: list[KittiObject], img_w: int, img_h: int
) -> list[tuple[int, float, float, float, float]]:
    """Filter to the mapped classes and convert to YOLO rows (class_id, cx, cy, w, h)."""
    rows = []
    for obj in objects:
        mapped = CLASS_MAP.get(obj.raw_class)
        if mapped is None:
            continue
        cx, cy, w, h = box_to_yolo(obj.bbox, img_w, img_h)
        rows.append((CLASS_TO_ID[mapped], cx, cy, w, h))
    return rows
