"""Convert KITTI labels to YOLO format and materialize the Ultralytics dataset layout.

Creates data/processed/images as a junction/symlink to the raw KITTI images (no
5.9 GB copy), writes one converted label file per frame into data/processed/labels,
and emits chen.yaml / random.yaml dataset configs pointing at the splits written by
perception/make_splits.py (run that first).
"""
from __future__ import annotations

import argparse
import platform
import subprocess
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from perception.kitti_labels import YOLO_CLASSES, convert_objects, parse_label_file

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_TRAINING = REPO_ROOT / "data" / "raw" / "training"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
IMAGES_LINK = PROCESSED_DIR / "images"
LABELS_DIR = PROCESSED_DIR / "labels"
SPLITS_DIR = PROCESSED_DIR / "splits"


def ensure_images_link() -> None:
    target = RAW_TRAINING / "image_2"
    if IMAGES_LINK.exists() or IMAGES_LINK.is_symlink():
        return
    IMAGES_LINK.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(IMAGES_LINK), str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to create junction {IMAGES_LINK} -> {target}: {result.stderr.strip()}\n"
                f'Create it manually with: mklink /J "{IMAGES_LINK}" "{target}"'
            )
    else:
        try:
            IMAGES_LINK.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            raise RuntimeError(f"failed to symlink {IMAGES_LINK} -> {target}: {exc}") from exc


def convert_all_labels() -> tuple[Counter, int]:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    class_counts: Counter = Counter()
    label_files = sorted((RAW_TRAINING / "label_2").glob("*.txt"))

    for label_path in label_files:
        frame_id = label_path.stem
        img_path = RAW_TRAINING / "image_2" / f"{frame_id}.png"
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        objects = parse_label_file(label_path)
        rows = convert_objects(objects, img_w, img_h)

        out_path = LABELS_DIR / f"{frame_id}.txt"
        body = "\n".join(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in rows)
        out_path.write_text(body + ("\n" if rows else ""))

        for cid, *_ in rows:
            class_counts[YOLO_CLASSES[cid]] += 1

    return class_counts, len(label_files)


def write_dataset_yaml(name: str) -> None:
    train_split = SPLITS_DIR / f"{name}_train.txt"
    val_split = SPLITS_DIR / f"{name}_val.txt"
    if not (train_split.exists() and val_split.exists()):
        raise FileNotFoundError(
            f"{train_split} / {val_split} not found — run perception/make_splits.py first"
        )

    config = {
        "path": str(PROCESSED_DIR),
        "train": str(train_split.relative_to(PROCESSED_DIR)),
        "val": str(val_split.relative_to(PROCESSED_DIR)),
        "names": {i: cls for i, cls in enumerate(YOLO_CLASSES)},
    }
    out_path = PROCESSED_DIR / f"{name}.yaml"
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    ensure_images_link()
    class_counts, num_frames = convert_all_labels()

    total = sum(class_counts.values())
    print(f"converted labels for {num_frames} frames, {total} boxes")
    for cls in YOLO_CLASSES:
        print(f"  {cls}: {class_counts[cls]}")

    for split_name in ("chen", "random"):
        write_dataset_yaml(split_name)


if __name__ == "__main__":
    main()
