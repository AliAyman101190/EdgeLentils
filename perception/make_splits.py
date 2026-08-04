"""Build train/val split files for the KITTI training set.

Produces two independent splits:
- "chen": the published Chen/3DOP 3712/3769 split (sequence-disjoint, no temporal
  leakage), used as the primary reported split.
- "random": a seeded, class-stratified 80/20 split, kept alongside "chen" to make
  the temporal-leakage gap between the two visible in the writeup.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import requests

from perception.kitti_labels import CLASS_MAP, parse_label_file

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_TRAINING = REPO_ROOT / "data" / "raw" / "training"
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "splits"
IMAGES_DIR = REPO_ROOT / "data" / "processed" / "images"

CHEN_TRAIN_URL = "https://raw.githubusercontent.com/charlesq34/frustum-pointnets/master/kitti/image_sets/train.txt"
CHEN_VAL_URL = "https://raw.githubusercontent.com/charlesq34/frustum-pointnets/master/kitti/image_sets/val.txt"

RANDOM_SEED = 42
VAL_FRACTION = 0.2


def all_frame_ids() -> list[str]:
    return sorted(p.stem for p in (RAW_TRAINING / "label_2").glob("*.txt"))


def fetch_chen_split() -> tuple[list[str], list[str]]:
    return _fetch_ids(CHEN_TRAIN_URL), _fetch_ids(CHEN_VAL_URL)


def _fetch_ids(url: str) -> list[str]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return [line.strip() for line in resp.text.splitlines() if line.strip()]


def frame_class_presence(frame_id: str) -> frozenset[str]:
    objects = parse_label_file(RAW_TRAINING / "label_2" / f"{frame_id}.txt")
    return frozenset(CLASS_MAP[obj.raw_class] for obj in objects if obj.raw_class in CLASS_MAP)


def stratified_random_split(
    frame_ids: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split frames 80/20, keeping each class's per-frame presence rate balanced.

    Groups frames by their exact class-presence signature (e.g. {"Car"},
    {"Car", "Pedestrian"}, ...) and splits each group independently, so rare
    combinations (e.g. Cyclist-only frames) land on both sides proportionally
    rather than being at the mercy of a single global shuffle.
    """
    rng = random.Random(seed)
    groups: dict[frozenset[str], list[str]] = {}
    for frame_id in frame_ids:
        groups.setdefault(frame_class_presence(frame_id), []).append(frame_id)

    train_ids: list[str] = []
    val_ids: list[str] = []
    for group_frames in groups.values():
        shuffled = group_frames[:]
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * val_fraction)
        val_ids.extend(shuffled[:n_val])
        train_ids.extend(shuffled[n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def write_split(name: str, train_ids: list[str], val_ids: list[str]) -> None:
    all_ids = set(all_frame_ids())
    train_set, val_set = set(train_ids), set(val_ids)

    overlap = train_set & val_set
    assert not overlap, f"{name}: train/val overlap ({len(overlap)} frames)"
    covered = train_set | val_set
    assert covered == all_ids, (
        f"{name}: split does not cover all frames "
        f"(missing {len(all_ids - covered)}, extra {len(covered - all_ids)})"
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, ids in (("train", train_ids), ("val", val_ids)):
        out_path = SPLITS_DIR / f"{name}_{split_name}.txt"
        lines = [str(IMAGES_DIR / f"{fid}.png") for fid in ids]
        out_path.write_text("\n".join(lines) + "\n")
        print(f"{out_path.relative_to(REPO_ROOT)}: {len(ids)} frames")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    chen_train, chen_val = fetch_chen_split()
    write_split("chen", chen_train, chen_val)

    frame_ids = all_frame_ids()
    random_train, random_val = stratified_random_split(frame_ids, VAL_FRACTION, RANDOM_SEED)
    write_split("random", random_train, random_val)


if __name__ == "__main__":
    main()
