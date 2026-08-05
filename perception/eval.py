"""Evaluate a trained YOLO checkpoint on a KITTI val split and record mAP.

Usage: python -m perception.eval --weights runs/detect/train/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(weights: str, data: str) -> dict:
    model = YOLO(weights)
    metrics = model.val(data=data, split="val")
    box = metrics.box

    # ap_class_index/ap50/ap are only populated for classes with ground truth
    # in the val split; iterating them together keeps per-class labels correct
    # even if a class were ever absent.
    per_class = {
        metrics.names[int(idx)]: {"ap50": float(ap50), "ap50_95": float(ap)}
        for idx, ap50, ap in zip(box.ap_class_index, box.ap50, box.ap)
    }

    return {
        "weights": str(weights),
        "data": str(data),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="path to a trained .pt checkpoint")
    parser.add_argument("--data", default="data/processed/chen.yaml", help="dataset yaml")
    parser.add_argument("--out", default="metrics.json", help="where to write the results json")
    args = parser.parse_args()

    results = evaluate(args.weights, args.data)
    Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"mAP50: {results['map50']:.4f}  mAP50-95: {results['map50_95']:.4f}")
    for cls, m in results["per_class"].items():
        print(f"  {cls}: AP50={m['ap50']:.4f}  AP50-95={m['ap50_95']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
