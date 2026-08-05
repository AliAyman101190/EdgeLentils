"""Fine-tune YOLOv8n on the KITTI dataset from a single config file.

Usage: python -m perception.train --config configs/train_baseline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    model_name = cfg.pop("model")
    return model_name, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a training config yaml")
    args = parser.parse_args()

    model_name, train_args = load_config(args.config)
    model = YOLO(model_name)
    model.train(**train_args)


if __name__ == "__main__":
    main()
