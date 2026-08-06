"""Validates configs/*.yaml without importing torch/ultralytics.

CI installs only numpy/pyyaml/pytest/ruff (see requirements-ci.txt) — no GPU
stack — so these tests only check each config's shape and internal
consistency, not that training actually runs.
"""
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

# Each config's own probed VRAM ceiling on this GTX 1650 (4.29 GB). Both
# happen to land on 16, but from independent probes at different resolutions
# (square 640 for the baseline, rect 1024 for the fine-tune) — don't assume
# one probe justifies the other if either config's imgsz/rect changes.
PROBED_BATCH_CEILINGS = {
    "train_baseline.yaml": 16,
    "finetune_rect1024.yaml": 16,
}

REQUIRED_KEYS = {"model", "data", "imgsz", "epochs", "batch", "workers"}

CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))
CONFIG_IDS = [p.name for p in CONFIG_PATHS]


@pytest.fixture(params=CONFIG_PATHS, ids=CONFIG_IDS)
def config_path(request) -> Path:
    return request.param


@pytest.fixture
def config(config_path) -> dict:
    return yaml.safe_load(config_path.read_text())


def test_at_least_one_config_found():
    assert CONFIG_PATHS, f"no yaml configs found in {CONFIG_DIR}"


def test_required_keys_present(config):
    missing = REQUIRED_KEYS - config.keys()
    assert not missing, f"config is missing required keys: {missing}"


def test_imgsz_is_a_plain_int(config):
    # Ultralytics 8.3.143 rejects a [h, w] list for train/val imgsz (only
    # predict/export accept that) — see checks.check_imgsz. A future edit
    # toward a non-square shape must fail here, not partway into a real run.
    # This holds even for rect configs: rect derives its own per-batch
    # rectangular shape from a single scalar imgsz, it doesn't take one.
    assert isinstance(config["imgsz"], int)


def test_batch_and_epochs_are_positive(config):
    assert config["batch"] > 0
    assert config["epochs"] > 0


def test_batch_matches_its_probed_vram_limit(config, config_path):
    ceiling = PROBED_BATCH_CEILINGS.get(config_path.name)
    assert ceiling is not None, f"no probed batch ceiling recorded for {config_path.name}"
    assert config["batch"] <= ceiling


def test_workers_is_low_for_constrained_system_ram(config):
    # Free system RAM on this machine has ranged 0.9-2.9 GB across probes;
    # the Ultralytics default of 8 dataloader workers was expected to thrash.
    assert config["workers"] <= 4


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed" / "chen.yaml").exists(),
    reason="processed KITTI dataset not present locally",
)
def test_data_yaml_referenced_by_config_exists(config):
    assert (REPO_ROOT / config["data"]).exists()
