"""Validates configs/train_baseline.yaml without importing torch/ultralytics.

CI installs only numpy/pyyaml/pytest/ruff (see requirements-ci.txt) — no GPU
stack — so this test only checks the config's shape and internal consistency,
not that training actually runs.
"""
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "train_baseline.yaml"

REQUIRED_KEYS = {"model", "data", "imgsz", "epochs", "batch", "workers"}


@pytest.fixture
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_config_file_exists():
    assert CONFIG_PATH.exists()


def test_required_keys_present(config):
    missing = REQUIRED_KEYS - config.keys()
    assert not missing, f"config is missing required keys: {missing}"


def test_imgsz_is_a_plain_int(config):
    # Ultralytics 8.3.143 rejects a [h, w] list for train/val imgsz (only
    # predict/export accept that) — see checks.check_imgsz. A future edit
    # toward a non-square shape must fail here, not partway into a real run.
    assert isinstance(config["imgsz"], int)


def test_batch_and_epochs_are_positive(config):
    assert config["batch"] > 0
    assert config["epochs"] > 0


def test_batch_matches_the_probed_vram_limit(config):
    # Probed on this GTX 1650 (4.29 GB): batch=16 stays inside physical VRAM
    # (~3.9 GB reserved); batch=20 already spilled into WDDM shared memory
    # and thrashed instead of failing cleanly. Don't raise without re-probing.
    assert config["batch"] <= 16


def test_workers_is_low_for_constrained_system_ram(config):
    # Only ~0.9 GB free system RAM when this was pinned; the Ultralytics
    # default of 8 dataloader workers was expected to thrash.
    assert config["workers"] <= 4


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed" / "chen.yaml").exists(),
    reason="processed KITTI dataset not present locally",
)
def test_data_yaml_referenced_by_config_exists(config):
    assert (REPO_ROOT / config["data"]).exists()
