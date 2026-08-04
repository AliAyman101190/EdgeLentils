from pathlib import Path

import numpy as np
import pytest

from perception.calibration import left_color_intrinsics, load_camera_intrinsics, parse_calib_file

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CALIB_FILE = REPO_ROOT / "data" / "raw" / "training" / "calib" / "000000.txt"

# Verbatim content of data/raw/training/calib/000000.txt, inlined so these tests
# don't depend on the gitignored raw dataset being present.
CALIB_000000 = """\
P0: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 0.000000000000e+00 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.797842000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03
P3: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.341081000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 2.330660000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.201153000000e-03
R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01
Tr_velo_to_cam: 6.927964000000e-03 -9.999722000000e-01 -2.757829000000e-03 -2.457729000000e-02 -1.162982000000e-03 2.749836000000e-03 -9.999955000000e-01 -6.127237000000e-02 9.999753000000e-01 6.931141000000e-03 -1.143899000000e-03 -3.321029000000e-01
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
"""


def test_parse_calib_file_shapes_and_values(tmp_path):
    calib_path = tmp_path / "000000.txt"
    calib_path.write_text(CALIB_000000)

    matrices = parse_calib_file(calib_path)

    assert set(matrices) == {"P0", "P1", "P2", "P3", "R0_rect", "Tr_velo_to_cam", "Tr_imu_to_velo"}
    assert matrices["P2"].shape == (3, 4)
    assert matrices["R0_rect"].shape == (3, 3)
    assert matrices["Tr_velo_to_cam"].shape == (3, 4)
    np.testing.assert_allclose(matrices["P2"][0, 0], 707.0493, atol=1e-3)


def test_parse_calib_file_missing_matrix_raises(tmp_path):
    calib_path = tmp_path / "incomplete.txt"
    calib_path.write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")

    with pytest.raises(ValueError, match="missing matrices"):
        parse_calib_file(calib_path)


def test_left_color_intrinsics(tmp_path):
    calib_path = tmp_path / "000000.txt"
    calib_path.write_text(CALIB_000000)

    intrinsics = left_color_intrinsics(parse_calib_file(calib_path))

    assert intrinsics.fx == pytest.approx(707.0493, abs=1e-3)
    assert intrinsics.fy == pytest.approx(707.0493, abs=1e-3)
    assert intrinsics.cx == pytest.approx(604.0814, abs=1e-3)
    assert intrinsics.cy == pytest.approx(180.5066, abs=1e-3)
    # P2[0,3] / -fx = 45.75831 / -707.0493
    assert intrinsics.baseline_offset == pytest.approx(-0.06471, abs=1e-4)


@pytest.mark.skipif(not REAL_CALIB_FILE.exists(), reason="raw KITTI dataset not present locally")
def test_load_camera_intrinsics_matches_real_file():
    intrinsics = load_camera_intrinsics(REAL_CALIB_FILE)
    assert intrinsics.fx == pytest.approx(707.0493, abs=1e-3)
    assert intrinsics.cx == pytest.approx(604.0814, abs=1e-3)
