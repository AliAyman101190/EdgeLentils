import pytest

from perception.distance import Detection
from perception.lanes import LaneResult
from perception.warnings import (
    CORRIDOR_HALF_WIDTH_M,
    WarningState,
    evaluate_warnings,
    fcws_warning,
    in_ego_corridor,
    ldws_warning,
)


def test_fcws_warning_below_threshold_triggers():
    assert fcws_warning(distance_m=10.0, obj_class="Car", threshold_m=15.0) is True


def test_fcws_warning_above_threshold_does_not_trigger():
    assert fcws_warning(distance_m=20.0, obj_class="Car", threshold_m=15.0) is False


def test_fcws_warning_at_threshold_does_not_trigger():
    assert fcws_warning(distance_m=15.0, obj_class="Car", threshold_m=15.0) is False


def test_in_ego_corridor_dead_ahead_triggers():
    assert in_ego_corridor(0.0, half_width_m=1.41) is True


def test_in_ego_corridor_within_half_width_triggers():
    assert in_ego_corridor(1.0, half_width_m=1.41) is True
    assert in_ego_corridor(-1.0, half_width_m=1.41) is True


def test_in_ego_corridor_beyond_half_width_does_not_trigger():
    assert in_ego_corridor(2.0, half_width_m=1.41) is False
    assert in_ego_corridor(-2.0, half_width_m=1.41) is False


def test_in_ego_corridor_at_half_width_does_not_trigger():
    assert in_ego_corridor(1.41, half_width_m=1.41) is False


def test_corridor_half_width_is_vehicle_footprint_plus_margin():
    # 1.82m VW Passat B6 body width / 2 + 0.5m margin
    assert CORRIDOR_HALF_WIDTH_M == pytest.approx(1.41)


def test_ldws_warning_beyond_threshold_triggers_either_direction():
    assert ldws_warning(0.5, threshold_m=0.3) is True
    assert ldws_warning(-0.5, threshold_m=0.3) is True


def test_ldws_warning_within_threshold_does_not_trigger():
    assert ldws_warning(0.1, threshold_m=0.3) is False


def test_ldws_warning_at_threshold_does_not_trigger():
    assert ldws_warning(0.3, threshold_m=0.3) is False


def test_ldws_warning_none_offset_does_not_trigger():
    assert ldws_warning(None, threshold_m=0.3) is False


def _lane_result(offset_m):
    return LaneResult(left_fit=None, right_fit=None, center_offset_m=offset_m, warp_matrix_inv=None)


def _det(obj_class, distance_m, lateral_m):
    return Detection(
        obj_class=obj_class, bbox=(0.0, 0.0, 10.0, 10.0), distance_m=distance_m, lateral_m=lateral_m
    )


def test_evaluate_warnings_combines_fcws_corridor_and_ldws():
    detections = [
        _det("Car", 8.0, 0.0),  # close, dead ahead -> triggers
        _det("Pedestrian", 30.0, 0.0),  # dead ahead but too far -> no trigger
        _det("Cyclist", 5.0, 3.0),  # close but clearly outside the corridor -> no trigger
        _det("Car", 5.0, 1.0),  # close, inside the corridor -> triggers
    ]

    state = evaluate_warnings(detections, _lane_result(0.5))

    assert isinstance(state, WarningState)
    assert state.fcws_triggered is True
    assert state.fcws_objects == [detections[0], detections[3]]
    assert state.ldws_triggered is True
    assert state.lane_offset_m == 0.5


def test_evaluate_warnings_filters_close_but_out_of_corridor_object():
    # A neighboring-lane car at the same depth as a genuine threat must not trigger.
    detections = [_det("Car", 5.0, 3.0)]

    state = evaluate_warnings(detections, _lane_result(0.0))

    assert state.fcws_triggered is False
    assert state.fcws_objects == []


def test_evaluate_warnings_no_triggers():
    detections = [_det("Car", 30.0, 0.0)]

    state = evaluate_warnings(detections, _lane_result(0.1))

    assert state.fcws_triggered is False
    assert state.fcws_objects == []
    assert state.ldws_triggered is False


def test_evaluate_warnings_fcws_independent_of_missing_lane():
    # FCWS no longer depends on lane detection succeeding at all.
    detections = [_det("Car", 5.0, 0.0)]

    state = evaluate_warnings(detections, _lane_result(None))

    assert state.fcws_triggered is True
    assert state.ldws_triggered is False
