"""FCWS/LDWS warning logic: simple, documented rule-based thresholds.

Deliberately simple, per the project plan: a single distance+corridor check
for FCWS (not per-class, not speed/reaction-time-based -- that's a real
production-system feature, out of scope here), and a single lane-offset
threshold for LDWS. LDWS is a per-frame instantaneous offset check, not
temporal drift tracking -- the KITTI 2D object detection frames this project
uses are independently-sampled stills, not a continuous drivable video
sequence, so there's no trend to track across frames in this dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

from perception.distance import Detection
from perception.lanes import LaneResult

DEFAULT_FCWS_THRESHOLD_M = 15.0
DEFAULT_LDWS_THRESHOLD_M = 0.3

# Ego-path corridor: FCWS relevance also requires an object to fall within a
# fixed lateral band around the vehicle's own path, not just be close in
# depth -- otherwise a car two lanes over at the same depth as a genuine
# in-lane threat triggers identically (the depth formula is correct; using
# depth alone as the *relevance* filter is not).
#
# Researched to the same bar as CAMERA_HEIGHT_M in distance.py:
#   - KITTI's recording vehicle is a VW Passat B6 (station wagon), body width
#     1.82m (Wikipedia's VW Passat B6 spec page, cross-checked against
#     auto-data.net) -- a real, citable fact about the actual recording rig.
#   - A forward-collision-warning patent states the baseline rule plainly:
#     "a target is predicted to be within the path of a host vehicle if the
#     lateral offset is less than half the width of the host vehicle" --
#     giving a citable core term, half-width = 1.82 / 2 = 0.91m.
#   - Unlike H, no single authoritative figure exists for the *extra* margin
#     beyond bare half-vehicle-width (ISO 15622's public abstract doesn't
#     expose the paywalled standard's actual clearance numbers; lateral
#     clearance figures found in other contexts, e.g. overtaking maneuvers,
#     answer a different question than "is this object in my forward path").
#     SAFETY_MARGIN_M is therefore a reasoned engineering judgment, not a
#     citation -- called out explicitly rather than dressed up as research.
#     It stands in for (a) a target object's own near-edge, not just its
#     detected center point, and (b) the lateral estimate's own uncertainty
#     (which inherits the same relative error as the distance it's derived
#     from -- see the Phase 3 checklist's bucket breakdown).
#
# Documented limitation: this corridor assumes the vehicle is travelling
# STRAIGHT -- not that the road is straight. During an actual lane change or
# turn it will misjudge relevance; there's no steering angle, yaw rate, or
# IMU data available in this dataset to curve the corridor to the vehicle's
# real heading. Given what this project has (independently-sampled detection
# frames, not a continuous driving/IMU product), "always straight" is the
# honestly-scoped baseline, not an oversight -- see EdgeADAS_Project_Plan.md
# Section 3 for the deferred heading-aware follow-up.
EGO_VEHICLE_WIDTH_M = 1.82
SAFETY_MARGIN_M = 0.5
CORRIDOR_HALF_WIDTH_M = EGO_VEHICLE_WIDTH_M / 2 + SAFETY_MARGIN_M


def fcws_warning(distance_m: float, obj_class: str, threshold_m: float = DEFAULT_FCWS_THRESHOLD_M) -> bool:
    """Forward collision warning: True if an object is closer than threshold_m."""
    return distance_m < threshold_m


def in_ego_corridor(lateral_m: float, half_width_m: float = CORRIDOR_HALF_WIDTH_M) -> bool:
    """True if a lateral offset falls within the fixed straight-ahead ego-path corridor."""
    return abs(lateral_m) < half_width_m


def ldws_warning(lane_center_offset_m: float | None, threshold_m: float = DEFAULT_LDWS_THRESHOLD_M) -> bool:
    """Lane departure warning: True if the current lane-center offset exceeds threshold_m.

    Returns False (no warning) rather than raising when offset_m is None
    (lane not found this frame) -- an undetected lane isn't evidence of
    departure, and a live pipeline shouldn't alarm on every dropped frame.
    """
    if lane_center_offset_m is None:
        return False
    return abs(lane_center_offset_m) > threshold_m


@dataclass
class WarningState:
    fcws_triggered: bool
    fcws_objects: list[Detection]  # detections that triggered (distance + in-corridor)
    ldws_triggered: bool
    lane_offset_m: float | None


def evaluate_warnings(
    detections: list[Detection],
    lane_result: LaneResult,
    fcws_threshold_m: float = DEFAULT_FCWS_THRESHOLD_M,
    corridor_half_width_m: float = CORRIDOR_HALF_WIDTH_M,
    ldws_threshold_m: float = DEFAULT_LDWS_THRESHOLD_M,
) -> WarningState:
    """Combined per-frame check. FCWS uses the fixed ego corridor, independent of lane detection."""
    triggered = [
        det
        for det in detections
        if fcws_warning(det.distance_m, det.obj_class, fcws_threshold_m)
        and in_ego_corridor(det.lateral_m, corridor_half_width_m)
    ]
    ldws = ldws_warning(lane_result.center_offset_m, ldws_threshold_m)
    return WarningState(
        fcws_triggered=bool(triggered),
        fcws_objects=triggered,
        ldws_triggered=ldws,
        lane_offset_m=lane_result.center_offset_m,
    )
