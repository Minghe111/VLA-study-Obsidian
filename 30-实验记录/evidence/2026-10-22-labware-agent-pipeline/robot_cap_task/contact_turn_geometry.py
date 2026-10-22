"""Arm-target arithmetic for a proposed small, contact-driven cap turn.

This module only returns a *robot palm target*. It never changes a cap/body
state, adds a force/torque, or assumes that measured contact will stay rigid.
The physical caller must re-read actual cap motion and contact after execution.
"""
import math
import numpy as np


def _pose(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite homogeneous pose")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5, rtol=0) or abs(np.linalg.det(rotation)-1) > 1e-5:
        raise ValueError("Expected a proper measured rotation")
    return matrix


def small_contact_turn_palm_target(body_world, cap_world, palm_world, *,
                                   delta_angle_rad, pitch_m_per_turn=.0048):
    """Recompute from current measured frames; no rigid-grasp success claim.

    Positive opening is rotation about actual body +Z. Rotation is about the
    cap axis, so a palm away from that axis must translate along an orbit;
    rotating the wrist orientation in place would be a different motion.
    """
    body = _pose(body_world)
    cap = _pose(cap_world)
    palm = _pose(palm_world)
    angle = float(delta_angle_rad)
    if not math.isfinite(angle) or not 0 < angle <= .1:
        raise ValueError("Only a bounded positive opening increment is supported")
    if not math.isfinite(pitch_m_per_turn) or pitch_m_per_turn <= 0:
        raise ValueError("Expected positive engineering pitch")
    axis = body[:3, 2]
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)
    proposed_cap = cap.copy()
    proposed_cap[:3, :3] = rotation@cap[:3, :3]
    proposed_cap[:3, 3] += axis*(pitch_m_per_turn*angle/(2*math.pi))
    measured_cap_to_palm = np.linalg.inv(cap)@palm
    target = proposed_cap@measured_cap_to_palm
    radial = palm[:3, 3]-cap[:3, 3]
    radial -= axis*float(axis@radial)
    return target, {"delta_angle_rad": angle,
                    "proposed_axial_motion_m": pitch_m_per_turn*angle/(2*math.pi),
                    "measured_palm_orbit_radius_m": float(np.linalg.norm(radial)),
                    "palm_translation_m": float(np.linalg.norm(target[:3, 3]-palm[:3, 3])),
                    "measured_cap_to_palm": measured_cap_to_palm.tolist(),
                    "cap_target_is_arithmetic_only": True,
                    "object_state_written": False,
                    "actual_cap_turn_or_contact_stick_inferred": False}
