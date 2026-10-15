"""Pure CPU expected carried-bottle Newton--Euler wrench and contact-LP preflight.

No SAPIEN/scene/GPU/robot interfaces. This is a rigid frozen-grasp prediction,
not sliding-bottle feedback or physical acceptance. World axes use +Z up.
Contact normals always mean force ON the bottle.

Local differentiation uses only nearest intervals: three-point COM acceleration,
SO(3) world log velocity on each interval, and adjacent interval acceleration.
There is no multi-frame low-pass or Savitzky--Golay fit that can hide a real
piecewise-linear joint-path knot. Peak samples can be included in sparse LP
checks, but only sample_stride=1 checks all discrete scheduled samples.
Endpoints default to one-sided continuation; use endpoint_mode="hold" to include
the acceleration implied by rest immediately before/after the path.
"""
from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation


def _finite(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return a.copy()


def _positive(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive number")
    return value


def _poses(value, label, multiple=False):
    a = np.asarray(value, dtype=float)
    if multiple:
        if a.ndim != 3 or a.shape[1:] != (4, 4) or len(a) < 3:
            raise ValueError(f"{label} must have shape (N>=3,4,4)")
    elif a.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4,4)")
    if not np.isfinite(a).all():
        raise ValueError(f"{label} must be finite")
    a = a.copy()
    p = a if multiple else a[None, ...]
    if not np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError(f"{label} must have homogeneous last rows")
    R = p[:, :3, :3]
    if (np.max(np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3))) > 1e-5
            or np.max(np.abs(np.linalg.det(R) - 1)) > 1e-5):
        raise ValueError(f"{label} rotations must belong to SO(3)")
    return a


def _translation_derivatives(p, t, endpoint_mode):
    """Nearest-neighbor quadratic derivative, preserving discrete velocity jumps."""
    d = np.diff(t)
    secants = np.diff(p, axis=0) / d[:, None]
    v = np.empty_like(p)
    a = np.empty_like(p)
    left, right = d[:-1], d[1:]
    # Three-point derivative at the interior timestamp, for unequal intervals.
    v[1:-1] = ((right[:, None] * secants[:-1]
                 + left[:, None] * secants[1:]) / (left + right)[:, None])
    a[1:-1] = 2 * (secants[1:] - secants[:-1]) / (left + right)[:, None]
    if endpoint_mode == "one_sided":
        v[0] = secants[0] - .5 * d[0] * a[1]
        v[-1] = secants[-1] + .5 * d[-1] * a[-2]
        a[0], a[-1] = a[1], a[-2]
    else:
        # Virtual adjacent rest samples at the first/last interval duration.
        v[0], v[-1] = .5 * secants[0], .5 * secants[-1]
        a[0], a[-1] = secants[0] / d[0], -secants[-1] / d[-1]
    return v, a


def _rotation_derivatives(rotations, t, endpoint_mode):
    d = np.diff(t)
    delta = rotations[1:] @ np.swapaxes(rotations[:-1], 1, 2)
    angle_vectors = Rotation.from_matrix(delta).as_rotvec()
    if np.any(np.linalg.norm(angle_vectors, axis=1) >= math.pi - 1e-6):
        raise ValueError("Adjacent rotations must differ by less than pi; unwrap/resample path")
    interval_w = angle_vectors / d[:, None]  # world/spatial angular velocity
    midpoint = .5 * (t[:-1] + t[1:])
    w, alpha = np.empty((len(t), 3)), np.empty((len(t), 3))
    left, right = d[:-1], d[1:]
    w[1:-1] = ((right[:, None] * interval_w[:-1]
                 + left[:, None] * interval_w[1:]) / (left + right)[:, None])
    alpha[1:-1] = (interval_w[1:] - interval_w[:-1]) / np.diff(midpoint)[:, None]
    if endpoint_mode == "one_sided":
        w[0] = interval_w[0] - .5 * d[0] * alpha[1]
        w[-1] = interval_w[-1] + .5 * d[-1] * alpha[-2]
        alpha[0], alpha[-1] = alpha[1], alpha[-2]
    else:
        w[0], w[-1] = .5 * interval_w[0], .5 * interval_w[-1]
        alpha[0], alpha[-1] = interval_w[0] / d[0], -interval_w[-1] / d[-1]
    return w, alpha, interval_w


def planned_carry_wrench(
    palm_world_matrices, frozen_palm_to_bottle, *,
    cmass_local_matrix, mass_kg, inertia_kg_m2, dt=.004,
    timestamps_s=None, gravity_world_m_s2=(0., 0., -9.8),
    endpoint_mode="one_sided",
):
    """Predict expected COM trajectory and contact wrench about its COM.

    inertia_kg_m2 is PhysxRigidDynamicComponent.inertia: three principal moments
    in the frame specified by cmass_local_pose. Both local COM translation and
    rotation are applied. No density/mesh/inertia/mass rescaling occurs.
    """
    palm = _poses(palm_world_matrices, "palm_world_matrices", multiple=True)
    relative = _poses(frozen_palm_to_bottle, "frozen_palm_to_bottle")
    cmass = _poses(cmass_local_matrix, "cmass_local_matrix")
    mass = _positive(mass_kg, "mass_kg")
    step = _positive(dt, "dt")
    inertia = _finite(inertia_kg_m2, (3,), "inertia_kg_m2")
    if np.any(inertia <= 0):
        raise ValueError("inertia_kg_m2 must have positive principal moments")
    gravity = _finite(gravity_world_m_s2, (3,), "gravity_world_m_s2")
    if endpoint_mode not in ("one_sided", "hold"):
        raise ValueError("endpoint_mode must be one_sided or hold")
    t = (np.arange(len(palm), dtype=float) * step if timestamps_s is None
         else _finite(timestamps_s, (len(palm),), "timestamps_s"))
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps_s must be strictly increasing")
    bottle = palm @ relative
    com_pose = bottle @ cmass
    com = com_pose[:, :3, 3]
    v, a = _translation_derivatives(com, t, endpoint_mode)
    # cmass rotation is constant in the body, hence shares its world omega/alpha.
    w, alpha, interval_w = _rotation_derivatives(com_pose[:, :3, :3], t, endpoint_mode)
    R = com_pose[:, :3, :3]
    world_inertia = (R * inertia[None, None, :]) @ np.swapaxes(R, 1, 2)
    force = mass * (a - gravity)
    torque = np.einsum("nij,nj->ni", world_inertia, alpha)
    gyro = np.cross(w, np.einsum("nij,nj->ni", world_inertia, w))
    torque += gyro
    return {
        "timestamps_s": t, "palm_world_matrices": palm,
        "planned_bottle_world_matrices": bottle, "planned_com_world_m": com,
        "planned_cmass_world_matrices": com_pose,
        "com_velocity_world_m_s": v, "com_acceleration_world_m_s2": a,
        "omega_world_rad_s": w, "alpha_world_rad_s2": alpha,
        "interval_omega_world_rad_s": interval_w,
        "inertia_world_kg_m2": world_inertia,
        "force_target_world_N": force, "torque_target_world_Nm": torque,
        "gyroscopic_torque_world_Nm": gyro,
        "mass_kg": mass, "principal_inertia_kg_m2": inertia,
        "cmass_local_matrix": cmass, "gravity_world_m_s2": gravity,
        "dt": step, "endpoint_mode": endpoint_mode,
        "derivative_scope": "nearest-interval discrete differences; no temporal low-pass",
    }


def _json_arrays(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_arrays(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_arrays(v) for v in value]
    return value


def preflight_contact_lp(
    schedule, initial_contacts_world, *, initial_bottle_world_matrix=None,
    sample_stride=50, include_peak_samples=True, previous_allocation_N=None,
    lp_options=None, include_cases=False,
):
    """Rotate frozen start contacts with expected bottle poses and check their LP.

    Original world points/normals are frozen in bottle-local coordinates, then
    transported by the planned bottle pose. Main fingers are mandatory in LP;
    other contacts are passed through and ignored there. This cannot predict
    a changing fingertip surface, contact loss, or slip. Sparse checking is
    explicitly not continuous/all-sample feasibility. Set sample_stride=1 for
    all discrete samples. All failed cases retain full input/result evidence.
    """
    from purecontact_lp import allocate_contact_forces, CHANNELS, DEFAULT_CAPS_N
    bottle = _poses(schedule["planned_bottle_world_matrices"], "planned bottle", multiple=True)
    t = _finite(schedule["timestamps_s"], (len(bottle),), "schedule timestamps")
    if np.any(np.diff(t) <= 0):
        raise ValueError("schedule timestamps must be strictly increasing")
    com = _finite(schedule["planned_com_world_m"], (len(bottle), 3), "schedule COM")
    F = _finite(schedule["force_target_world_N"], (len(bottle), 3), "schedule force")
    tau = _finite(schedule["torque_target_world_Nm"], (len(bottle), 3), "schedule torque")
    mass = _positive(schedule["mass_kg"], "schedule mass")
    if isinstance(sample_stride, bool) or not isinstance(sample_stride, (int, np.integer)) or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer")
    start = (_poses(initial_bottle_world_matrix, "initial_bottle_world_matrix")
             if initial_bottle_world_matrix is not None else bottle[0])
    local = []
    for c in initial_contacts_world:
        if c.get("channel") not in CHANNELS:
            raise ValueError("unknown contact channel")
        p = _finite(c["point_world_m"], (3,), "contact point")
        n = _finite(c["force_on_bottle_normal_world"], (3,), "contact normal")
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("contact normal cannot be zero")
        local.append((c["channel"], start[:3, :3].T @ (p-start[:3, 3]),
                      start[:3, :3].T @ (n/norm)))
    options = dict(lp_options or {})
    reserved = {"contacts", "com_world_m", "mass_kg", "previous_allocation_N",
                "target_force_world_N", "target_torque_world_Nm"}
    if reserved.intersection(options):
        raise ValueError("lp_options cannot override contacts, COM, mass, targets or previous")
    caps = np.asarray(options.get("max_channel_force_N", DEFAULT_CAPS_N), float)
    indices = set(range(0, len(t), int(sample_stride))) | {len(t)-1}
    peaks = set()
    if include_peak_samples:
        # Include extrema in every component as well as magnitude; do not smooth.
        for array in (schedule["com_acceleration_world_m_s2"], schedule["alpha_world_rad_s2"], F, tau):
            array = np.asarray(array)
            found = {int(extreme(array[:, j])) for j in range(3)
                     for extreme in (np.argmin, np.argmax)}
            found.add(int(np.argmax(np.linalg.norm(array, axis=1))))
            for i in found:
                peaks.update(range(max(0, i-1), min(len(t), i+2)))
        indices.update(peaks)
    indices = sorted(indices)
    previous = previous_allocation_N
    cases, failures, feasible_allocations = [], [], []
    elapsed = time.monotonic()
    maximum_force_residual, maximum_torque_residual = 0., 0.
    for i in indices:
        pose = bottle[i]
        contacts = [{"channel": channel,
                     "point_world_m": (pose[:3, :3] @ point + pose[:3, 3]).tolist(),
                     "force_on_bottle_normal_world": (pose[:3, :3] @ normal).tolist()}
                    for channel, point, normal in local]
        result = allocate_contact_forces(
            contacts, com_world_m=com[i], mass_kg=mass,
            previous_allocation_N=previous, target_force_world_N=F[i],
            target_torque_world_Nm=tau[i], **options)
        case = {"sample_index": i, "timestamp_s": float(t[i]),
                "force_target_world_N": F[i].tolist(), "torque_target_world_Nm": tau[i].tolist(),
                "com_world_m": com[i].tolist(), "contacts": contacts, "allocation": result}
        if include_cases:
            cases.append(case)
        if result["usable"]:
            previous = result["target_normal_forces_N"]
            feasible_allocations.append((i, previous))
            maximum_force_residual = max(maximum_force_residual, result["force_residual_inf_N"])
            maximum_torque_residual = max(maximum_torque_residual, result["torque_residual_inf_Nm"])
        else:
            failures.append(case)
    arrays = np.array([a for _, a in feasible_allocations])
    def peak_stat(array):
        magnitude = np.linalg.norm(array, axis=1); i = int(np.argmax(magnitude))
        return {"sample_index": i, "timestamp_s": float(t[i]), "norm": float(magnitude[i]),
                "vector": np.asarray(array[i]).tolist()}
    summary = {
        "passed": not failures, "checked_count": len(indices), "path_sample_count": len(t),
        "sample_stride": int(sample_stride), "checked_indices": indices,
        "added_peak_indices": sorted(peaks),
        "coverage": "all_discrete_samples" if len(indices) == len(t) else "selected_discrete_samples_only",
        "continuous_feasibility_certified": False,
        "frozen_contact_geometry_only": True,
        "failure_count": len(failures), "failure_reason_counts": dict(Counter(c["allocation"]["reason"] for c in failures)),
        "failed_cases": failures, "first_failure": failures[0] if failures else None,
        "maximum_checked_force_residual_inf_N": maximum_force_residual if len(arrays) else None,
        "maximum_checked_torque_residual_inf_Nm": maximum_torque_residual if len(arrays) else None,
        "peak_COM_acceleration_m_s2": peak_stat(schedule["com_acceleration_world_m_s2"]),
        "peak_alpha_rad_s2": peak_stat(schedule["alpha_world_rad_s2"]),
        "peak_force_target_N": peak_stat(F), "peak_torque_target_Nm": peak_stat(tau),
        "elapsed_s": time.monotonic()-elapsed, "channel_order": list(CHANNELS),
        "allocation_min_N": arrays.min(0).tolist() if len(arrays) else None,
        "allocation_max_N": arrays.max(0).tolist() if len(arrays) else None,
        "maximum_cap_utilization": (float(np.max(arrays[:, 2:][:, caps[2:]>0] / caps[2:][caps[2:]>0]))
                                    if len(arrays) and np.any(caps[2:]>0) else (0. if len(arrays) else None)),
        "final_previous_allocation_N": previous,
        "scope": "pure CPU rigid expected-trajectory contact wrench; physical slip/contact guards remain mandatory",
    }
    if include_cases:
        summary["cases"] = cases
    return _json_arrays(summary)


if __name__ == "__main__":
    import json
    import sys
    request = json.load(sys.stdin)
    result = planned_carry_wrench(**request)
    print(json.dumps(_json_arrays(result), indent=2))

