"""Pure CPU contact wrench LP for the RealMan simulation adapter.

No SAPIEN, robot, scene, stepping, drive or network interfaces are imported.
Input normals MUST be forces acting on the bottle, in world coordinates:
    point.normal if the bottle is PhysX body0, else -point.normal.
The caller computes world COM as actor_pose @ cmass_local_pose and supplies
only current hand-bottle contacts. Ring/little contacts are ignored; their
output targets are zero. Every main channel must have at least one contact.

LP feasibility certifies a theoretical contact wrench in the supplied geometry,
not measured support, grasp stability, force calibration or rollout success.
On missing contacts/failure a previous target can be returned with usable=False;
its short fallback timeout and slip/lost-contact abort belong to the controller.

Contact schema:
    {"channel": "middle" | "index" | "thumb",
     "point_world_m": [x,y,z],
     "force_on_bottle_normal_world": [nx,ny,nz]}
Result target_normal_forces_N order: little, ring, middle, index, thumb.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

CHANNELS = ("little", "ring", "middle", "index", "thumb")
MAIN_IDS = (2, 3, 4)
DEFAULT_CAPS_N = (0., 0., 1.5, 1.5, 2.5)


def _vector(value, size, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite vector of length {size}")
    return array.copy()


def _number(value, label, lower=0., strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < lower or (strict and value <= lower):
        raise ValueError(f"{label} must be finite and {'>' if strict else '>='} {lower}")
    return value


def _tangent_basis(normal):
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    tangent1 = np.cross(normal, axis)
    tangent1 /= np.linalg.norm(tangent1)
    return tangent1, np.cross(normal, tangent1)


def _fallback(reason, previous, **metadata):
    result = {
        "feasible": False, "usable": False, "reason": reason,
        "stale": previous is not None,
        "target_normal_forces_N": None if previous is None else previous.tolist(),
        "target_source": "none" if previous is None else "previous_unverified",
        "force_residual_inf_N": None, "torque_residual_inf_Nm": None,
    }
    result.update(metadata)
    return result


def allocate_contact_forces(
    contacts: Sequence[Mapping],
    *,
    com_world_m,
    mass_kg,
    previous_allocation_N=None,
    mu=.35,
    cone_edges=16,
    min_main_force_N=.1,
    max_channel_force_N=DEFAULT_CAPS_N,
    continuity_weight=2.,
    total_force_weight=1.,
    gravity_acceleration_m_s2=9.8,
    target_force_world_N=None,
    target_torque_world_Nm=(0., 0., 0.),
    moment_scale_m=.05,
    force_residual_tolerance_N=1e-7,
    torque_residual_tolerance_Nm=1e-8,
    solver_time_limit_s=.25,
):
    """Return a JSON-compatible target and a checked theoretical wrench.

    The 16-edge friction polygon is inscribed in the Coulomb disk, so no
    out-of-cone forces are introduced. Each nonnegative ray coefficient
    equals its contribution to contact normal force. Bounds and continuity
    penalties act on the SUM over every contact point in each channel.

    Objective = total_force_weight * sum(Fn)
              + continuity_weight * sum(abs(Fn - previous_Fn)).
    Default continuity_weight=2 means any feasible previous allocation is
    preferred to merely reducing normal load, without hard-clipping a new
    solution outside the feasible wrench set. No temporal rate guarantee
    is implied; the drive controller must smooth/rate-limit actual commands.
    """
    center = _vector(com_world_m, 3, "com_world_m")
    mass = _number(mass_kg, "mass_kg", strict=True)
    friction = _number(mu, "mu")
    minimum = _number(min_main_force_N, "min_main_force_N")
    continuity = _number(continuity_weight, "continuity_weight")
    force_cost = _number(total_force_weight, "total_force_weight", strict=True)
    gravity = _number(gravity_acceleration_m_s2, "gravity_acceleration_m_s2")
    scale = _number(moment_scale_m, "moment_scale_m", strict=True)
    force_tol = _number(force_residual_tolerance_N, "force_residual_tolerance_N", strict=True)
    torque_tol = _number(torque_residual_tolerance_Nm, "torque_residual_tolerance_Nm", strict=True)
    time_limit = _number(solver_time_limit_s, "solver_time_limit_s", strict=True)
    if isinstance(cone_edges, bool) or not isinstance(cone_edges, (int, np.integer)) or cone_edges < 4:
        raise ValueError("cone_edges must be an integer >=4")
    edges = int(cone_edges)
    caps = _vector(max_channel_force_N, 5, "max_channel_force_N")
    if np.any(caps < 0) or caps[0] != 0 or caps[1] != 0 or np.any(caps[list(MAIN_IDS)] < minimum):
        raise ValueError("caps must disable little/ring and allow the main-channel minimum")
    previous = None
    if previous_allocation_N is not None:
        previous = _vector(previous_allocation_N, 5, "previous_allocation_N")
        if np.any(previous < 0) or previous[0] != 0 or previous[1] != 0:
            raise ValueError("previous allocation must be nonnegative with little/ring zero")
        if np.any(previous > caps + force_tol):
            raise ValueError("previous allocation exceeds current configured channel caps")
    force_target = (_vector(target_force_world_N, 3, "target_force_world_N")
                    if target_force_world_N is not None else np.array([0., 0., mass * gravity]))
    torque_target = _vector(target_torque_world_Nm, 3, "target_torque_world_Nm")
    target = np.r_[force_target, torque_target]
    cleaned = []
    ignored = 0
    for contact in contacts:
        if not isinstance(contact, Mapping):
            raise ValueError("each contact must be a mapping")
        channel = contact.get("channel")
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}")
        if channel in ("little", "ring"):
            ignored += 1
            continue
        point = _vector(contact["point_world_m"], 3, "point_world_m")
        normal = _vector(contact["force_on_bottle_normal_world"], 3, "force_on_bottle_normal_world")
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            raise ValueError("contact normal cannot be zero")
        cleaned.append((CHANNELS.index(channel), point, normal / norm))
    present = {channel for channel, _, _ in cleaned}
    missing = [CHANNELS[channel] for channel in MAIN_IDS if channel not in present]
    common = {
        "channels": list(CHANNELS), "mu": friction, "cone_edges": edges,
        "ignored_ring_little_contacts": ignored,
        "contact_count": len(cleaned), "target_wrench_world": target.tolist(),
    }
    if missing:
        return _fallback("missing_main_contacts", previous, missing_channels=missing, **common)

    ray_count = len(cleaned) * edges
    wrench = np.zeros((6, ray_count))
    normal_sum = np.zeros((5, ray_count))
    for contact_index, (channel, point, normal) in enumerate(cleaned):
        tangent1, tangent2 = _tangent_basis(normal)
        for edge in range(edges):
            angle = 2 * math.pi * edge / edges
            force = normal + friction * (math.cos(angle) * tangent1 + math.sin(angle) * tangent2)
            column = contact_index * edges + edge
            wrench[:3, column] = force
            wrench[3:, column] = np.cross(point - center, force)
            normal_sum[channel, column] = 1.
    group_matrix = normal_sum[list(MAIN_IDS)]
    use_previous = previous is not None and continuity > 0
    variable_count = ray_count + (3 if use_previous else 0)
    a_eq = np.zeros((6, variable_count))
    a_eq[:, :ray_count] = wrench
    a_eq[3:] /= scale
    b_eq = target.copy()
    b_eq[3:] /= scale
    a_ub = np.zeros((6 + (6 if use_previous else 0), variable_count))
    a_ub[:3, :ray_count] = group_matrix
    a_ub[3:6, :ray_count] = -group_matrix
    b_ub = np.r_[caps[list(MAIN_IDS)], -np.full(3, minimum)]
    cost = np.full(variable_count, force_cost)
    if use_previous:
        a_ub[6:9, :ray_count] = group_matrix
        a_ub[9:12, :ray_count] = -group_matrix
        a_ub[6:9, ray_count:] = -np.eye(3)
        a_ub[9:12, ray_count:] = -np.eye(3)
        b_ub = np.r_[b_ub, previous[list(MAIN_IDS)], -previous[list(MAIN_IDS)]]
        cost[ray_count:] = continuity
    try:
        solved = linprog(
            cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=(0., None), method="highs",
            options={"time_limit": time_limit, "primal_feasibility_tolerance": 1e-9,
                     "dual_feasibility_tolerance": 1e-9},
        )
    except Exception as error:
        return _fallback("solver_exception", previous, solver_message=str(error), **common)
    solver_metadata = {"solver_status": int(solved.status), "solver_message": str(solved.message)}
    if not solved.success or solved.x is None or not np.isfinite(solved.x).all():
        reason = "lp_infeasible" if solved.status == 2 else "lp_failure"
        return _fallback(reason, previous, **solver_metadata, **common)

    coefficients = np.asarray(solved.x[:ray_count])
    allocation = normal_sum @ coefficients
    net_wrench = wrench @ coefficients
    residual = net_wrench - target
    force_error = float(np.max(np.abs(residual[:3])))
    torque_error = float(np.max(np.abs(residual[3:])))
    cap_violation = float(np.max(allocation - caps))
    minimum_violation = float(np.max(minimum - allocation[list(MAIN_IDS)]))
    ray_violation = max(0., float(-coefficients.min()))
    point_forces, point_normals, cone_violation = [], [], 0.
    for contact_index, (_, _, normal) in enumerate(cleaned):
        section = slice(contact_index * edges, (contact_index + 1) * edges)
        point_force = wrench[:3, section] @ coefficients[section]
        point_normal = float(coefficients[section].sum())
        tangential = point_force - point_normal * normal
        cone_violation = max(cone_violation, float(np.linalg.norm(tangential) - friction * point_normal))
        point_forces.append(point_force.tolist())
        point_normals.append(point_normal)
    valid = (force_error <= force_tol and torque_error <= torque_tol
             and max(cap_violation, minimum_violation, ray_violation, cone_violation) <= force_tol)
    checks = {
        "force_residual_inf_N": force_error, "torque_residual_inf_Nm": torque_error,
        "cap_violation_N": max(0., cap_violation),
        "minimum_violation_N": max(0., minimum_violation),
        "cone_violation_N": max(0., cone_violation),
        "ray_nonnegative_violation_N": ray_violation,
    }
    if not valid:
        return _fallback("residual_check_failed", previous, **checks, **solver_metadata, **common)
    return {
        "feasible": True, "usable": True, "reason": "feasible_contact_wrench", "stale": False,
        "target_source": "contact_lp", "target_normal_forces_N": allocation.tolist(),
        "normal_force_change_N": None if previous is None else (allocation - previous).tolist(),
        "theoretical_net_wrench_world": net_wrench.tolist(),
        "theoretical_contact_forces_world_N": point_forces,
        "theoretical_contact_normal_forces_N": point_normals,
        "objective": float(solved.fun), "continuity_weight": continuity,
        "total_force_weight": force_cost,
        **checks, **solver_metadata, **common,
    }


if __name__ == "__main__":
    import json
    import sys
    request = json.load(sys.stdin)
    print(json.dumps(allocate_contact_forces(**request), indent=2))

