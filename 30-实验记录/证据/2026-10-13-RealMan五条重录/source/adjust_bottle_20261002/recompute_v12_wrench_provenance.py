"""Read-only CPU re-computation of v12 saved planned carried-bottle wrench.

Call recompute_v12_wrench_provenance(task, out_dir) with the ALREADY EXISTING
simulation task after its physics_trace.json has been finalized. This module
does not create a scene, enqueue work, step physics, drive joints or do robot IO.
Only a newly constructed independent FullMeshPathAuditor MPlib model is posed.

planned_carry_path_N -> actual dense_call N+1. Freeze all 36 recorded DOFs from
the last trace sample before that execution, replace only planned left-arm
rows, apply that segment's saved planning_reference, and recompute dt=.004
endpoint_mode="hold" from unchanged PhysX mass/inertia/cmass. Saved F/torque
arrays are compared elementwise. Initial cumulative physical-slip acceptance
is external and is not replaced/rebased by this provenance audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

EXPECTED_WRENCH_SOURCE_SHA256 = "85ed08200cec7ec384868bcae4cb69fe64aef56e4461ed938a035ee3287ed9ca"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix(pose):
    return np.asarray(pose.to_transformation_matrix(), dtype=float).copy()


def _component(task):
    # Avoid importing SAPIEN or constructing an engine: use the existing body.
    matches = [c for c in task.bottle.actor.get_components()
               if hasattr(c, "mass") and hasattr(c, "inertia") and hasattr(c, "cmass_local_pose")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one existing dynamic bottle body")
    return matches[0]


def _snapshot(task):
    art = task.robot.left_entity
    body = _component(task)
    return {
        "articulation_qpos_rad": np.asarray(art.get_qpos(), float).copy(),
        "articulation_qvel_rad_s": np.asarray(art.get_qvel(), float).copy(),
        "root_world_matrix": _matrix(art.root_pose),
        "bottle_world_matrix": _matrix(task.bottle.get_pose()),
        "mass_kg": float(body.mass),
        "inertia_kg_m2": np.asarray(body.inertia, float).copy(),
        "cmass_local_matrix": _matrix(body.cmass_local_pose),
        "component_type": type(body).__module__ + "." + type(body).__name__,
    }


def _unchanged(first, last):
    changed = []
    for key, value in first.items():
        if isinstance(value, np.ndarray):
            if not np.array_equal(value, last[key]):
                changed.append(key)
        elif value != last[key]:
            changed.append(key)
    return {"passed": not changed, "changed_fields": changed,
            "comparison": "Exact arrays/scalars; no pose, qpos, velocity, mass, inertia or COM write allowed"}


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value, key=str)]
    return value


def _boundary_record(trace, path_number):
    actual_dense = path_number + 1
    eligible = [r for r in trace
                if isinstance(r.get("phase", {}).get("dense_call"), int)
                and r["phase"]["dense_call"] < actual_dense]
    if not eligible:
        raise ValueError(f"No prior full-DOF physics frame for path{path_number}->dense{actual_dense}")
    record = eligible[-1]
    if record["phase"]["dense_call"] != path_number:
        raise ValueError(f"Boundary is dense{record['phase']['dense_call']}, expected dense{path_number}")
    if "full_qpos_actual_rad" not in record:
        raise ValueError("Boundary trace lacks complete actual DOFs")
    return record


def _reordered_q(record, recorded_joint_names, model_joint_names):
    if len(set(recorded_joint_names)) != len(recorded_joint_names):
        raise ValueError("Recorded joint names contain duplicates")
    if set(recorded_joint_names) != set(model_joint_names):
        raise ValueError("Trace/native full-DOF joint sets differ")
    q = np.asarray(record["full_qpos_actual_rad"], float)
    if q.shape != (len(recorded_joint_names),) or not np.isfinite(q).all():
        raise ValueError("Invalid recorded complete joint vector")
    lookup = {n: i for i, n in enumerate(recorded_joint_names)}
    return q[[lookup[n] for n in model_joint_names]].copy()


def _difference(expected, saved, label, tolerance):
    a, b = np.asarray(expected, float), np.asarray(saved, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{label} shape/finiteness mismatch")
    delta = a-b
    flat = int(np.argmax(np.abs(delta)))
    row, component = np.unravel_index(flat, delta.shape)
    return {
        "passed": bool(np.max(np.abs(delta)) <= tolerance),
        "shape": list(a.shape), "max_abs_error": float(np.max(np.abs(delta))),
        "max_vector_error": float(np.linalg.norm(delta, axis=1).max()),
        "rms_error": float(np.sqrt(np.mean(delta**2))),
        "worst_sample": int(row), "worst_component": int(component),
        "expected_at_worst": float(a[row, component]), "saved_at_worst": float(b[row, component]),
        "absolute_tolerance": float(tolerance),
    }


def _pose_difference(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    u, _, vt = np.linalg.svd(a[:3, :3] @ b[:3, :3].T)
    return {
        "translation_m": float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
        "rotation_rad": float(np.arccos(np.clip((np.trace(u@vt)-1)/2, -1, 1))),
    }


def recompute_v12_wrench_provenance(
    task, out_dir, *, base_dir=None, trace_path=None, report_path=None,
    force_tolerance_N=1e-10, torque_tolerance_Nm=1e-12,
    expected_wrench_source_sha256=EXPECTED_WRENCH_SOURCE_SHA256,
    save=True,
):
    """Use the existing task only as a read-only source of real model/body data.

    Saved report_path defaults to a NEW provenance report in out_dir. Existing
    reports are not overwritten. No new physical initialization is performed.
    Original canonical source hashes and saved bottle readback must still match.
    """
    started = time.monotonic()
    output = Path(out_dir).resolve()
    base = Path(base_dir).resolve() if base_dir is not None else output.parent
    trace_file = Path(trace_path) if trace_path is not None else output/"physics_trace.json"
    target_report = Path(report_path) if report_path is not None else output/"wrench_provenance_recomputed_20261003.json"
    if save and target_report.exists():
        raise FileExistsError(f"Refusing to overwrite {target_report}")
    for tolerance in (force_tolerance_N, torque_tolerance_Nm):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Positive finite comparison tolerances required")
    attempt_file = output/"attempt_report.json"
    attempt = json.loads(attempt_file.read_text(encoding="utf-8"))
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    if not trace or any(trace[i]["step"] >= trace[i+1]["step"] for i in range(len(trace)-1)):
        raise ValueError("Finalized trace must have strictly increasing physics steps")
    saved_numbers = sorted(int(p.stem.rsplit("_",1)[1]) for p in output.glob("planned_wrench_*.npz"))
    if not saved_numbers:
        raise ValueError("No saved planned wrench segment to recompute")
    recorded_names = attempt.get("full_qpos_joint_names")
    if not recorded_names:
        raise ValueError("Attempt report lacks full_qpos_joint_names")

    import full_mesh_path_audit as fma
    import planned_carry_wrench as pcw
    source_files = {"provenance_helper": Path(__file__), "planned_carry_wrench": Path(pcw.__file__),
                    "full_mesh_path_audit": Path(fma.__file__)}
    if (expected_wrench_source_sha256 is not None
            and _sha(source_files["planned_carry_wrench"]) != expected_wrench_source_sha256.lower()):
        raise RuntimeError("Current wrench helper does not match the audited CPU version")
    canonical_checks = {}
    for name, expected in attempt.get("source_hashes",{}).items():
        path = Path(name)
        canonical_checks[name] = {"expected_sha256": expected, "actual_sha256": _sha(path),
                                 "passed": _sha(path) == expected}
    if not canonical_checks or not all(v["passed"] for v in canonical_checks.values()):
        raise RuntimeError("Original canonical sources changed or no original hashes were saved")

    before = _snapshot(task)
    readback_path = base/"bottle_mass_inertia_cmass_20261003.json"
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    body_matches = (before["mass_kg"] == readback["mass"]
                    and np.array_equal(before["inertia_kg_m2"], np.asarray(readback["inertia"],float))
                    and np.array_equal(before["cmass_local_matrix"], np.asarray(readback["cmass_local_pose"],float)))
    if not body_matches:
        raise RuntimeError("Current original PhysX mass/inertia/cmass differs from saved readback")
    body = _component(task)
    auditor = fma.FullMeshPathAuditor(task)
    old_model_q = np.asarray(auditor.model.get_qpos(), float).copy()
    names = list(auditor.sim_joints)
    arm_names = list(task.robot.left_arm_joints_name)
    arm_indices = [names.index(n) for n in arm_names]
    frozen_indices = [i for i in range(len(names)) if i not in arm_indices]
    results = []
    report = {
        "scope": "Independent CPU FK and expected inertial-wrench provenance only; no physical episode acceptance",
        "out_dir": str(output), "trace_path": str(trace_file), "trace_sha256": _sha(trace_file),
        "attempt_report_sha256": _sha(attempt_file),
        "source_sha256": {name: _sha(path) for name,path in source_files.items()},
        "canonical_source_checks": canonical_checks, "bottle_readback_sha256": _sha(readback_path),
        "original_body_snapshot": before, "original_body_readback_matches": body_matches,
        "recorded_full_joint_names": recorded_names, "model_full_joint_names": names,
        "commanded_left_arm_joint_names": arm_names,
        "frozen_joint_names": [names[i] for i in frozen_indices],
        "fk_model_metadata": auditor.metadata, "segments": results,
        "dt_s": .004, "endpoint_mode": "hold",
    }
    failure = None
    try:
        for number in saved_numbers:
            segment_started = time.monotonic()
            path_file = output/f"planned_carry_path_{number}.npz"
            wrench_file = output/f"planned_wrench_{number}.npz"
            record = _boundary_record(trace, number)
            full = _reordered_q(record, recorded_names, names)
            with np.load(path_file, allow_pickle=False) as data:
                arm_positions = np.array(data["position"], dtype=float, copy=True)
                reference = np.array(data["planning_reference"], dtype=float, copy=True)
            if arm_positions.ndim != 2 or arm_positions.shape[1] != len(arm_indices) or len(arm_positions) < 3:
                raise ValueError("Planned arm path must have >=3 rows matching the actual arm joint count")
            report_reference = attempt.get("segment_planning_references",{}).get(str(number))
            if report_reference is None or not np.array_equal(reference, np.asarray(report_reference,float)):
                raise ValueError("Saved segment planning reference differs between NPZ and report")
            auditor.model.set_qpos(full, full=True)
            boundary_palm = fma._pose_matrix(auditor.objects["l_inspire_base_link"].pose)
            boundary_bottle = np.asarray(record["bottle"]["pose"], float)
            reference_error = _pose_difference(boundary_palm@reference, boundary_bottle)
            palms = []
            for arm_row in arm_positions:
                full[arm_indices] = arm_row
                auditor.model.set_qpos(full, full=True)
                palms.append(fma._pose_matrix(auditor.objects["l_inspire_base_link"].pose))
            schedule = pcw.planned_carry_wrench(
                palms, reference, cmass_local_matrix=before["cmass_local_matrix"],
                mass_kg=before["mass_kg"], inertia_kg_m2=before["inertia_kg_m2"],
                dt=.004, endpoint_mode="hold")
            with np.load(wrench_file, allow_pickle=False) as data:
                saved_force = np.array(data["force_target_world_N"],float,copy=True)
                saved_torque = np.array(data["torque_target_world_Nm"],float,copy=True)
            F = _difference(schedule["force_target_world_N"], saved_force, "force", force_tolerance_N)
            tau = _difference(schedule["torque_target_world_Nm"], saved_torque, "torque", torque_tolerance_Nm)
            results.append({
                "path_number": number, "actual_execution_dense_call": number+1,
                "boundary_trace_step": int(record["step"]), "boundary_phase": record["phase"],
                "boundary_full_qpos_actual_rad": record["full_qpos_actual_rad"],
                "model_reordered_boundary_full_qpos_rad": _reordered_q(record,recorded_names,names),
                "planning_reference": reference, "reference_NPZ_report_exact_match": True,
                "boundary_reference_pose_error": reference_error,
                "boundary_bottle_world_matrix": boundary_bottle,
                "boundary_palm_FK_world_matrix": boundary_palm,
                "frozen_noncommanded_joint_rad": full[frozen_indices].copy(),
                "first_arm_path_vs_record_max_rad": float(np.max(np.abs(arm_positions[0]-_reordered_q(record,recorded_names,names)[arm_indices]))),
                "path_sha256": _sha(path_file), "saved_wrench_sha256": _sha(wrench_file),
                "path_samples": len(arm_positions), "duration_s": (len(arm_positions)-1)*.004,
                "planned_first_bottle_world_matrix": schedule["planned_bottle_world_matrices"][0],
                "planned_last_bottle_world_matrix": schedule["planned_bottle_world_matrices"][-1],
                "planned_first_COM_world_m": schedule["planned_com_world_m"][0],
                "planned_last_COM_world_m": schedule["planned_com_world_m"][-1],
                "force_comparison": F, "torque_comparison": tau,
                "passed": F["passed"] and tau["passed"],
                "recompute_elapsed_s": time.monotonic()-segment_started,
            })
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        auditor.model.set_qpos(old_model_q, full=True)
        after = _snapshot(task)
        report["scene_unchanged_guard"] = _unchanged(before,after)
        report["independent_model_restore_exact"] = np.array_equal(
            np.asarray(auditor.model.get_qpos(),float),old_model_q)
    report["error"] = failure
    report["passed"] = (failure is None and bool(results) and len(results)==len(saved_numbers)
                        and all(r["passed"] for r in results)
                        and report["scene_unchanged_guard"]["passed"]
                        and report["independent_model_restore_exact"])
    report["elapsed_s"] = time.monotonic()-started
    report["checked_segment_count"] = len(results)
    report["max_force_error_N"] = max((r["force_comparison"]["max_abs_error"] for r in results), default=None)
    report["max_torque_error_Nm"] = max((r["torque_comparison"]["max_abs_error"] for r in results), default=None)
    result = _jsonable(report)
    if save:
        target_report.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit("Import recompute_v12_wrench_provenance(task, out_dir) in the existing CPU diagnostic context; no scene/queue is created.")

