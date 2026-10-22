"""External-wrench component bench ONLY: not a robot expert or dataset.

Use the frozen dynamic assembly unchanged. Apply a bounded equal-and-opposite
torque, stop at measured release, then observe free dynamics. There are no
post-initialization pose/velocity writes or joint motors in this bench.
"""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import math
import numpy as np
import sapien
from transforms3d.quaternions import quat2mat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "robot_cap_assembly.py"
EXPECTED_SHA = "0785eb4e2a487f4ec4a7c637011e163ceabcce18bbc990fc17980e86b95708d5"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SHA
spec = importlib.util.spec_from_file_location("frozen_full_release_assembly", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trial(dt):
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(dt)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.5, .5, .01],
                              material=scene.create_physical_material(.5, .5, 0))
    builder.initial_pose = sapien.Pose([0, 0, -.01])
    builder.build_static(name="component_bench_table")
    assembly = module.RobotCapAssembly(scene, pose=sapien.Pose([0, 0, .0005]),
                                      seed=0, visual=False)
    contract = assembly.contract()
    rows = []
    total_work = 0.
    torque_duration = 0.
    release_tick = None

    def step(phase, external_torque=0.):
        nonlocal total_work, torque_duration, release_tick
        prior = assembly.state()
        assembly.before_step()
        axis = quat2mat(assembly.body.pose.q)[:, 2]
        body = module.rigid(assembly.body)
        cap = module.rigid(assembly.cap)
        relative_omega = float((np.asarray(cap.angular_velocity)
                               - np.asarray(body.angular_velocity)) @ axis)
        if external_torque:
            cap.add_force_torque(np.zeros(3), axis * external_torque)
            body.add_force_torque(np.zeros(3), -axis * external_torque)
            torque_duration += dt
        scene.step()
        assembly.after_step()
        current = assembly.state()
        dtheta = current["rotation_measured_rad"] - prior["rotation_measured_rad"]
        total_work += external_torque * dtheta
        if current["released"] and release_tick is None:
            release_tick = current["physics_step"]
        points = np.asarray(assembly.tablet_local_positions())
        rows.append({"phase": phase, "state": current,
                     "external_pair_torque_Nm": external_torque,
                     "pre_step_projected_relative_omega_rad_s": relative_omega,
                     "measured_theta_increment_rad": dtheta,
                     "theta_increment_minus_omega_dt_rad": dtheta-relative_omega*dt,
                     "external_rotation_work_J": total_work,
                     "tablet_local_positions_m": points.tolist(),
                     "body_linear_velocity_m_s": np.asarray(body.linear_velocity).tolist(),
                     "body_angular_velocity_rad_s": np.asarray(body.angular_velocity).tolist(),
                     "cap_linear_velocity_m_s": np.asarray(cap.linear_velocity).tolist(),
                     "cap_angular_velocity_rad_s": np.asarray(cap.angular_velocity).tolist()})

    for _ in range(round(1./dt)):
        step("settle_no_external_wrench")
    closed_baseline = assembly.state()
    for _ in range(round(1./dt)):
        step("bounded_external_torque_pair", .003)
        if assembly.released:
            break
    last_torque = assembly.state()
    for _ in range(round(.75/dt)):
        step("zero_external_wrench_followup")
    helix_rows = [row for row in rows if not row["state"]["released"]]
    diagnostics = {
        "max_abs_theta_omega_increment_residual_rad": max(
            abs(row["theta_increment_minus_omega_dt_rad"]) for row in helix_rows),
        "sum_theta_omega_increment_residual_rad": sum(
            row["theta_increment_minus_omega_dt_rad"] for row in helix_rows),
    }
    receipt = {"dt_s": dt, "contract_at_start": contract,
               "baseline_after_settle": closed_baseline,
               "external_pair_torque_Nm": .003,
               "actual_torque_duration_s": torque_duration,
               "finite_torque_budget_s": 1.,
               "release_tick": release_tick,
               "release_receipt": assembly.release_receipt,
               "state_at_end_of_external_torque": last_torque,
               "state_after_zero_wrench_followup": assembly.state(),
               "total_external_rotation_work_J": total_work,
               "diagnostics_while_guides_attached": diagnostics,
               "physical_release_observed": bool(assembly.released),
               "states_written_after_initialization": False,
               "robot_present": False, "robot_success": False,
               "usable_robot_demonstration": False,
               "interpretation": "External-wrench mechanical component probe; not contact-driven robot opening, cap handling, or pouring."}
    return receipt, rows


def main():
    out = ROOT / "cap_component_probes" / "passive_full_release_v1"
    out.mkdir(exist_ok=False)
    results = []
    for dt in (.004, .001):
        receipt, rows = trial(dt)
        slug = str(dt).replace(".", "p")
        trace = out / ("trace_dt_" + slug + ".json.gz")
        with gzip.open(trace, "wt") as stream:
            json.dump(rows, stream, allow_nan=False)
        receipt["trace_path"] = str(trace)
        receipt["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
        results.append(receipt)
        print(json.dumps({"dt_s": dt, "released": receipt["physical_release_observed"],
                          "torque_duration_s": receipt["actual_torque_duration_s"],
                          "release_tick": receipt["release_tick"],
                          "state_at_torque_end": receipt["state_at_end_of_external_torque"]}), flush=True)
    payload = {"source_sha256": EXPECTED_SHA,
               "source_still_unchanged": hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SHA,
               "scope": "CPU-only external-wrench mechanical bench. Cannot fulfill robot task acceptance.",
               "trials": results}
    (out / "receipt.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
