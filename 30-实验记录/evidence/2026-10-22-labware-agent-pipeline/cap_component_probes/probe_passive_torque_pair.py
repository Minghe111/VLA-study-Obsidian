"""Paired declared external-wrench bench tests; NOT robot or dataset examples."""
from pathlib import Path
import gzip
import importlib.util
import json
import math

import numpy as np
import sapien
from transforms3d.quaternions import quat2mat

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("passive_torque_pair_cap", ROOT / "robot_cap_assembly.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trial(dt, torque):
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(dt)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.5, .5, .01], material=sapien.physx.PhysxMaterial(.5, .5, 0.))
    builder.initial_pose = sapien.Pose([0, 0, -.01])
    builder.build_static("ordinary_support_table")
    assembly = module.RobotCapAssembly(scene, pose=sapien.Pose([0, 0, .0005]))
    for _ in range(round(1. / dt)):
        assembly.before_step(); scene.step(); assembly.after_step()
    baseline = assembly.rotation_measured
    start = len(assembly.trace)
    external_work = 0.
    for _ in range(round(.2 / dt)):
        previous = assembly.rotation_measured
        assembly.before_step()
        axis = quat2mat(assembly.body.pose.q)[:, 2]
        # The test wrench is declared and applied to the actual dynamic bodies.
        # It is not a cap motor, hand, robot controller, or robot-training data.
        module.rigid(assembly.cap).add_force_torque(np.zeros(3), axis * torque)
        module.rigid(assembly.body).add_force_torque(np.zeros(3), -axis * torque)
        scene.step(); assembly.after_step()
        external_work += torque * (assembly.rotation_measured - previous)
    rows = assembly.trace[start:]
    delta = assembly.rotation_measured - baseline
    bounds_valid = all(abs(r["friction_torque_Nm"]) <= r["friction_static_limit_Nm"] + 1e-12 for r in rows)
    assert bounds_valid
    assert all(r["friction_damping_dissipation_W"] >= 0. and r["friction_plastic_loss_this_step_J"] >= 0. for r in rows)
    # Reactions are calculated before a step; trace poses are measured after it.
    # Compare stop torque to the matching pre-step angle, never the next angle.
    assert all(r["tightening_stop_torque_Nm"] == 0. for r in rows if r["reaction_evaluated_at_theta_rad"] > 0.)
    spin_at_wrench_end = assembly.rotation_measured
    for _ in range(round(.6 / dt)):
        assembly.before_step(); scene.step(); assembly.after_step()
    return {
        "dt_s": dt, "external_torque_Nm": torque, "external_torque_duration_s": .2,
        "initial_spin_rad": baseline, "spin_at_wrench_end_rad": spin_at_wrench_end,
        "after_zero_wrench_spin_rad": assembly.rotation_measured, "zero_wrench_followup_s": .6,
        "rotation_change_rad": delta, "external_relative_work_J": external_work,
        "friction_bound_max_Nm": max(r["friction_static_limit_Nm"] for r in rows),
        "friction_torque_abs_max_Nm": max(abs(r["friction_torque_Nm"]) for r in rows),
        "any_sliding": any(r["friction_sliding"] for r in rows),
        "friction_plastic_loss_J": sum(r["friction_plastic_loss_this_step_J"] for r in rows),
        "friction_damping_loss_J": sum(r["friction_damping_dissipation_W"] * dt for r in rows),
        "friction_bounds_valid": bounds_valid, "release_occurred": assembly.released,
        "all_motor_properties_zero": assembly.contract()["no_rotation_target_or_motor"],
        "frozen_inputs_unchanged": assembly.contract()["frozen_inputs_unchanged"],
        "not_robot_data": True,
    }, assembly.trace


def main():
    output = ROOT / "cap_component_probes/passive_torque_pair_final"
    output.mkdir(exist_ok=False)
    cases = []
    for dt in (.001, .004):
        pair = []
        for name, torque in (("small", .0002), ("large", .003)):
            result, trace = trial(dt, torque)
            result["label"] = name
            pair.append(result); cases.append(result)
            with gzip.open(output / (str(dt).replace(".", "p") + "_" + name + "_trace.json.gz"), "wt") as stream:
                json.dump(trace, stream, indent=2)
        assert abs(pair[0]["rotation_change_rad"]) < .03, "Small torque did not remain in micro-slip range"
        assert pair[1]["rotation_change_rad"] > 1., "Large torque did not overcome finite friction"
        assert pair[1]["after_zero_wrench_spin_rad"] > 1., "Large slipped angle was servoed back toward zero"
    result = {"date": "2026-10-22", "passed": True,
              "scope": "actual CPU PhysX small/large direct external-wrench component bench; not robot manipulation or dataset collection",
              "component_sha256": module.sha(ROOT / "robot_cap_assembly.py"),
              "small_torque_micro_slip_threshold_rad": .03, "large_torque_motion_threshold_rad": 1.,
              "cases": cases, "robot_opening_claim": False}
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
