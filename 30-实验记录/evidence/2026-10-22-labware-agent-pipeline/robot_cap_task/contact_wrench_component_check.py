"""Installed SAPIEN momentum/sign check; no robot or task success claim."""
from pathlib import Path
import hashlib
import json
import numpy as np
import sapien
from contact_wrench_observer import read_contact_wrench

ROOT = Path(__file__).resolve().parent


def main():
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(.004)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.1, .1, .01])
    builder.initial_pose = sapien.Pose([0, 0, -.01])
    floor = builder.build_static("wrench_sign_floor")
    builder = scene.create_actor_builder()
    builder.add_sphere_collision(radius=.01)
    builder.initial_pose = sapien.Pose([0, 0, .012])
    sphere = builder.build("wrench_sign_sphere")
    rigid = sphere.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    rigid.mass = .1
    rigid.linear_damping = 0
    rigid.angular_damping = 0
    rigid.sleep_threshold = 0
    for tick in range(1, 101):
        before = np.asarray(rigid.linear_velocity).copy()
        scene.step()
        # Deliberately use an offset observer origin: upward support force at
        # x=0 must have positive y-axis moment about origin x=+0.02 m.
        args = dict(origin_world_m=[.02, 0, 0], axis_world=[0, 1, 0])
        moving = read_contact_wrench(scene, sphere, **args)
        if moving["loaded_points"]:
            fixed = read_contact_wrench(scene, floor, **args)
            expected = .1*(np.asarray(rigid.linear_velocity)-before)-np.array([0, 0, -9.81])*.1*.004
            measured = np.array(moving["groups"]["all"]["reported_force_world_N"])*.004
            opposite = np.array(fixed["groups"]["all"]["reported_force_world_N"])*.004
            error = float(np.max(abs(measured-expected)))
            assert error < 1e-7
            assert np.allclose(measured, -opposite, atol=1e-12, rtol=0)
            assert moving["groups"]["all"]["reported_axis_moment_Nm"] > 0
            assert abs(moving["groups"]["all"]["reported_axis_moment_Nm"]
                       + fixed["groups"]["all"]["reported_axis_moment_Nm"]) < 1e-12
            report = {"scope": "CPU PhysX observer sign and moment component only; no robot",
                      "passed": True, "actual_steps": tick, "dt_s": float(scene.get_timestep()),
                      "momentum_expected_contact_impulse_Ns": expected.tolist(),
                      "observed_contact_impulse_Ns": measured.tolist(),
                      "max_momentum_residual_Ns": error,
                      "both_contact_body_orders_checked": True,
                      "sphere_readback": moving, "floor_readback": fixed,
                      "robot_turn_or_demo_verified": False,
                      "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/"contact_wrench_observer.py"]}}
            (ROOT/"contact_wrench_component_check.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(json.dumps({"passed": True, "actual_steps": tick,
                              "max_momentum_residual_Ns": error}))
            return
    raise RuntimeError("No actual loaded contact within bounded component run")


if __name__ == "__main__":
    main()
