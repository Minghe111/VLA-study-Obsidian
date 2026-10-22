"""Declared CPU constraint-response test; external test wrench is not a robot.

The test never attempts opening or claims robot-driven cap manipulation.
"""
from pathlib import Path
import importlib.util
import json

import numpy as np
import sapien

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("constraint_probe_robot_cap", ROOT / "robot_cap_assembly.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
scene.set_timestep(.004)
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[.5, .5, .01], material=sapien.physx.PhysxMaterial(.5, .5, .0))
builder.initial_pose = sapien.Pose([0, 0, -.01])
builder.build_static("ordinary_support_table")
assembly = module.RobotCapAssembly(scene, pose=sapien.Pose([0, 0, .0005]), visual=False)
for i in range(500):
    assembly.before_step()
    if 250 <= i < 350:
        # Equal/opposite lateral wrench probes only radial/tilt constraint modes.
        # No rotation about the cap axis, no prescribed cap pose, no motor.
        module.rigid(assembly.cap).add_force_torque([.1, 0, 0], [0, .001, 0])
        module.rigid(assembly.body).add_force_torque([-.1, 0, 0], [0, -.001, 0])
    scene.step()
    assembly.after_step()
radial = max(row["radial_error_m"] for row in assembly.trace)
tilt = max(row["axis_tilt_rad"] for row in assembly.trace)
result = {"date": "2026-10-22", "scope": "actual CPU PhysX component constraint-response fixture; no robot, opening or pouring trial",
          "passed": radial < 1e-4 and tilt < .01 and not assembly.released,
          "applied_test_wrench_steps": 100, "test_force_N": [.1, 0, 0], "test_torque_Nm": [0, .001, 0],
          "equal_opposite_body_wrench": True, "radial_error_max_m": radial, "axis_tilt_max_rad": tilt,
          "no_axial_rotation_motor": assembly.contract()["no_rotation_target_or_motor"],
          "component_sha256": module.sha(ROOT / "robot_cap_assembly.py"), "contract": assembly.contract()}
path = Path(__file__).with_suffix(".json")
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("passed", "radial_error_max_m", "axis_tilt_max_rad", "no_axial_rotation_motor")}, indent=2))
if not result["passed"]:
    raise SystemExit(1)
