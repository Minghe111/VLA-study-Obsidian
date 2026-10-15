"""Additive, simulation-only bimanual bottle expert for the RGB50 collection.

Official task/robot/vendor assets and frozen RGB20 helpers are read-only.
The original contact controller is extracted and parameterized for the chosen
side; the LP feasible set, drive gains, friction and five-finger timing remain.
No object attachment or object-pose changes are made in this module.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np


DEFAULT_SETTING = {
    "lift_axis": "world", "pitch_deg": 10., "hand_x": 0.,
    "hand_y": .085, "hand_z": .165,
    "open_sdk": [1000, 1000, 1000, 1000, 1000, 0],
    "close_sdk": [650, 670, 610, 610, 887, 0],
    "finger_force_allocation_N": [0., 0., .12, .12, .24],
    "lp_min_main_force_N": 1.15, "feedback_joint_rate_cap_rad_s": .16,
    "hand_joint_stiffness": 20., "hand_joint_damping": .2,
}


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return str(value)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _extract(path, names, context, transformer=None):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise RuntimeError("Frozen helper definitions not found: " + str(names))
    module = ast.Module(body=selected, type_ignores=[])
    if transformer is not None: module = transformer.visit(module)
    ast.fix_missing_locations(module)
    exec(compile(module, str(path) + ":rgb50_additive", "exec"), context)
    return hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()


class _SideTransformer(ast.NodeTransformer):
    """Select a hand without changing controller arithmetic or LP constraints."""
    def __init__(self, side): self.side = side
    def visit_Constant(self, node):
        if self.side == "right" and isinstance(node.value, str):
            value = node.value
            if value == "left": value = "right"
            elif value.startswith("l_"): value = "r_" + value[2:]
            if value != node.value: return ast.copy_location(ast.Constant(value), node)
        return node
    def visit_Attribute(self, node):
        self.generic_visit(node)
        # Both entities reference the same articulation. All side-specific
        # channels/targets, however, must use the selected hand.
        if self.side == "right" and node.attr.startswith("left_"):
            node.attr = "right_" + node.attr[5:]
        return node


def install_initializer(ready, legacy_dir, *, task):
    """Call before setup_demo. Return the original initializer for restoration."""
    import sapien
    from integrations.realman_rm65b_gen4.native_robot import RealManRobot
    legacy = Path(legacy_dir)
    original = RealManRobot._initialize_hands
    context = {"np": np, "sapien": sapien, "RealManRobot": RealManRobot,
               "ready": copy.deepcopy(ready), "BASE": legacy,
               "original_initialize_hands": original, "task": task}
    _extract(legacy / "try_coupled_side_entry_v29.py",
             {"install_hand_coupling", "initialize_coupled_hands"}, context)
    RealManRobot._initialize_hands = context["initialize_coupled_hands"]
    return original


def command24(robot):
    values = []
    for side in ("left", "right"):
        values.extend(float(j.get_drive_target()[0]) for j in getattr(robot, side + "_arm_joints"))
        h = robot._hand_maps[side]
        values.extend(h.to_sdk({n: float(robot._hand_joints[n].get_drive_target()[0]) for n in h.limits}))
    result = np.asarray(values, dtype=float)
    if result.shape != (24,) or not np.isfinite(result).all(): raise RuntimeError("Invalid actual command24")
    return result


def measured24(robot):
    return np.asarray(robot.get_state24(measured=True), dtype=float)


def install_expert(task, sample_spec, output_dir, *, legacy_dir=None):
    """Install after scene setup. Return an ExpertSession (restored by close)."""
    if legacy_dir is None:
        legacy_dir = Path(__file__).resolve().parent.parent / "adjust_bottle_20261002"
    return ExpertSession(task, sample_spec, output_dir, legacy_dir)


class ExpertSession:
    def __init__(self, task, sample_spec, output_dir, legacy_dir):
        import sapien
        import transforms3d as t3d
        from envs._base_task import Base_Task
        from envs.robot.robot import Robot
        from integrations.realman_rm65b_gen4.native_robot import RealManRobot
        from continuous_servo import ContinuousHandMapping, SDKWaypointProfile
        import full_mesh_path_audit as fma
        import full_native_carry_planner_v31 as ncp
        from observed_full_trace_audit_v13 import collect_actual_self_contacts, collect_actual_bottle_contacts

        self.task, self.spec = task, copy.deepcopy(sample_spec)
        self.out, self.legacy = Path(output_dir), Path(legacy_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.side = self.spec.get("arm", self.spec.get("side"))
        if self.side not in ("left", "right"): raise ValueError("sample_spec arm must be left/right")
        if int(task.qpose_tag) != (0 if self.side == "left" else 1):
            raise RuntimeError("Official arm selection disagrees with sampled nearest arm")
        if getattr(task.robot, "communication_flag", False): raise RuntimeError("Physical IO forbidden")
        self.prefix = "l_" if self.side == "left" else "r_"
        self.setting = copy.deepcopy(DEFAULT_SETTING)
        self.setting.update(copy.deepcopy(self.spec.get("grasp_setting", {})))
        self.stage = {"name": "ready", "gripper": False, "dense_call": 0}
        self.steps, self.dense_calls, self.last_capture = 0, 0, -1
        self.traces, self.telemetry, self.sample_metadata, self.hold_rows = [], [], [], []
        self.capture_callback = None
        self.native_record = False
        self.holding = False
        self.reference = None
        self.before = self.bottle_state()
        self.fma, self.ncp, self.t3d, self.sapien = fma, ncp, t3d, sapien
        self.collect_self, self.collect_bottle = collect_actual_self_contacts, collect_actual_bottle_contacts
        self.Base_Task, self.Robot = Base_Task, Robot
        self.original_step, self.original_obs = task.scene.step, task.get_obs
        self.original_dense = task.take_dense_action
        self.originals = []
        self._save_original(task, "choose_grasp_pose")
        self._save_original(task, "get_place_pose")
        self._save_original(task, "move_by_displacement")
        self._save_original(task.robot, self.side + "_plan_path")
        self._save_original(task.robot, self.side + "_plan_grippers")
        self._save_original(task.robot, "set_gripper")
        # Source pin covers controller, exact LP and all collision/planner helpers.
        helper_names = ["try_coupled_side_entry_v29.py", "contact_capture_control_v26.py",
                        "continuous_servo.py", "purecontact_lp_v26.py", "planned_carry_wrench_v26.py",
                        "full_mesh_path_audit.py", "carry_mesh_path_audit.py", "full_native_carry_planner_v31.py",
                        "support_departure_certificate_v20.py", "observed_full_trace_audit_v13.py"]
        self.sources = {str(self.legacy / n): _sha(self.legacy / n) for n in helper_names}
        self.sources[str(Path(__file__).resolve())] = _sha(Path(__file__).resolve())
        context = {"np": np, "sapien": sapien, "MethodType": MethodType,
                   "RealManRobot": RealManRobot, "ContinuousHandMapping": ContinuousHandMapping,
                   "SDKWaypointProfile": SDKWaypointProfile, "stage": self.stage}
        controller_digest = _extract(self.legacy / "contact_capture_control_v26.py",
                                      {"_feedback_rate_limits", "install_contact_capture"},
                                      context, _SideTransformer(self.side))
        self.feedback, self.contact_update = context["install_contact_capture"](task, self.setting)
        self.audit = fma.FullMeshPathAuditor(task)
        self.allowed_self = self.audit.allowed
        self.robot_links = set(self.audit.objects)
        self.hand_names = [n for s in ("left", "right") for n in task.robot._manifest["hands"][s]["limits"]]
        self.hand_ids = [task.robot._joint_index[n] for n in self.hand_names]
        self.hand_limits = {n: lim for s in ("left", "right") for n, lim in task.robot._manifest["hands"][s]["limits"].items()}
        comp = next(c for c in task.bottle.actor.get_components() if hasattr(c, "collision_shapes"))
        self.vertices, _ = fma._link_vertices(comp)
        self.table_top = float(self.audit.static_boxes["table"][1, 2])
        self.baseline_gap = self.bottle_gap()
        self.carry_support_baseline = None
        self.released = False
        self.report = {"result": "running", "accepted": False, "sample_spec": self.spec,
                       "active_arm": self.side, "hardware_io": False,
                       "grasp_setting": self.setting, "source_hashes": self.sources,
                       "controller_overlay_ast_sha256": controller_digest,
                       "official_task_methods_unmodified": True,
                       "before": self.before, "bottle_mass_kg_actual": float(comp.mass),
                       "table_top_z_m": self.table_top, "initial_bottle_gap_m": self.baseline_gap,
                       "full_qpos_joint_names": list(self.audit.sim_joints),
                       "action_clock": {"physics_dt_s": float(task.scene.get_timestep()), "record_every_physics_steps": 15}}
        for side in ("left", "right"):
            task.robot._synergy[side] = {"open": np.asarray(self.setting["open_sdk"], float),
                                       "close": np.asarray(self.setting["close_sdk"], float)}
        setattr(task.robot, self.side + "_plan_grippers", MethodType(self.feedback["plan"], task.robot))
        setattr(task.robot, self.side + "_plan_path", MethodType(lambda robot, *a, **kw: self.plan_path(robot, *a, **kw), task.robot))
        task.choose_grasp_pose = MethodType(lambda current, *a, **kw: self.grasp_pose(*a, **kw), task)
        task.get_place_pose = MethodType(lambda current, *a, **kw: self.place_pose(*a, **kw), task)
        task.move_by_displacement = MethodType(lambda current, *a, **kw: self.lift(*a, **kw), task)
        task.take_dense_action = MethodType(lambda current, *a, **kw: self.dense(*a, **kw), task)
        task.scene.step = self.step
        task.get_obs = self.get_obs

    def _save_original(self, owner, name):
        self.originals.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))

    def bottle_gap(self):
        matrix = self.task.bottle.get_pose().to_transformation_matrix()
        return float((self.vertices @ matrix[:3, :3].T + matrix[:3, 3])[:, 2].min() - self.table_top)

    def bottle_state(self):
        pose = self.task.bottle.get_pose().to_transformation_matrix()
        return {"model_id": int(self.task.model_id), "qpose_tag": int(self.task.qpose_tag),
                "pose": pose.tolist(), "functional_point0": np.asarray(self.task.bottle.get_functional_point(0)).tolist(),
                "official_success": bool(self.task.check_success()),
                "upright_angle_deg": float(np.degrees(np.arccos(np.clip(pose[2, 1], -1, 1))))}

    def grip(self):
        groups = {"thumb": [0., np.zeros(3)], "fingers": [0., np.zeros(3)]}
        raw, table_force = [], 0.
        dt = self.task.scene.get_timestep()
        for contact in self.task.scene.get_contacts():
            names = [b.entity.name for b in contact.bodies]
            if "001_bottle" not in names: continue
            other = names[1] if names[0] == "001_bottle" else names[0]
            for point in contact.points:
                normal = np.asarray(point.normal) * (-1 if names[0] == "001_bottle" else 1)
                force = abs(float(np.dot(point.impulse, normal))) / dt
                if "table" in other.lower(): table_force += force
                if not other.startswith(self.prefix) or force < 1e-6: continue
                group = "thumb" if "thumb" in other else ("fingers" if any(t in other for t in ("index", "middle", "ring", "little")) else None)
                if group is None: continue
                groups[group][0] += force; groups[group][1] += force * normal
                raw.append({"link": other, "normal_force_N": force, "normal_bottle_to_hand": normal.tolist(),
                            "point": np.asarray(point.position).tolist(), "impulse_Ns": np.asarray(point.impulse).tolist()})
        a, b = groups["thumb"][1], groups["fingers"][1]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        dot = float(a @ b / norm) if norm > 1e-12 else 1.
        palm = self.task.robot.left_entity.find_link_by_name(self.prefix + "inspire_base_link").entity.pose.to_transformation_matrix()
        relative = np.linalg.inv(palm) @ self.task.bottle.get_pose().to_transformation_matrix()
        return {"thumb_normal_N": groups["thumb"][0], "fingers_normal_N": groups["fingers"][0],
                "opposing_normals_dot": dot, "table_normal_N": table_force,
                "relative_bottle_palm": relative.tolist(), "contacts": raw}

    def grasp_pose(self, actor, arm_tag, pre_dis=.1, target_dis=0., contact_point_id=None):
        from transforms3d.quaternions import mat2quat
        matrix = actor.get_pose().to_transformation_matrix()
        reflection = np.diag([-1., 1., 1.]) if self.side == "right" else np.eye(3)
        axis = reflection @ matrix[:3, 1]
        # Construct the accepted left geometry in reflected WORLD coordinates.
        # The vendor right mesh is mirrored about PALM local X, so both world
        # reflection and palm reflection are required for a proper rotation.
        if abs(axis[0]) > .1:
            if axis[0] < 0: axis = -axis
        elif axis[2] < 0: axis = -axis
        axis /= np.linalg.norm(axis)
        angle = math.radians(float(self.setting["pitch_deg"]))
        radial = np.array([0., math.cos(angle), -math.sin(angle)])
        radial -= axis * (axis @ radial)
        if np.linalg.norm(radial) < .1:
            radial = np.array([0., 0., -1.]); radial -= axis * (axis @ radial)
        radial /= np.linalg.norm(radial)
        lateral = np.cross(radial, axis)
        canonical_rotation = np.stack([radial, lateral, -axis], axis=1)
        delta = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        palm_rotation = canonical_rotation @ delta
        local_reflection = np.diag([-1., 1., 1.]) if self.side == "right" else np.eye(3)
        rotation = reflection @ palm_rotation @ local_reflection @ delta.T
        center = reflection @ actor.get_contact_point(0, "matrix")[:3, 3]
        canonical_goal = center - (self.setting["hand_z"] - .06) * radial - self.setting["hand_y"] * lateral - self.setting["hand_x"] * axis
        goal = reflection @ canonical_goal
        # The frozen side-entry approaches along the lateral closure clearance.
        # Upright examples retain the same construction in the bottle frame.
        world_lateral = reflection @ lateral
        pre = goal - (pre_dis + .095 - self.setting["hand_y"]) * world_lateral
        self.report["grasp_geometry"] = {"bottle_axis_world": matrix[:3, 1].tolist(), "selected_axis_world": axis.tolist(),
                                         "approach_vector_world": world_lateral.tolist(), "pre_pose": [*pre, *mat2quat(rotation)],
                                         "grasp_pose": [*goal, *mat2quat(rotation)]}
        return [*pre, *mat2quat(rotation)], [*(goal - target_dis * world_lateral), *mat2quat(rotation)]

    def place_pose(self, actor, arm_tag, target_pose, **kwargs):
        if str(arm_tag) != self.side: raise RuntimeError("Unexpected inactive-arm place")
        live = getattr(self.task.robot, "get_" + self.side + "_ee_pose")()
        axis = self.t3d.quaternions.quat2mat(live[-4:])[:, 2]
        kwargs.update(constrain="align", actor_axis=axis.tolist(), actor_axis_type="world",
                      align_axis=[-1. if self.side == "left" else 1., 0., 0.])
        return self.Base_Task.get_place_pose(self.task, actor, arm_tag, target_pose, **kwargs)

    def lift(self, arm_tag, x=0., y=0., z=0., quat=None, move_axis="world"):
        axis = "world" if str(arm_tag) == self.side and move_axis == "arm" and x == y == 0 and z == .1 else move_axis
        return self.Base_Task.move_by_displacement(self.task, arm_tag, x=x, y=y, z=z, quat=quat, move_axis=axis)

    def plan_path(self, robot, target_pose, constraint_pose=None, **kwargs):
        import planned_carry_wrench_v26 as wrench
        index = self.dense_calls
        if index >= 3:
            reference = np.asarray(self.grip()["relative_bottle_palm"])
            result = self.ncp.plan_carry_to_task_pose(self.task, target_pose, self.side, palm_to_bottle=reference,
                    time_step=.004, max_joint_velocity=1., max_joint_acceleration=1., planning_time_s=10.,
                    ik_attempts=24, support_departure=(index == 3))
            if result["status"] == "Success":
                model = self.audit
                full = model.initial_sim_q.copy()
                # Auditor must use the ACTUAL current full state for this segment.
                full = np.asarray(robot.left_entity.get_qpos()).copy()
                indices = [model.sim_joints.index(n) for n in getattr(robot, self.side + "_arm_joints_name")]
                palms = []
                try:
                    for arm_q in result["position"]:
                        full[indices] = arm_q; model.model.set_qpos(full, full=True)
                        palms.append(self.fma._pose_matrix(model.objects[self.prefix + "inspire_base_link"].pose))
                finally: model.model.set_qpos(model.old_model_q, full=True)
                comp = next(c for c in self.task.bottle.actor.get_components() if hasattr(c, "cmass_local_pose"))
                schedule = wrench.planned_carry_wrench(palms, reference, cmass_local_matrix=comp.cmass_local_pose.to_transformation_matrix(),
                        mass_kg=float(comp.mass), inertia_kg_m2=comp.inertia, dt=.004, endpoint_mode="hold")
                inputs = []
                for contact in self.grip()["contacts"]:
                    token = next((t for t in ("middle", "index", "thumb") if t in contact["link"]), None)
                    if token: inputs.append({"channel": token, "point_world_m": contact["point"],
                                             "force_on_bottle_normal_world": (-np.asarray(contact["normal_bottle_to_hand"])).tolist()})
                check = wrench.preflight_contact_lp(schedule, inputs, initial_bottle_world_matrix=self.task.bottle.get_pose().to_transformation_matrix(),
                         sample_stride=1, include_cases=False, previous_allocation_N=self.feedback["force_target_N"], lp_options={"min_main_force_N": 1.15})
                (self.out / f"carry_wrench_preflight_{index}.json").write_text(json.dumps(check, default=_json))
                if not check["passed"]: result = {"status": "Fail", "expected_wrench_preflight_failed": True}
                else:
                    self.feedback.update(wrench_schedule=schedule, wrench_index=0, last_lp_phase=None)
        elif constraint_pose is not None:
            palm = robot._trans_from_gripper_to_endlink(target_pose, arm_tag=self.side)
            result = getattr(robot, self.side + "_mplib_planner").planner.plan_screw(goal_pose=palm,
                     current_qpos=robot.left_entity.get_qpos(), time_step=.004, qpos_step=.005)
        else:
            result = getattr(self.Robot, self.side + "_plan_path")(robot, target_pose, constraint_pose=constraint_pose, **kwargs)
        if result.get("status") == "Success" and index < 3:
            audit = self.fma.audit_path(self.task, result, self.side, min_bottle_clearance_m=.0005,
                                       min_table_clearance_m=0., min_self_clearance_m=0.)
            (self.out / f"mesh_path_{index}.json").write_text(json.dumps(audit, default=_json))
            if not audit["passed"]: result = {"status": "Fail", "full_mesh_audit_failed": True}
        if "position" in result:
            np.savez(self.out / f"planned_path_{index}.npz", position=result["position"], velocity=result["velocity"])
        self.report.setdefault("planning_calls", []).append({"index": index, "side": self.side,
                "status": result.get("status"), "target_pose": np.asarray(target_pose).tolist(),
                "samples": len(result.get("position", [])), "reason": result.get("reason")})
        self.write()
        return result

    def dense(self, sequence, **kwargs):
        self.dense_calls += 1
        closing = sequence.get(self.side + "_gripper") is not None
        phase = "close" if closing else ("approach" if self.dense_calls <= 2 else "after_grasp")
        if self.holding: phase = "after_grasp"
        if phase in ("after_grasp", "hold") and self.reference is None:
            grip = self.grip()
            if not (grip["thumb_normal_N"] > .05 and grip["fingers_normal_N"] > .05 and grip["opposing_normals_dot"] < -.3):
                raise RuntimeError("Carry started without actual opposing fingertip contact")
            self.reference = np.asarray(grip["relative_bottle_palm"])
            self.carry_support_baseline = self.bottle_gap()
            self.report["pre_retreat_grip"] = {"passed": True, "reference_step": self.steps,
                                               "actual": grip}
        self.stage.update(name=phase, gripper=closing, dense_call=self.dense_calls,
                          terminal_hold=self.holding)
        # Fixed simulator clock replaces duplicate native action-boundary frames.
        return self.Base_Task.take_dense_action(self.task, sequence, save_freq=None)

    def get_obs(self):
        obs = self.original_obs()
        obs["recording_clock"] = {"physics_step": self.steps, "sim_time_s": self.steps * self.task.scene.get_timestep()}
        obs["object_audit"] = self.bottle_state()
        dex = dict(obs.get("dexterous_state", {}))
        dex.update(state24=measured24(self.task.robot).tolist(), command24=command24(self.task.robot).tolist(),
                   command_source="actual articulation drive targets")
        obs["dexterous_state"] = dex
        return obs

    def capture(self):
        if self.steps == self.last_capture: return
        self.task._update_render()
        if self.native_record: self.task._take_picture()
        if self.capture_callback is not None: self.capture_callback(self.get_obs(), self.steps)
        self.sample_metadata.append({"physics_step": self.steps, "sim_time_s": self.steps * self.task.scene.get_timestep(),
                                     "measured_state24": measured24(self.task.robot).tolist(),
                                     "command24": command24(self.task.robot).tolist()})
        self.last_capture = self.steps

    def step(self):
        robot, dt = self.task.robot, self.task.scene.get_timestep()
        before = np.asarray(robot.left_entity.get_qpos()).copy()
        before24 = measured24(robot)
        self.contact_update()
        applied = command24(robot)
        targets = np.array([robot._hand_joints[n].get_drive_target()[0] for n in self.hand_names])
        self.original_step(); self.steps += 1
        after = np.asarray(robot.left_entity.get_qpos()).copy()
        velocity = np.asarray(robot.left_entity.get_qvel()).copy(); fd = (after - before) / dt
        state, grip = self.bottle_state(), self.grip()
        self_contacts = self.collect_self(self.task, self.robot_links, self.allowed_self)
        bottle_contacts = self.collect_bottle(self.task)
        positions = robot._measured_positions()
        mimic = max(abs(positions[n] - float(m["multiplier"]) * positions[m["joint"]] - float(m.get("offset", 0)))
                    for hand in robot._manifest["hands"].values() for n, m in hand["mimic"].items())
        excess = max(max(lim["lower"] - positions[n], positions[n] - lim["upper"], 0.) for n, lim in self.hand_limits.items())
        fd_ratio = max(abs(fd[robot._joint_index[n]]) / lim["velocity"] for n, lim in self.hand_limits.items())
        fp = np.asarray(state["functional_point0"][:3]); initial = np.asarray(self.before["functional_point0"][:3])
        delta = float(np.linalg.norm(fp - initial))
        initial_rotation = np.asarray(self.before["pose"])[:3, :3]; rotation = np.asarray(state["pose"])[:3, :3]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(initial_rotation.T @ rotation) - 1) / 2, -1, 1))))
        gap = self.bottle_gap()
        slip = tilt = 0.
        if self.reference is not None:
            relative = np.asarray(grip["relative_bottle_palm"])
            slip = float(np.linalg.norm(relative[:3, 3] - self.reference[:3, 3]))
            tilt = float(np.degrees(np.arccos(np.clip((np.trace(self.reference[:3, :3].T @ relative[:3, :3]) - 1) / 2, -1, 1))))
        self.telemetry.append((self.steps, (self.steps - 1) * dt, self.steps * dt, before24, applied, measured24(robot),
                               before[self.hand_ids], after[self.hand_ids], targets, fd[self.hand_ids], velocity[self.hand_ids], before, after))
        row = {"step": self.steps, "sim_time_s": self.steps * dt, "phase": dict(self.stage),
               "fp_delta_m": delta, "rotation_delta_deg": angle, "bottle": state, "grip": grip,
               "full_qpos_actual_rad": after.tolist(), "arm_qpos": {s: [positions[n] for n in getattr(robot, s + "_arm_joints_name")] for s in ("left", "right")},
               "actual_self_contacts": self_contacts, "bottle_actual_contacts": bottle_contacts,
               "capture_latched": list(self.feedback["latched"]), "capture_filtered_N": self.feedback["filtered_N"].tolist(),
               "capture_actual_pair_streak": self.feedback["actual_pair_streak"], "mimic_residual_rad": mimic,
               "hand_actual_fd_max_rad_s": float(np.max(np.abs(fd[self.hand_ids]))), "command24_actual": applied.tolist(),
               "bottle_bottom_gap_m": gap, "relative_translation_drift_m": slip, "relative_rotation_drift_deg": tilt}
        self.traces.append(row)
        if self.holding: self.hold_rows.append(row)
        if not np.isfinite(after).all() or not np.isfinite(velocity).all(): raise RuntimeError("Nonfinite actual articulation")
        if any(c["active"] for c in self_contacts): raise RuntimeError("Actual native nonexempt self contact")
        if fd_ratio > 1.05 or mimic > .01 or excess > .0001: raise RuntimeError("Hand actual velocity/mimic/limit guard")
        if self.stage["name"] == "approach" and (delta > .001 or angle > 1 or grip["contacts"]): raise RuntimeError("Approach moved/touched bottle")
        if self.stage["name"] in ("close", "close_hold") and (delta > .002 or angle > 2): raise RuntimeError("Closure pushed/rotated bottle")
        if self.reference is not None and (slip > .002 or tilt > 2): raise RuntimeError("Carry slipped beyond 2mm/2deg")
        if self.released and gap < 0.: raise RuntimeError("Bottle intersected table after departure")
        if gap >= .001: self.released = True
        if self.reference is not None and gap < self.carry_support_baseline - .000020: raise RuntimeError("Carry pressed bottle into table")
        if self.steps % 15 == 0: self.capture()

    def hold(self, seconds):
        robot = self.task.robot
        count = int(math.ceil(seconds / self.task.scene.get_timestep()))
        count += (-(self.steps + count)) % 15
        sequence = {"left_arm": None, "right_arm": None, "left_gripper": None, "right_gripper": None}
        for side in ("left", "right"):
            q = np.array([j.get_drive_target()[0] for j in getattr(robot, side + "_arm_joints")])
            sequence[side + "_arm"] = {"position": np.tile(q, (count, 1)), "velocity": np.zeros((count, 6))}
        self.holding = True
        # Terminal static hold uses measured contact forces and unchanged LP.
        self.feedback["wrench_schedule"] = None
        self.dense(sequence)
        self.holding = False

    def write(self):
        (self.out / "expert_report.json").write_text(json.dumps(self.report, indent=2, default=_json), encoding="utf-8")

    def run(self, *, record=False, capture_callback=None, validate_actual_geometry=True):
        self.native_record, self.capture_callback = bool(record), capture_callback
        try:
            self.capture()
            info = self.task.play_once()
            immediate = self.bottle_state()
            if not self.task.plan_success or not self.task.check_success(): raise RuntimeError("Official task failed")
            self.hold(1.)
            final = self.bottle_state()
            hold_pass = bool(len(self.hold_rows) >= 250 and all(
                r["bottle"]["official_success"] and r["bottle_bottom_gap_m"] >= .01
                and r["grip"]["table_normal_N"] < 1e-6 and r["grip"]["thumb_normal_N"] > .05
                and r["grip"]["fingers_normal_N"] > .05 and r["grip"]["opposing_normals_dot"] < -.3
                and r["relative_translation_drift_m"] < .002 and r["relative_rotation_drift_deg"] < 2
                for r in self.hold_rows))
            if not hold_pass: raise RuntimeError("Actual airborne opposing contact did not remain stable for 1s")
            unchanged = all(_sha(p) == digest for p, digest in self.sources.items())
            if not unchanged: raise RuntimeError("Frozen source changed during expert")
            self.report.update(result="physics_success", accepted=True, canonical_sources_unchanged=True,
                physical_steps=self.steps, recorded_frames=len(self.sample_metadata), info=info,
                immediate=immediate, after_hold=final, hold_validation_s=len(self.hold_rows) * self.task.scene.get_timestep(),
                minimum_hold_bottle_gap_m=min(r["bottle_bottom_gap_m"] for r in self.hold_rows),
                hold_passed=hold_pass, adaptive_force_allocations=self.feedback["lp_history"])
            self.save_evidence()
            if validate_actual_geometry:
                from observed_full_trace_audit_v13 import audit_accepted_actual_trace
                # This independent auditor uses side from qpose_tag and the exact
                # historical grasp anchor, not only planned trajectory samples.
                self.report["bottle_attempts"] = [{"before": self.before, "accepted": True}]
                self.report["grasp_reference"] = self.reference.tolist()
                self.write()
                geometry = audit_accepted_actual_trace(self.task, self.out / "physics_trace.json",
                    report_path=self.out / "expert_report.json", output_path=self.out / "actual36_geometry_audit.json")
                if geometry.get("passed") is not True: raise RuntimeError("Independent actual36 geometry audit failed")
                self.report["actual36_geometry_audit"] = {k: v for k, v in geometry.items() if k != "samples"}
            self.write()
            return self.report
        except BaseException as error:
            import traceback
            self.report.update(result="failed", accepted=False, error=str(error), traceback=traceback.format_exc(),
                               physical_steps=self.steps, canonical_sources_unchanged=all(_sha(p) == d for p, d in self.sources.items()))
            self.save_evidence()
            raise
        finally: self.close()

    def save_evidence(self):
        (self.out / "physics_trace.json").write_text(json.dumps(self.traces, default=_json), encoding="utf-8")
        (self.out / "recording_samples.json").write_text(json.dumps(self.sample_metadata, default=_json), encoding="utf-8")
        fields = ("physics_step", "sim_time_before_s", "sim_time_after_s", "measured_state24_before", "applied_command24",
                  "measured_state24_after", "hand24_qpos_before_rad", "hand24_qpos_after_rad", "hand24_drive_target_rad",
                  "hand24_actual_fd_rad_s", "hand24_engine_qvel_rad_s", "full36_qpos_before_rad", "full36_qpos_after_rad")
        if self.telemetry: np.savez_compressed(self.out / "physics_telemetry.npz", **{name: np.asarray([row[i] for row in self.telemetry]) for i, name in enumerate(fields)})
        self.write()

    def close(self):
        self.task.scene.step = self.original_step
        self.task.get_obs = self.original_obs
        self.task.take_dense_action = self.original_dense
        for owner, name, existed, value in reversed(self.originals):
            if existed: owner.__dict__[name] = value
            else: owner.__dict__.pop(name, None)
        try: self.audit.model.set_qpos(self.audit.old_model_q, full=True)
        except Exception: pass


def run_episode(task, sample_spec, output_dir, *, legacy_dir=None, record=False,
                capture_callback=None, validate_actual_geometry=True):
    session = install_expert(task, sample_spec, output_dir, legacy_dir=legacy_dir)
    return session.run(record=record, capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
