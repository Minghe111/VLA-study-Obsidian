"""RoboTwin Robot adapter for the vendor RM65-B dual arm and gen4 Inspire hands.

The task-facing action remains [arm6 rad, scalar synergy, arm6 rad, scalar
synergy]. Explicit SDK6 hand APIs expose independent controls and measured 24-D
state; they do not turn RoboTwin's legacy data/policy pipeline into a 24-D one.
No ROS, sockets or physical robot commands are used here.
"""
from __future__ import annotations

import json
import math
import numbers
from pathlib import Path

import numpy as np

from envs.robot.robot import Robot
from .controller import HandMapping
from .physics_setup import configure_collisions


DEFAULT_SYNERGY = {
    # Archived RealMan SDK demonstration presets, not a grasp calibration.
    "open": [1000, 1000, 1000, 1000, 1000, 0],
    "close": [0, 0, 0, 0, 0, 1000],
}


class RealManRobot(Robot):
    """Reuse official arm planners/drives and independently adapt both hands."""

    preserve_vendor_inertia = True

    def _init_robot_(self, scene, need_topp=False, **kwargs):
        super()._init_robot_(scene, need_topp, **kwargs)
        if not self.is_dual_arm or self.left_entity is not self.right_entity:
            raise ValueError("RealMan gen4 requires its single vendor dual-arm articulation")
        self._manifest = json.loads(
            (Path(kwargs["left_robot_file"]) / "source_manifest.json").read_text(encoding="utf-8"))
        self._model_path = Path(kwargs["left_robot_file"])
        self._hand_config = {
            "left": kwargs["left_embodiment_config"],
            "right": kwargs["right_embodiment_config"],
        }

    def set_planner(self, scene=None):
        super().set_planner(scene)
        from .planner_adapter import configure_planner_world
        self.planner_world_audit = configure_planner_world(self, self._hand_config["left"])

    def reset(self, scene, need_topp=False, **kwargs):
        super().reset(scene, need_topp, **kwargs)
        # Native reset replaces the articulation while reusing CuRobo. MPLib's
        # SapienPlanningWorld owns scene/entity references and must be rebound.
        self._rebind_mplib(scene)
        from .planner_adapter import configure_planner_world
        self.planner_world_audit = configure_planner_world(self, self._hand_config["left"])

    def _rebind_mplib(self, scene):
        if not self.need_topp:
            return
        from envs.robot.planner import MplibPlanner
        for side in ("left", "right"):
            planner = MplibPlanner(
                getattr(self, f"{side}_urdf_path"),
                getattr(self, f"{side}_srdf_path"),
                getattr(self, f"{side}_move_group"),
                getattr(self, f"{side}_entity_origion_pose"),
                getattr(self, f"{side}_entity"),
                getattr(self, f"{side}_planner_type"),
                scene,
            )
            setattr(self, f"{side}_mplib_planner", planner)

    def init_joints(self):
        super().init_joints()
        self._initialize_hands()
        self.physics_audit = configure_collisions(self.left_entity, self._model_path)
        # The vendor arm effort limits are retained in addition to native gains.
        for side in ("left", "right"):
            cfg = self._hand_config[side]
            for joint in getattr(self, f"{side}_arm_joints"):
                joint.set_drive_property(
                    stiffness=cfg.get("joint_stiffness", 1000),
                    damping=cfg.get("joint_damping", 200),
                    force_limit=self._manifest["arm_limits"][joint.get_name()]["effort"],
                    mode="force",
                )
        self.left_entity.set_solver_position_iterations(40)
        self.left_entity.set_solver_velocity_iterations(10)
        self.left_entity.set_sleep_threshold(0)

    def _initialize_hands(self):
        self._hand_joints = {j.get_name(): j for j in self.left_entity.get_active_joints()}
        self._joint_index = {n: i for i, n in enumerate(self._hand_joints)}
        self._hand_maps = {s: HandMapping(self._manifest["hands"][s]) for s in ("left", "right")}
        self._synergy = {}
        self._hand_sdk_target = {}
        contact_links = set(self.gripper_name)
        for side in ("left", "right"):
            cfg = self._hand_config[side]
            presets = cfg.get("hand_synergy", DEFAULT_SYNERGY)
            self._synergy[side] = {}
            for name in ("open", "close"):
                values = presets[name]
                self._hand_maps[side].from_sdk(values, self._measured_positions())
                if any(v == -1 for v in values):
                    raise ValueError("A hand synergy endpoint cannot contain -1")
                self._synergy[side][name] = np.asarray(values, dtype=float)
            if np.array_equal(self._synergy[side]["open"], self._synergy[side]["close"]):
                raise ValueError("Open and close synergy endpoints must differ")
            for name, lim in self._hand_maps[side].limits.items():
                joint = self._hand_joints[name]
                joint.set_drive_property(
                    stiffness=cfg.get("hand_joint_stiffness", 5),
                    damping=cfg.get("hand_joint_damping", 0.1),
                    force_limit=lim["effort"],
                    mode="force",
                )
                contact_links.add(joint.child_link.get_name())
            contact_links.add(self._manifest["hands"][side]["root"])
            contact_links.update(self._manifest["hands"][side]["sensor_links"])
            sdk = self._hand_maps[side].to_sdk(self._measured_positions())
            self._hand_sdk_target[side] = sdk
            setattr(self, f"{side}_gripper_val", self._scalar_projection(side, sdk))
        self.gripper_name = sorted(contact_links)

    @staticmethod
    def _side(side):
        if side not in ("left", "right"):
            raise ValueError("arm_tag must be left or right")
        return side

    def _measured_positions(self):
        qpos = np.asarray(self.left_entity.get_qpos(), dtype=float)
        if qpos.shape != (len(self._hand_joints),) or not np.isfinite(qpos).all():
            raise RuntimeError("Invalid articulation measured joint positions")
        return dict(zip(self._hand_joints, map(float, qpos)))

    def _drive_positions(self):
        positions = {n: float(j.get_drive_target()[0]) for n, j in self._hand_joints.items()}
        if not all(math.isfinite(v) for v in positions.values()):
            raise RuntimeError("Non-finite articulation drive targets")
        return positions

    def _scalar_projection(self, side, sdk):
        start = self._synergy[side]["close"]
        delta = self._synergy[side]["open"] - start
        return float(np.clip(np.dot(np.asarray(sdk) - start, delta) / np.dot(delta, delta), 0, 1))

    def set_hand_angle(self, side, values):
        """Set SDK6 targets; -1 stops that channel at its measured model angle.

        Endpoint conversion follows model limits and vendor direction, not a
        measured SDK encoder calibration. All inputs validate before any drive
        target changes. Only this hand's independent and mimic joints change.
        """
        side = self._side(side)
        targets = self._hand_maps[side].from_sdk(values, self._measured_positions())
        for name, target in targets.items():
            self._hand_joints[name].set_drive_target(target)
            self._hand_joints[name].set_drive_velocity_target(0.0)
        resolved_sdk = self._hand_maps[side].to_sdk(targets)
        self._hand_sdk_target[side] = resolved_sdk
        setattr(self, f"{side}_gripper_val", self._scalar_projection(side, resolved_sdk))
        self._entity_qf(self.left_entity)

    def set_gripper(self, gripper_val, arm_tag, gripper_eps=0.1):
        """Official scalar API: 0=close preset, 1=open preset.

        The native task planner interpolates the scalar over physics steps. The
        optional signed increment follows Robot.set_gripper's drive-target limiting
        convention, then expands each scalar into six channels and mimic joints.
        """
        side = self._side(arm_tag)
        for value in (gripper_val, gripper_eps):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError("Gripper value and increment must be finite real scalars")
        requested = float(np.clip(gripper_val, 0, 1))
        applied = requested
        previous_target = self._scalar_projection(side, self.get_hand_sdk(side, measured=False))
        if ((gripper_eps > 0 and applied - previous_target > gripper_eps)
                or (gripper_eps < 0 and applied - previous_target < gripper_eps)):
            applied = float(np.clip(previous_target + gripper_eps, 0, 1))
        sdk = np.rint(self._synergy[side]["close"] + applied * (
            self._synergy[side]["open"] - self._synergy[side]["close"])).astype(int).tolist()
        self.set_hand_angle(side, sdk)
        # Preserve the official command-based scalar observation contract.
        setattr(self, f"{side}_gripper_val", requested)

    def get_left_gripper_val(self):
        return float(self.left_gripper_val)

    def get_right_gripper_val(self):
        return float(self.right_gripper_val)

    def get_normal_real_gripper_val(self):
        """Measured SDK6 projected onto the scalar preset path, not aperture."""
        return [self._scalar_projection(s, self.get_hand_sdk(s)) for s in ("left", "right")]

    def get_hand_sdk(self, side, measured=True):
        side = self._side(side)
        q = self._measured_positions() if measured else self._drive_positions()
        return self._hand_maps[side].to_sdk(q)

    def get_state24(self, measured=True):
        """[Larm rad6, Lhand model-SDK6, Rarm rad6, Rhand model-SDK6]."""
        q = self._measured_positions() if measured else self._drive_positions()
        result = []
        for side in ("left", "right"):
            result.extend(q[n] for n in getattr(self, f"{side}_arm_joints_name"))
            result.extend(self._hand_maps[side].to_sdk(q))
        return result

    def hand_state(self):
        """Keep measured state, drive targets and task synergy command distinct."""
        measured = self._measured_positions()
        drive = self._drive_positions()
        return {
            "joint_position_rad": measured,
            "joint_target_rad": drive,
            "sdk_angle_estimate": {s: self._hand_maps[s].to_sdk(measured) for s in ("left", "right")},
            "sdk_angle_target": {s: self._hand_maps[s].to_sdk(drive) for s in ("left", "right")},
            "scalar_synergy_command": {s: float(getattr(self, f"{s}_gripper_val")) for s in ("left", "right")},
            "state24": self.get_state24(measured=True),
            "command24": self.get_state24(measured=False),
            "state24_units": (["rad"] * 6 + ["model_sdk_0_1000"] * 6) * 2,
            "sdk_mapping_calibrated": False,
            "tactile_electronics_simulated": False,
            "task_action_representation": "14D arm6+scalar-synergy per side; independent 24D policy/data not enabled",
        }

    def validate_task_action(self, action, action_type="qpos"):
        """Reject mis-sized/full-hand vectors before the legacy task can slice them."""
        if action_type not in ("qpos", "ee"):
            raise ValueError("action_type must be qpos or ee")
        raw = np.asarray(action)
        dimension = 14 if action_type == "qpos" else 16
        if raw.shape != (dimension,):
            raise ValueError(f"RealMan scalar task expects one {dimension}D {action_type} vector; independent SDK6 hands use separate APIs")
        if (not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating)
                or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(action, dtype=object))):
            raise ValueError("Task action values must be real numbers, not bool/string/complex")
        values = raw.astype(float)
        if not np.isfinite(values).all():
            raise ValueError("Task action contains non-finite values")
        arm_dim = 6 if action_type == "qpos" else 7
        for side, offset in (("left", 0), ("right", arm_dim + 1)):
            if not 0 <= values[offset + arm_dim] <= 1:
                raise ValueError("Scalar task hand synergy must be in 0..1")
            if action_type == "qpos":
                for name, value in zip(getattr(self, f"{side}_arm_joints_name"), values[offset:offset + 6]):
                    lim = self._manifest["arm_limits"][name]
                    if not lim["lower"] <= value <= lim["upper"]:
                        raise ValueError(f"Task arm limit violated: {name}")
            else:
                norm = float(np.linalg.norm(values[offset + 3:offset + 7]))
                if not np.isclose(norm, 1.0, atol=1e-3, rtol=0):
                    raise ValueError("Task ee quaternion must be normalized in qw,qx,qy,qz order")
