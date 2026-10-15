"""Additive simulation-only acquisition before the first carry/holding audit.

Install before constructing an expert. Frozen controller/LP/Session.step files
remain unchanged. Only current drive targets can continue toward the six SDK
targets already resolved from sample_spec. Every extra tick uses Session.step.
The receipt is acquisition evidence, never full-episode or hardware acceptance.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

MAX_EXTRA_STEPS = 500
PHYSICS_DT_S = .004
STABLE_PAIR_STEPS = 50
TARGET_RATE_CAP_RAD_S = .16
VELOCITY_HANDOFF_DURATION_S = .04
SOURCE_PINS = {
    "expert_generation.py": "ab70935c945b4df83edf31b2cb914dff13b8c4c16c8a944fbf23025c1d639ccc",
    "contact_capture_control_v26.py": "cf6de939d9cc0d165859b7c01ee95ed992fe2c11ed52e4a62a24e04eca2cf6a7",
    "continuous_servo.py": "18716a2e0aca64425f1a2a3936bfa4e1f7886b3df7fa31f3b37284e779746c19",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def actual_opposing_pair(grip):
    """Use the original real-impulse grip predicate, with finite-value checks."""
    values = np.asarray([grip["thumb_normal_N"], grip["fingers_normal_N"],
                         grip["opposing_normals_dot"]], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite actual acquisition contact observation")
    return bool(values[0] > .05 and values[1] > .05 and values[2] < -.3)


class GraspAcquisitionGuard:
    """Instance hook; no actor pose, qpos, scene parameters or network writes."""

    def __init__(self, session, mapping, registration):
        self.session, self.mapping = session, mapping
        self.robot = session.task.robot
        if getattr(self.robot, "communication_flag", False):
            raise RuntimeError("Physical IO forbidden by acquisition guard")
        dt = float(session.task.scene.get_timestep())
        if not np.isfinite(dt) or abs(dt - PHYSICS_DT_S) > 1e-8:
            raise RuntimeError("Acquisition requires the original actual 4ms clock")
        self.dt = dt
        self.h = self.robot._hand_maps[session.side]
        if tuple(mapping.channels) != tuple(self.h.channels):
            raise RuntimeError("Original SDK6 channel order differs from acquisition mapping")
        nominal = np.asarray(session.setting["close_sdk"], dtype=float)
        parsed = np.asarray(session.spec["grasp_setting"]["close_sdk"], dtype=float)
        if nominal.shape != (6,) or not np.array_equal(nominal, parsed):
            raise RuntimeError("Acquisition nominal targets must equal resolved sample_spec")
        self.nominal_sdk = nominal.copy()
        positions, _ = mapping.state(nominal)
        self.nominal_q = np.array([positions[n] for n in mapping.channels])
        feedback_rates = session.feedback["feedback_effective_primary_rate_limit_rad_s"]
        self.rate_caps = np.array([min(TARGET_RATE_CAP_RAD_S, float(feedback_rates[n]))
                                   for n in mapping.channels])
        if not np.isfinite(self.rate_caps).all() or np.any(self.rate_caps <= 0):
            raise RuntimeError("Original folded mimic rate limits required")
        self.original_update = session.contact_update
        self.active = self.attempted = self.completed = False
        self.pair_seen = False
        self.streak = 0
        self.reanchored = set()
        self.last_update = None
        self.rows, self.events = [], []
        self.path = Path(session.out) / "grasp_acquisition_receipt.json"
        self.receipt = {
            "schema": "realman_grasp_acquisition_v1",
            "status": "registered_before_expert_motion",
            "registration": copy.deepcopy(registration),
            "actual_physics_dt_s": dt,
            "maximum_extra_physics_steps": MAX_EXTRA_STEPS,
            "maximum_nominal_extra_duration_s": 2.,
            "stable_actual_pair_steps_required": STABLE_PAIR_STEPS,
            "nominal_close_sdk6": nominal.tolist(),
            "nominal_primary_joint_targets_rad": self.nominal_q.tolist(),
            "effective_target_rate_caps_rad_s": self.rate_caps.tolist(),
            "SDK24_and_15x4ms_recording_retained": True,
            "original_controller_updates_and_LP_used": True,
            "original_acceptance_and_geometry_predicates_unchanged": True,
            "original_single_finger_latch_history_retained": True,
            "fake_contacts_or_feedback_counters_written": False,
            "physical_IO": False,
            "this_receipt_is_not_full_episode_acceptance": True,
            "rows": self.rows, "events": self.events,
        }
        self.flush()

    def _targets(self):
        return np.asarray([self.robot._hand_joints[n].get_drive_target()[0]
                           for n in self.mapping.channels], dtype=float)

    def _velocities(self):
        return np.asarray([self.robot._hand_joints[n].get_drive_velocity_target()[0]
                           for n in self.mapping.channels], dtype=float)

    def _drive_readback(self):
        names = tuple(self.mapping.limits)
        targets = np.asarray([self.robot._hand_joints[n].get_drive_target()[0]
                              for n in names], dtype=float)
        velocities = np.asarray([self.robot._hand_joints[n].get_drive_velocity_target()[0]
                                 for n in names], dtype=float)
        if not np.isfinite(targets).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Nonfinite actual hand drive readback")
        return names, targets, velocities

    def _feedback_drive_handoff_ready(self):
        if self.last_update is None:
            return False
        names, targets, velocities = self._drive_readback()
        return bool(list(names) == self.last_update["feedback_drive_joint_names"] and
                    np.array_equal(targets, self.last_update["original_feedback_drive_target_readback_rad"]) and
                    np.array_equal(velocities, self.last_update["original_feedback_drive_velocity_readback_rad_s"]))

    def _original_command_witness(self, names, readback_q, readback_velocity):
        # At completed closure the original profile is at its nominal endpoint;
        # contacted channels use the original controller's double latched_q.
        # Reconstruct that existing command only to validate its exact coupling.
        # Any profile/recapture mismatch fails the all-joint readback identity.
        primary = dict(zip(self.mapping.channels, self.nominal_q))
        feedback = self.session.feedback
        for i, name in enumerate(self.mapping.channels[:5]):
            if feedback["latched"][i]:
                primary[name] = float(feedback["latched_q"][i])
        targets = self.h.coupled(primary)
        observed_velocity = dict(zip(names, readback_velocity))
        velocities = {n: observed_velocity[n] for n in self.mapping.channels}
        for n, parent, multiplier, _ in self.mapping.mimic_order:
            velocities[n] = multiplier * velocities[parent]
        self.mapping.validate(targets, velocities)
        if not (np.array_equal(np.asarray([targets[n] for n in names], dtype=np.float32).astype(float), readback_q)
                and np.array_equal(np.asarray([velocities[n] for n in names], dtype=np.float32).astype(float), readback_velocity)):
            return None
        rates = feedback["feedback_applied_primary_target_rate_rad_s"]
        if any(abs(float(rates[n])) > self.rate_caps[i] + 1e-10
               for i, n in enumerate(self.mapping.channels)):
            raise RuntimeError("Original feedback target slew exceeds acquisition rate budget")
        return targets, velocities

    def _validate_drive_readback(self, targets, velocities, raw_q):
        # SAPIEN returns float32 drive targets. The exact nominal upper limit
        # can round one representable value above its decimal URDF bound.
        # Validate that endpoint as its original nominal value without changing
        # the drive command; every other limit and the mimic/velocity checks
        # retain ContinuousHandMapping's strict tolerance.
        checked = dict(targets)
        equivalent = []
        follower_parents = {parent for _, parent, _, _ in self.mapping.mimic_order}
        for i, name in enumerate(self.mapping.channels):
            upper = float(self.mapping.limits[name]["upper"])
            rounded_upper = float(np.float32(upper))
            if (name not in follower_parents and np.isfinite(rounded_upper)
                    and rounded_upper > upper and self.nominal_q[i] == upper
                    and float(raw_q[i]) == rounded_upper
                    and float(targets[name]) == rounded_upper):
                checked[name] = upper
                equivalent.append(name)
        self.mapping.validate(checked, velocities)
        return equivalent

    def _sdk24(self, *, measured):
        if measured:
            result = np.asarray(self.robot.get_state24(measured=True), dtype=float)
        else:
            values = []
            for side in ("left", "right"):
                values.extend(float(j.get_drive_target()[0])
                              for j in getattr(self.robot, side + "_arm_joints"))
                hand = self.robot._hand_maps[side]
                values.extend(hand.to_sdk({n: float(self.robot._hand_joints[n].get_drive_target()[0])
                                           for n in hand.limits}))
            result = np.asarray(values, dtype=float)
        if result.shape != (24,) or not np.isfinite(result).all():
            raise RuntimeError("Acquisition requires finite actual SDK24")
        return result.tolist()

    def flush(self):
        self.receipt["extra_physics_steps"] = len(self.rows)
        self.receipt["actual_extra_duration_s"] = len(self.rows) * self.dt
        self.path.write_text(json.dumps(self.receipt, indent=2, default=_json,
                                       allow_nan=False) + "\n", encoding="utf-8")
        self.session.report["grasp_acquisition_guard"] = {
            "status": self.receipt["status"], "receipt": str(self.path),
            "source_sha256": self.receipt["registration"]["installer_sha256"],
            "extra_physics_steps": len(self.rows),
            "original_physical_acceptance_retained": True,
        }

    def contact_update(self):
        if not self.active:
            return self.original_update()
        session, feedback = self.session, self.session.feedback
        actual = session.grip()
        paired = actual_opposing_pair(actual)
        measured = self.robot._measured_positions()
        previous_latches = copy.deepcopy(feedback["latched_q"])
        _, previous_drive_targets, _ = self._drive_readback()
        # Original updates real impulses, filter, latch booleans, pair counters,
        # force feedback and LP. No acquisition code writes those fields.
        result = self.original_update()
        raw_q, raw_velocity = self._targets(), self._velocities()
        drive_names, feedback_targets, feedback_velocities = self._drive_readback()
        original_witness = self._original_command_witness(drive_names, feedback_targets, feedback_velocities)
        if paired:
            self.pair_seen = True
            # A previous isolated touch can leave a stale captured position.
            # Re-capture only a currently contacting, genuinely latched channel.
            for i in (2, 3, 4):
                if i in self.reanchored or not feedback["latched"][i] or feedback["normal_N"][i] <= .025:
                    continue
                n = self.mapping.channels[i]
                value = float(measured[n])
                if not np.isfinite(value):
                    raise RuntimeError("Nonfinite actual paired latch measurement")
                feedback["latched_q"][i] = value
                raw_q[i], raw_velocity[i] = value, 0.
                self.reanchored.add(i)
                self.events.append({
                    "event": "actual_paired_measured_channel_recapture",
                    "physics_step_before": session.steps, "channel": n,
                    "channel_index": i, "original_latched_boolean": bool(feedback["latched"][i]),
                    "previous_latched_q_rad": previous_latches[i],
                    "measured_latched_q_rad": value,
                    "actual_channel_normal_N": float(feedback["normal_N"][i]),
                    "actual_grip": copy.deepcopy(actual),
                    "contacts_counters_filters_not_fabricated": True,
                })
        self.streak = self.streak + 1 if paired else 0
        if not self.pair_seen:
            # Already tighter original feedback targets are held, never opened
            # to nominal and never tightened further by the acquisition policy.
            goal = np.maximum(self.nominal_q, self.initial_q)
            # Preserve the frozen controller's passive-contact stop exactly.
            for i in (0, 1):
                if feedback["latched"][i]:
                    goal[i] = raw_q[i]
            # Acquire only by moving toward the unchanged nominal target; never
            # beyond it or by changing a historical single-finger latch.
            q = self.last_q + np.clip(goal - self.last_q,
                                     -self.rate_caps * self.dt, self.rate_caps * self.dt)
            requested_velocity = (q - self.last_q) / self.dt
        else:
            # Position slew prevents the original old latch from jumping drives.
            # Original feedback/LP values are retained, with a finite bridge to
            # their live targets and original drive-velocity targets.
            q = self.last_q + np.clip(raw_q - self.last_q,
                                     -self.rate_caps * self.dt, self.rate_caps * self.dt)
            requested_velocity = raw_velocity
        velocity_increment = self.rate_caps * self.dt / VELOCITY_HANDOFF_DURATION_S
        velocity = self.last_velocity + np.clip(requested_velocity - self.last_velocity,
                                                -velocity_increment, velocity_increment)
        primary = dict(zip(self.mapping.channels, q))
        targets = self.h.coupled(primary)
        velocities = dict(zip(self.mapping.channels, velocity))
        for n, parent, multiplier, _ in self.mapping.mimic_order:
            velocities[n] = multiplier * velocities[parent]
        quantized_upper_readbacks = self._validate_drive_readback(targets, velocities, raw_q)
        for n, value in targets.items():
            joint = self.robot._hand_joints[n]
            joint.set_drive_target(float(value))
            joint.set_drive_velocity_target(float(velocities[n]))
        self.robot._hand_sdk_target[session.side] = self.h.to_sdk(targets)
        _, applied_targets, applied_velocities = self._drive_readback()
        primary_indices = [drive_names.index(n) for n in self.mapping.channels]
        retained_original_followers = []
        if (original_witness is not None and
                np.array_equal(applied_targets[primary_indices], feedback_targets[primary_indices]) and
                np.array_equal(applied_velocities[primary_indices], feedback_velocities[primary_indices])):
            # The six physical primary commands already equal the controller's.
            # Preserve its just-written follower commands, avoiding a second
            # mimic calculation from rounded primary readbacks. The witness has
            # passed the original limit/mimic/velocity validator; the actual
            # follower transition must also fit its original hardware rate.
            original_targets, original_velocities = original_witness
            for n, _, _, _ in self.mapping.mimic_order:
                i = drive_names.index(n)
                if abs(feedback_targets[i] - previous_drive_targets[i]) > self.mapping.limits[n]["velocity"] * self.dt:
                    raise RuntimeError("Original follower handoff exceeds hardware target slew budget")
                joint = self.robot._hand_joints[n]
                joint.set_drive_target(float(original_targets[n]))
                joint.set_drive_velocity_target(float(original_velocities[n]))
                retained_original_followers.append(n)
            _, applied_targets, applied_velocities = self._drive_readback()
        self.last_update = {
            "actual_grip_before_step": copy.deepcopy(actual),
            "original_normal_N": np.asarray(feedback["normal_N"]).tolist(),
            "original_filtered_N": np.asarray(feedback["filtered_N"]).tolist(),
            "original_latched": list(feedback["latched"]),
            "original_latched_q_rad": copy.deepcopy(feedback["latched_q"]),
            "original_had_pair": bool(feedback["had_pair"]),
            "original_actual_pair_streak": int(feedback["actual_pair_streak"]),
            "real_opposing_pair_streak": self.streak,
            "original_feedback_target_rad": raw_q.tolist(),
            "applied_primary_target_rad": q.tolist(),
            "float32_nominal_upper_readbacks": quantized_upper_readbacks,
            "applied_primary_target_fd_rad_s": ((q - self.last_q) / self.dt).tolist(),
            "applied_primary_drive_velocity_rad_s": velocity.tolist(),
            "maximum_feedback_handoff_target_gap_rad": float(np.max(abs(raw_q - q))),
            "maximum_feedback_handoff_drive_velocity_gap_rad_s": float(np.max(abs(raw_velocity - velocity))),
            # Compare actual drive readbacks on both sides, including followers.
            # SAPIEN quantizes a double target to float32; a nonzero arithmetic
            # gap can represent the exact same physical drive command. No
            # tolerance or adjacent-representable-value exception is allowed.
            "feedback_drive_joint_names": list(drive_names),
            "original_feedback_drive_target_readback_rad": feedback_targets.tolist(),
            "original_feedback_drive_velocity_readback_rad_s": feedback_velocities.tolist(),
            "applied_drive_target_readback_rad": applied_targets.tolist(),
            "applied_drive_velocity_readback_rad_s": applied_velocities.tolist(),
            "feedback_drive_target_readbacks_exactly_equal": bool(np.array_equal(feedback_targets, applied_targets)),
            "feedback_drive_velocity_readbacks_exactly_equal": bool(np.array_equal(feedback_velocities, applied_velocities)),
            "original_follower_drive_commands_retained": retained_original_followers,
            "original_continuous_command_witness_validated": original_witness is not None,
        }
        self.last_q, self.last_velocity = q, velocity
        return result

    def _ready(self):
        feedback = self.session.feedback
        latest = feedback["lp_history"][-1] if feedback["lp_history"] else None
        return bool(self.last_update is not None and
                    self.streak >= STABLE_PAIR_STEPS and
                    int(feedback["actual_pair_streak"]) >= STABLE_PAIR_STEPS and
                    actual_opposing_pair(self.session.grip()) and
                    latest is not None and latest["allocation"]["usable"] and
                    int(feedback["lp_failure_steps"]) == 0 and
                    self._feedback_drive_handoff_ready())

    def before_carry(self, trigger):
        session = self.session
        if self.completed or (trigger == "plan_path" and session.dense_calls < 3):
            return
        if self.attempted:
            raise RuntimeError("Carry started without actual opposing fingertip contact")
        if getattr(self.robot, "communication_flag", False):
            raise RuntimeError("Physical IO forbidden by acquisition guard")
        if not session.feedback["active"] or session.steps < 1:
            raise RuntimeError("Acquisition requires the original completed close phase")
        if session.stage.get("name") not in ("close", "close_hold"):
            raise RuntimeError("Acquisition is allowed only before first carry/holding")
        latest = session.feedback["lp_history"][-1] if session.feedback["lp_history"] else None
        history = session.traces[-STABLE_PAIR_STEPS:]
        stable_original_history = (
            len(history) == STABLE_PAIR_STEPS and
            all(row.get("step") == session.steps - STABLE_PAIR_STEPS + 1 + i and
                "grip" in row and actual_opposing_pair(row["grip"])
                for i, row in enumerate(history)))
        if (stable_original_history and actual_opposing_pair(session.grip()) and
                int(session.feedback["actual_pair_streak"]) >= STABLE_PAIR_STEPS and
                latest is not None and latest["allocation"]["usable"] and
                int(session.feedback["lp_failure_steps"]) == 0):
            self.attempted = self.completed = True
            self.receipt.update(status="original_stable_actual_pair_no_extra_steps_needed",
                trigger=trigger, handoff_physics_step=session.steps,
                actual_handoff_grip=copy.deepcopy(session.grip()),
                original_actual_pair_streak=int(session.feedback["actual_pair_streak"]))
            self.flush()
            return
        self.attempted = True
        self.last_q, self.last_velocity = self._targets(), self._velocities()
        self.initial_q = self.last_q.copy()
        if not np.isfinite(self.last_q).all() or not np.isfinite(self.last_velocity).all():
            raise RuntimeError("Nonfinite original hand drive command")
        arm_targets = {
            side: [(float(j.get_drive_target()[0]), float(j.get_drive_velocity_target()[0]))
                   for j in getattr(self.robot, side + "_arm_joints")]
            for side in ("left", "right")
        }
        if any(abs(v) > 1e-8 for rows in arm_targets.values() for _, v in rows):
            raise RuntimeError("Acquisition requires original terminal arm hold commands")
        original_stage, original_holding = copy.deepcopy(session.stage), session.holding
        start = session.steps
        self.receipt.update(status="acquiring", trigger=trigger, start_physics_step=start,
                            start_sim_time_s=start * self.dt,
                            initial_actual_grip=copy.deepcopy(session.grip()),
                            initial_primary_targets_rad=self.initial_q.tolist(),
                            already_at_or_beyond_nominal_channels=[n for i, n in enumerate(self.mapping.channels)
                                                                  if self.initial_q[i] >= self.nominal_q[i] - 1e-12],
                            initial_original_feedback={k: copy.deepcopy(session.feedback[k])
                                                       for k in ("latched", "latched_q", "had_pair", "actual_pair_streak")},
                            original_arm_hold_commands=arm_targets)
        session.stage.update(name="close", gripper=True, terminal_hold=False)
        session.holding = False
        self.active = True
        self.flush()
        try:
            for _ in range(MAX_EXTRA_STEPS):
                before = session.steps
                telemetry_before, traces_before = len(session.telemetry), len(session.traces)
                captures_before = len(session.sample_metadata)
                measured_before = self._sdk24(measured=True)
                step_error = None
                try:
                    # No original_step or scene.step call can bypass its guards,
                    # SDK24 telemetry, geometry checks or every-15-step capture.
                    session.step()
                except BaseException as error:
                    step_error = str(error)
                    raise
                finally:
                    if session.steps == before + 1:
                        row = {"physics_step": session.steps,
                               "sim_time_before_s": before * self.dt,
                               "sim_time_after_s": session.steps * self.dt,
                               "measured_state24_before": measured_before,
                               "applied_command24": self._sdk24(measured=False),
                               "measured_state24_after": self._sdk24(measured=True),
                               "actual_grip_after_step": copy.deepcopy(session.grip()),
                               "phase": copy.deepcopy(session.stage),
                               "contact_update": copy.deepcopy(self.last_update),
                               "original_step_error": step_error}
                        self.rows.append(row)
                        if len(session.telemetry) != telemetry_before + 1 or len(session.traces) != traces_before + 1:
                            raise RuntimeError("Original Session.step telemetry was bypassed")
                        if step_error is None:
                            expected_captures = captures_before + int(session.steps % 15 == 0)
                            if len(session.sample_metadata) != expected_captures:
                                raise RuntimeError("Original every-15-step capture clock was bypassed")
                            if session.steps % 15 == 0 and session.sample_metadata[-1]["physics_step"] != session.steps:
                                raise RuntimeError("Original capture reports the wrong actual physics step")
                        current_arms = {
                            side: [(float(j.get_drive_target()[0]), float(j.get_drive_velocity_target()[0]))
                                   for j in getattr(self.robot, side + "_arm_joints")]
                            for side in ("left", "right")
                        }
                        if current_arms != arm_targets:
                            raise RuntimeError("Acquisition changed original arm hold commands")
                    elif session.steps != before:
                        raise RuntimeError("Acquisition bypassed the actual one-step clock")
                if self._ready():
                    grip = session.grip()
                    if session.reference is not None:
                        previous = np.asarray(session.reference).copy()
                        previous_baseline = session.carry_support_baseline
                        session.reference = np.asarray(grip["relative_bottle_palm"]).copy()
                        session.carry_support_baseline = session.bottle_gap()
                        self.events.append({"event": "actual_stable_pair_carry_reference_handoff",
                            "physics_step": session.steps, "previous_reference": previous.tolist(),
                            "actual_current_reference": session.reference.tolist(),
                            "previous_carry_support_baseline_m": previous_baseline,
                            "actual_current_carry_support_baseline_m": session.carry_support_baseline,
                            "actual_grip": copy.deepcopy(grip), "actor_pose_modified": False})
                    self.completed = True
                    self.receipt.update(status="actual_pair_handed_back_to_original_feedback_LP",
                        handoff_physics_step=session.steps, handoff_sim_time_s=session.steps * self.dt,
                        actual_handoff_grip=copy.deepcopy(grip),
                        original_LP_handoff=copy.deepcopy(session.feedback["lp_history"][-1]),
                        original_actual_pair_streak=int(session.feedback["actual_pair_streak"]))
                    self.events.append({"event": "original_feedback_LP_handoff",
                        "physics_step": session.steps, "no_feedback_target_jump": True,
                        "original_drive_velocity_targets_reached": True,
                        "original_pair_counters_not_modified": True})
                    return
            self.receipt["status"] = "extra_2s_budget_exhausted_original_failure_retained"
            raise RuntimeError("Carry started without actual opposing fingertip contact")
        except BaseException as error:
            if self.receipt["status"] == "acquiring":
                self.receipt["status"] = "original_guard_or_acquisition_rejected"
            self.receipt["error"] = str(error)
            raise
        finally:
            self.active = False
            session.stage.clear()
            session.stage.update(original_stage)
            session.holding = original_holding
            self.flush()


def bind_session(session, mapping, registration):
    """Bind the outermost methods, ahead of every inherited CMA construction."""
    guard = GraspAcquisitionGuard(session, mapping, registration)
    original_plan, original_dense = session.plan_path, session.dense
    original_save = session.save_evidence

    def plan_path(*args, **kwargs):
        guard.before_carry("plan_path")
        return original_plan(*args, **kwargs)

    def dense(sequence, **kwargs):
        closing = sequence.get(session.side + "_gripper") is not None
        if session.holding or (session.dense_calls >= 3 and not closing):
            guard.before_carry("dense_or_holding")
        return original_dense(sequence, **kwargs)

    def save_evidence(*args, **kwargs):
        guard.flush()
        return original_save(*args, **kwargs)

    session.contact_update = guard.contact_update
    session.plan_path, session.dense = plan_path, dense
    session.save_evidence = save_evidence
    session._rerecord_grasp_acquisition_guard = guard
    return guard


def install_grasp_acquisition_guard(source_package, legacy_directory, *, receipt_path=None):
    """Register a reviewed additive policy before constructing any expert."""
    package, legacy = Path(source_package).resolve(), Path(legacy_directory).resolve()
    source_files = {package / "expert_generation.py": SOURCE_PINS["expert_generation.py"],
                    legacy / "contact_capture_control_v26.py": SOURCE_PINS["contact_capture_control_v26.py"],
                    legacy / "continuous_servo.py": SOURCE_PINS["continuous_servo.py"]}
    if any(_sha(path) != digest for path, digest in source_files.items()):
        raise RuntimeError("Reviewed acquisition source pin changed")
    for directory in (package, legacy):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    module = importlib.import_module("expert_generation")
    servo = importlib.import_module("continuous_servo")
    if Path(module.__file__).resolve() != package / "expert_generation.py" or Path(servo.__file__).resolve() != legacy / "continuous_servo.py":
        raise RuntimeError("Wrong frozen acquisition module imported")
    base = module.ExpertSession
    if getattr(base, "_rerecord_acquisition_installed", False):
        raise RuntimeError("Acquisition installer must run once per worker")
    registration = {"schema": "grasp_acquisition_registration_v1",
        "installer": str(Path(__file__).resolve()), "installer_sha256": _sha(__file__),
        "original_source_hashes": {str(p): d for p, d in source_files.items()},
        "installed_before_expert_construction": True, "source_files_modified": False,
        "policy_scope": "first pre-carry acquisition, at most 500 original 4ms Session.step ticks",
        "SDK6_target_source": "resolved session.spec.grasp_setting.close_sdk equals session.setting.close_sdk",
        "rate_cap_rad_s": TARGET_RATE_CAP_RAD_S, "stable_pair_steps": STABLE_PAIR_STEPS,
        "original_acceptance_unchanged": True, "hardware_IO": False,
        "producer_must_include_installer_SHA_before_expert": True}
    original_init = base.__init__

    def init(session, task, sample_spec, output_dir, legacy_dir):
        if getattr(task.robot, "communication_flag", False):
            raise RuntimeError("Physical IO forbidden by acquisition guard")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "grasp_acquisition_registration.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(registration, indent=2) + "\n")
        original_init(session, task, sample_spec, output_dir, legacy_dir)
        session.sources[registration["installer"]] = registration["installer_sha256"]
        session.report["source_hashes"] = session.sources
        mapping = servo.ContinuousHandMapping(task.robot._manifest["hands"][session.side])
        bind_session(session, mapping, registration)

    base.__init__ = init
    base._rerecord_acquisition_installed = True
    if receipt_path is not None:
        with Path(receipt_path).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(registration, indent=2) + "\n")
    return registration
