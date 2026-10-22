"""Bounded cap-contact SDK proposals; no simulator or actuator state writes.

This is an unvalidated component for the future physical left-hand probe.
It does not assume a rigid attachment or use the body three-finger LP. The
caller supplies an actual ContinuousHandMapping and a geometrically screened
open/close hand configuration, then executes each returned SDK target through
the existing 60-ms environment driver.
"""
from __future__ import annotations
import copy
import numbers
import numpy as np


def read_hand_contacts(scene, actor, side):
    """Read normal impulses on the requested entity, matched by entity ID."""
    if side not in ("left", "right"):
        raise ValueError("Unknown hand side")
    entity = getattr(actor, "actor", actor)
    actor_id = entity.per_scene_id
    prefix = "l_inspire_" if side == "left" else "r_inspire_"
    names = ("little", "ring", "middle", "index", "thumb")
    forces = np.zeros(5)
    vectors = np.zeros((5, 3))
    rows = []
    dt = float(scene.get_timestep())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("A finite positive actual contact timestep is required")
    for contact in scene.get_contacts():
        ids = [b.entity.per_scene_id for b in contact.bodies]
        if actor_id not in ids:
            continue
        first = ids[0] == actor_id
        other = contact.bodies[1 if first else 0].entity
        if not other.name.startswith(prefix):
            continue
        token = next((n for n in names if n in other.name), None)
        if token is None:
            continue
        i = names.index(token)
        for point in contact.points:
            normal = np.asarray(point.normal, dtype=float)
            impulse = np.asarray(point.impulse, dtype=float)
            position = np.asarray(point.position, dtype=float)
            if (any(v.shape != (3,) or not np.isfinite(v).all()
                    for v in (normal, impulse, position))
                    or abs(float(np.linalg.norm(normal))-1.) > 1e-5):
                raise ValueError("Invalid physical contact sample")
            force = abs(float(impulse @ normal))/dt
            if not np.isfinite(force):
                raise ValueError("Nonfinite normal contact force")
            vector = normal*(1. if first else -1.)*force
            forces[i] += force
            vectors[i] += vector
            if force > 1e-6:
                rows.append({"hand_link": other.name, "channel": token,
                             "normal_force_N": force,
                             "position_world_m": position.tolist(),
                             "weighted_normal_world": vector.tolist()})
    if not np.isfinite(forces).all() or not np.isfinite(vectors).all():
        raise ValueError("Nonfinite aggregate contact force")
    denom = float(np.linalg.norm(vectors[3])*np.linalg.norm(vectors[4]))
    if not np.isfinite(denom):
        raise ValueError("Unrepresentable contact-normal magnitude")
    dot = float(np.clip(vectors[3] @ vectors[4]/denom, -1., 1.)) if denom > 1e-12 else 1.
    return {"actor_id": int(actor_id), "actor_name": entity.name,
            "forces_N": forces.tolist(), "index_thumb_normals_dot": dot,
            "opposed": bool(forces[3] > .05 and forces[4] > .05 and dot < -.3),
            "contacts": rows, "measured_physics_dt_s": dt}


class CapContactController:
    """Two independent finger contacts with explicit low-speed force feedback.

    Readiness is a diagnostic; it cannot prove that the cap is held or can be
    turned. Only a real closed-loop physical probe can establish those facts.
    """
    def __init__(self, task, mapping, *, side, actor, open_sdk, close_sdk,
                 force_targets_N=(.5, .5), rate_cap_rad_s=.08,
                 feedback_gain_rad_s_per_N=.05, pair_actuator_indices=(3, 4),
                 allow_fixed_feedback_actuator=False):
        if side not in ("left", "right"):
            raise ValueError("Unknown hand side")
        if getattr(task.robot, "communication_flag", True):
            raise RuntimeError("Simulation-only controller")
        self.task, self.mapping, self.side, self.actor = task, mapping, side, actor
        self.dt = .06
        self.expected_ticks = 15
        self._check_clock()
        if tuple(task.robot._manifest["hands"][side]["channels"]) != tuple(mapping.channels):
            raise ValueError("Mapping belongs to a different hand")
        pair = tuple(pair_actuator_indices)
        if (len(pair) != 2 or len(set(pair)) != 2
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, numbers.Integral)
                       or not 0 <= i < 6 for i in pair)):
            raise ValueError("Expected distinct index/thumb actuator indices")
        self.pair_indices = tuple(int(i) for i in pair)
        # Validate raw types as well as complete primary/mimic endpoints before
        # NumPy conversion could silently accept booleans or numeric strings.
        mapping.state(open_sdk)
        mapping.state(close_sdk)
        self.open_sdk = np.array(open_sdk, dtype=float, copy=True)
        self.close_sdk = np.array(close_sdk, dtype=float, copy=True)
        if any(x.shape != (6,) or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1000)
               for x in (self.open_sdk, self.close_sdk)):
            raise ValueError("Expected valid SDK6 configurations")
        self.open_q = mapping.lower+(1-self.open_sdk/1000)*mapping.width
        self.close_q = mapping.lower+(1-self.close_sdk/1000)*mapping.width
        self.screened_lower = np.minimum(self.open_q, self.close_q)
        self.screened_upper = np.maximum(self.open_q, self.close_q)
        self.close_sign = np.sign(self.close_q-self.open_q)
        if not isinstance(allow_fixed_feedback_actuator, (bool, np.bool_)):
            raise ValueError("Explicit boolean required for fixed feedback channel")
        self.fixed_feedback_indices = tuple(i for i in self.pair_indices if self.close_sign[i] == 0)
        if self.fixed_feedback_indices and (not allow_fixed_feedback_actuator
                                             or len(self.fixed_feedback_indices) != 1):
            raise ValueError("Each feedback actuator needs a nonzero screened closure interval")
        self._measured_q()
        # The real 60-ms clock starts from drive targets, not measured qpos.
        self.target_q = self._drive_q()
        if (np.any(self.target_q < self.screened_lower-1e-6)
                or np.any(self.target_q > self.screened_upper+1e-6)):
            raise ValueError("Actual drive must start inside the screened posture interval")
        self.force_targets = np.asarray(force_targets_N, dtype=float)
        if self.force_targets.shape != (2,) or not np.isfinite(self.force_targets).all() or np.any(self.force_targets <= 0):
            raise ValueError("Two positive force targets required")
        if not 0 < rate_cap_rad_s <= .08 or not 0 < feedback_gain_rad_s_per_N <= .1:
            raise ValueError("Unbounded or invalid controller gains")
        self.gain = float(feedback_gain_rad_s_per_N)
        self.caps = np.asarray([min(rate_cap_rad_s, mapping.limits[n]["velocity"])
                                for n in mapping.channels], dtype=float)
        roots = {n: (i, 1.) for i, n in enumerate(mapping.channels)}
        for name, parent, multiplier, _ in mapping.mimic_order:
            root, derivative = roots[parent]
            derivative *= multiplier
            roots[name] = (root, derivative)
            if abs(derivative) > 0:
                self.caps[root] = min(self.caps[root],
                    min(rate_cap_rad_s, mapping.limits[name]["velocity"])/abs(derivative))
        self.filtered = np.zeros(2)
        self.latched = np.zeros(2, dtype=bool)
        self.last_tick = None
        self.cached = None
        self.ready_streak = 0

    def _check_clock(self):
        dt = float(self.task.scene.get_timestep())
        clock = self.task.clock
        if (not np.isfinite(dt) or abs(dt-.004) > 1e-8
                or not np.isfinite(float(clock.dt)) or abs(float(clock.dt)-dt) > 1e-8
                or clock.DEFAULT_STEPS != self.expected_ticks):
            raise ValueError("Requires the actual 4-ms physics / 15-step driver")

    def _measured_q(self):
        values = self.task.robot._measured_positions()
        q = np.asarray([values[n] for n in self.mapping.channels], dtype=float)
        # Tiny numerical joint-limit excursions are not new search ranges.
        if (not np.isfinite(q).all()
                or np.any(q < self.mapping.lower-1e-6)
                or np.any(q > self.mapping.lower+self.mapping.width+1e-6)):
            raise ValueError("Invalid actual primary joint readback")
        return q

    def _drive_q(self):
        q = np.asarray([self.task.robot._hand_joints[n].get_drive_target()[0]
                        for n in self.mapping.channels], dtype=float)
        if (not np.isfinite(q).all()
                or np.any(q < self.mapping.lower-1e-6)
                or np.any(q > self.mapping.lower+self.mapping.width+1e-6)):
            raise ValueError("Invalid actual primary drive readback")
        return q

    def propose(self, physics_tick):
        self._check_clock()
        if (isinstance(physics_tick, (bool, np.bool_))
                or not isinstance(physics_tick, numbers.Integral) or physics_tick < 0
                or physics_tick % self.expected_ticks != 0
                or physics_tick != self.task.physics_step_count):
            raise ValueError("Expected the actual nonnegative 15-tick boundary")
        physics_tick = int(physics_tick)
        if self.last_tick == physics_tick:
            return copy.deepcopy(self.cached)
        if self.last_tick is not None and physics_tick-self.last_tick != self.expected_ticks:
            raise RuntimeError("Call once after each complete 15-tick control window")
        measured = self._measured_q()
        drive = self._drive_q()
        if np.any(np.abs(drive-self.target_q) > 1e-6):
            raise RuntimeError("Previous proposal was not executed or the hand drive changed")
        evidence = read_hand_contacts(self.task.scene, self.actor, self.side)
        force = np.asarray(evidence["forces_N"])[[3, 4]]
        filtered = self.filtered+self.dt/(.04+self.dt)*(force-self.filtered)
        latched = self.latched.copy()
        previous_q = self.target_q.copy()
        # Non-pinching channels only move toward the screened close posture.
        # No free search beyond the explicitly supplied geometric target.
        target_q = previous_q+np.clip(self.close_q-previous_q,
                                     -self.caps*self.dt, self.caps*self.dt)
        for j, i in enumerate(self.pair_indices):
            if i in self.fixed_feedback_indices:
                # An ordered one-channel stage observes both contacts but never
                # uses the fixed channel's force to alter its actual drive.
                continue
            if force[j] > .025 and not latched[j]:
                latched[j] = True
                # Clamp a newly detected contact from the actual measured joint,
                # with the same per-command slew bound as all other proposals.
                target_q[i] = previous_q[i]+np.clip(
                    measured[i]-previous_q[i],
                    -self.caps[i]*self.dt, self.caps[i]*self.dt)
            elif latched[j]:
                rate = np.clip(self.close_sign[i]*self.gain*(self.force_targets[j]-filtered[j]),
                               -self.caps[i], self.caps[i])
                target_q[i] = previous_q[i]+rate*self.dt
        target_q = np.clip(target_q, self.screened_lower, self.screened_upper)
        for i in self.fixed_feedback_indices:
            # The actual float drive can differ slightly from the canonical
            # single-point SDK interval. Do not silently move it by clipping.
            target_q[i] = previous_q[i]
        if np.any(np.abs(target_q-previous_q) > self.caps*self.dt+1e-12):
            raise RuntimeError("Controller exceeded primary/mimic slew bound")
        sdk = 1000*(1-(target_q-self.mapping.lower)/self.mapping.width)
        if not np.isfinite(sdk).all() or np.any(sdk < -1e-7) or np.any(sdk > 1000+1e-7):
            raise RuntimeError("Controller produced invalid SDK6")
        # Validate follower positions and velocities, including nested/negative mimics.
        sdk_rate = -1000*(target_q-previous_q)/(self.mapping.width*self.dt)
        self.mapping.state(sdk, sdk_rate)
        ready = bool(evidence["opposed"] and np.all(filtered >= .8*self.force_targets))
        self.target_q, self.filtered, self.latched = target_q, filtered, latched
        self.ready_streak = self.ready_streak+1 if ready else 0
        self.last_tick = physics_tick
        self.cached = {"sdk6": sdk.tolist(), "contact": evidence,
                       "filtered_index_thumb_force_N": self.filtered.tolist(),
                       "target_index_thumb_force_N": self.force_targets.tolist(),
                       "primary_rate_caps_rad_s": self.caps.tolist(),
                       "latched_index_thumb": self.latched.tolist(),
                       "pair_actuator_indices": list(self.pair_indices),
                       "fixed_feedback_actuator_indices": list(self.fixed_feedback_indices),
                       "screened_posture_range_only": True,
                       "ready_boundaries": self.ready_streak,
                       "ready": self.ready_streak >= 3,
                       "grasp_or_turn_success_inferred": False,
                       "state_or_drive_writes": False}
        return copy.deepcopy(self.cached)
