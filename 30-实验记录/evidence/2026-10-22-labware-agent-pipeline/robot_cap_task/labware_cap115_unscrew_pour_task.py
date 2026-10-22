"""Full-task world and measured final gate; the robot expert is not ready yet.

This independent entry never falls back to the body-hold expert.  Adding this
class is not a physical unscrewing/pouring result or collection acceptance.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import sapien

from labware_cap115_task import labware_closed_cap115_pick
from robot_expert import adapted_native
from robot_cap_assembly import rigid
from petri_receiver import load_dynamic_petri
from capsule_geometry import read_capsule_entities

ROOT = Path(__file__).resolve().parent
FULL_WORKFLOW_READY = False


def final_gate_condition(*, body_hold_seen, released, cap_clear, capsule_count,
                         complete_inside_count, names_unique, tablet_table_force_N,
                         tablet_bottle_force_N, max_tablet_speed_m_s,
                         receiver_speed_m_s):
    """One measured instant, not a stable-window or successful-control claim."""
    numbers = [tablet_table_force_N, tablet_bottle_force_N,
               max_tablet_speed_m_s, receiver_speed_m_s]
    if not all(math.isfinite(float(v)) and v >= 0 for v in numbers):
        raise ValueError("Invalid physical readback")
    return bool(body_hold_seen and released and cap_clear and names_unique
                and capsule_count == 12 and complete_inside_count == 12
                and tablet_table_force_N < 1e-5 and tablet_bottle_force_N < 1e-5
                and max_tablet_speed_m_s < .015 and receiver_speed_m_s < .015)


class labware_cap115_unscrew_pour(labware_closed_cap115_pick):
    """Dynamic 115-mm filled bottle and dynamic receiver, SDK24 action clock."""

    def setup_demo(self, **kwargs):
        # This position is a separate initial-layout hypothesis.  It is not the
        # conditional .16-m post-opening pour endpoint from the geometry audit.
        xy = np.asarray(kwargs.pop("receiver_center_xy", [.21, -.12]), dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Expected finite receiver XY")
        self.receiver_center_xy = xy
        self.receiver_entity = None
        self.full_observation_trace = []
        self._full_stable_ticks = 0
        self._full_gate_last_tick = None
        kwargs["task_name"] = "labware_cap115_unscrew_pour"
        result = super().setup_demo(**kwargs)
        self.config.max_episode_steps = 4000
        self.step_lim = 4000
        self.set_instruction("Hold the tablet bottle with the right hand, unscrew and remove its cap with the left hand, and pour all twelve capsules into the petri dish.")
        self.info["info"] = json.loads((ROOT / "full_agent_metadata/episode_language_bindings.json").read_text())
        self.info["next_goal_scope"] = "Actual bilateral unscrew, cap removal, twelve complete capsules into the dynamic receiver; body-hold alone is never final success."
        self.info["full_workflow_ready"] = FULL_WORKFLOW_READY
        self.info["receiver_contract"] = self.receiver_contract
        self.info["initial_layout_scope"] = self.initial_layout_receipt
        return result

    reset = setup_demo

    def load_actors(self):
        super().load_actors()
        radius = .0435
        separation = float(np.linalg.norm(self.receiver_center_xy - np.asarray(self.center_xy))) - (.025 + radius)
        if separation <= 0:
            raise RuntimeError("Initial bottle/dish bounding cylinders overlap; a pour endpoint is not a valid initial layout")
        self.receiver_entity, self.receiver_contract = load_dynamic_petri(
            self.scene, sapien.Pose([*self.receiver_center_xy, .7405]),
            visual=True, name="labware_full_dynamic_petri")
        from envs.utils.actor_utils import Actor
        self.receiver = Actor(self.receiver_entity, copy.deepcopy(self.receiver_contract["metadata"]), mass=.03)
        self.dish = self.receiver
        self.initial_layout_receipt = {
            "bottle_dish_bounding_cylinder_separation_m": separation,
            "receiver_initial_center_xy_m": self.receiver_center_xy.tolist(),
            "receiver_world_fixture": False,
            "scope": "Initial bounding-cylinder separation only; full robot/path and runtime setup validation pending",
            "full_world_runtime_verified": False}
        self._cap_local_vertices = adapted_native().fma._link_vertices(rigid(self.assembly.cap))[0]
        # The physical 48-sided cavity is slightly inside the nominal circle.
        # Read its actual cooked wall vertices once, without moving the dish.
        planes = []
        component = rigid(self.receiver_entity)
        for shape in component.collision_shapes:
            vv = np.asarray(shape.vertices)*np.asarray(shape.scale)
            transform = np.asarray(shape.local_pose.to_transformation_matrix())
            vv = vv @ transform[:3, :3].T + transform[:3, 3]
            if float(np.ptp(vv[:, 2])) < .005:
                continue  # bottom, not an inner side wall
            inner = vv[np.isclose(np.linalg.norm(vv[:, :2], axis=1), .0425, atol=1e-7), :2]
            if len(inner) < 2:
                raise RuntimeError("Unrecognized actual receiver wall geometry")
            normal = inner.mean(axis=0)
            normal = normal/np.linalg.norm(normal)
            planes.append((normal, float(np.mean(inner @ normal))))
        if len(planes) != 48:
            raise RuntimeError("Expected 48 actual inner wall planes")
        self._receiver_inner_planes = planes

    def _update_metrics(self):
        super()._update_metrics()
        if not getattr(self, "_active", False) or self.receiver_entity is None:
            return
        dt = float(self.scene.get_timestep())
        if abs(dt-.004) > 1e-8:
            raise RuntimeError("Full task requires the actual 4-ms physical clock")
        cfg, rec = self.assembly.spec["tablets"], self.receiver_contract["cavity_dimensions"]
        whole = read_capsule_entities(self.assembly.tablets, self.receiver_entity.pose,
            half_length_m=cfg["half_length_m"], capsule_radius_m=cfg["radius_m"],
            inner_radius_m=rec["inner_radius_m"], floor_z_m=rec["base_thickness_m"],
            rim_z_m=rec["height_m"], tolerance_m=self.receiver_contract["contact_offset_m"])
        tolerance = self.receiver_contract["contact_offset_m"]
        polygon_inside = 0
        for row in whole["tablets"]:
            c, axis = np.asarray(row["center_in_container_m"]), np.asarray(row["axis_in_container"])
            endpoints = c[None, :]+np.asarray([-1., 1.])[:, None]*cfg["half_length_m"]*axis
            margin = min(offset-float(np.max(endpoints[:, :2] @ normal))-cfg["radius_m"]
                         for normal, offset in self._receiver_inner_planes)
            row["actual_polygon_wall_margin_m"] = margin
            row["inside_actual_receiver"] = bool(margin >= -tolerance
                and row["floor_margin_m"] >= -tolerance and row["rim_margin_m"] >= -tolerance)
            polygon_inside += row["inside_actual_receiver"]
        whole["actual_polygon_complete_count"] = polygon_inside
        cap_in_body = np.asarray((self.assembly.body.pose.inv()*self.assembly.cap.pose).to_transformation_matrix())
        vertices = self._cap_local_vertices @ cap_in_body[:3, :3].T + cap_in_body[:3, 3]
        # A radial lower bound for the whole convex pieces, not just their
        # vertices: the nearest point of an edge may be closer than either end.
        center_xy = cap_in_body[:2, 3]
        radial_lower = float(np.linalg.norm(center_xy)
            - np.linalg.norm(vertices[:, :2]-center_xy, axis=1).max())
        cap_clear = bool(vertices[:, 2].min() > self.assembly.spec["bottle"]["height_m"]+.01
            or radial_lower > self.assembly.spec["bottle"]["outer_radius_m"]+.01)
        ids = {int(e.per_scene_id) for e in self.assembly.tablets}
        table_id = int(self.table.per_scene_id)
        bottle_ids = {int(self.assembly.body.per_scene_id), int(self.assembly.cap.per_scene_id)}
        table_force = bottle_force = 0.
        for contact in self.scene.get_contacts():
            pair = {int(c.entity.per_scene_id) for c in contact.bodies}
            if not pair & ids:
                continue
            force = sum(abs(float(np.asarray(p.impulse) @ np.asarray(p.normal)))/dt for p in contact.points)
            if table_id in pair:
                table_force += force
            if pair & bottle_ids:
                bottle_force += force
        speed = max(float(np.linalg.norm(rigid(e).linear_velocity)) for e in self.assembly.tablets)
        receiver_speed = float(np.linalg.norm(rigid(self.receiver_entity).linear_velocity))
        instantaneous = final_gate_condition(body_hold_seen=self.body_hold_stage_success(),
            released=bool(self.assembly.released), cap_clear=cap_clear,
            capsule_count=whole["count"], complete_inside_count=polygon_inside,
            names_unique=whole["names_unique"] and len(ids) == 12,
            tablet_table_force_N=table_force, tablet_bottle_force_N=bottle_force,
            max_tablet_speed_m_s=speed, receiver_speed_m_s=receiver_speed)
        tick = int(self.physics_step_count)
        if self._full_gate_last_tick != tick:
            if self._full_gate_last_tick is not None and tick != self._full_gate_last_tick+1:
                raise RuntimeError("Missing physical steps in final stable-window observer")
            self._full_stable_ticks = self._full_stable_ticks+1 if instantaneous and tick > 0 else 0
            self._full_gate_last_tick = tick
        success = self._full_stable_ticks >= math.ceil(.5/dt)
        self.metrics.update(success=success, full_unscrew_pour_success=success,
            unscrew_verified=bool(self.assembly.released and cap_clear), pour_verified=success,
            receiver_complete_capsules=polygon_inside, cap_physically_clear=cap_clear,
            cap_axial_lower_vertex_in_body_m=float(vertices[:, 2].min()),
            cap_radial_clearance_lower_bound_m=radial_lower,
            receiver_contact_tolerance_m=tolerance,
            receiver_stable_seconds=self._full_stable_ticks*dt,
            tablet_table_force_N=table_force, tablet_bottle_force_N=bottle_force,
            max_tablet_speed_m_s=speed, receiver_speed_m_s=receiver_speed,
            stage=self.current_phase)
        self.full_observation_trace.append({"physics_tick": tick,
            "receiver_pose_pq": np.r_[self.receiver_entity.pose.p, self.receiver_entity.pose.q].tolist(),
            "capsule_world_pose_pq": [np.r_[e.pose.p, e.pose.q].tolist() for e in self.assembly.tablets],
            "whole_capsule_readback": whole, "metrics": copy.deepcopy(self.metrics)})
        if tick % 15 == 0:
            self.physics_trace[-1].update(self.metrics)
            self.assembly_observation_trace[-1]["full_task_metrics"] = copy.deepcopy(self.metrics)

    def check_success(self):
        return bool(self.metrics.get("full_unscrew_pour_success", False))

    def play_once(self):
        raise RuntimeError("Full robot unscrew/remove/pour expert is not yet verified; refusing the body-hold fallback")
