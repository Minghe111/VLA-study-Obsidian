"""Dynamic contact-driven bottle/cap/pellets, with a passive screw surrogate.

No robot controller, fixture pose command, prescribed twist, cap motor or cap
holder exists here. The task must call before_step()/after_step() around each
actual PhysX step. External robot contact must supply the rotation work.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import sapien
from transforms3d.quaternions import quat2mat

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parent / "2026-10-22-labware-integration"
PROBES = ROOT / "cap_component_probes"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_collision_helpers():
    """Use original collision/tuning helpers in a private module namespace.

    Their generated disc cache is redirected into this new component's root.
    The frozen source/spec/geometry are never written.
    """
    path = FROZEN / "lab_physics.py"
    spec = importlib.util.spec_from_file_location("labware_robot_cap_collision_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = PROBES / "generated_collision_cache"
    cache.mkdir(parents=True, exist_ok=True)
    module.ROOT = cache
    return module


def rigid(entity):
    result = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if result is None:
        raise TypeError("Bottle components must be dynamic rigid bodies")
    return result


class RobotCapAssembly:
    """Two free dynamic bodies plus 12 independent capsules.

    Two separated D6 radial bearings keep the axes collinear but leave axial
    translation and spin free. A spring constrains x = pitch*theta/(2*pi),
    using measured relative displacement and measured unwrapped spin only.

    The spring's matching reaction torque is -pitch/(2*pi)*F, never an angular
    target. Equal/opposite forces and torques are applied to cap/body. Thus the
    helix can transmit contact work; it cannot turn the cap toward a requested
    angle. It is a declared surrogate, not mesh-thread or hardware calibration.
    """

    def __init__(self, scene, *, pose=None, seed=0, visual=False, spec_path=None,
                 auto_release=True, name_prefix="robot_labware", passive_params=None):
        self.scene = scene
        self.spec_path = Path(spec_path or FROZEN / "physics_spec.json")
        self.spec = copy.deepcopy(json.loads(self.spec_path.read_text()))
        self.helpers = load_collision_helpers()
        self.auto_release = bool(auto_release)
        self.dt = float(scene.get_timestep())
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("A finite positive actual physics timestep is required")
        self.name_prefix = str(name_prefix)
        self.closed_z = float(self.spec["cap"]["closed_center_z_m"])
        self.pitch = float(self.spec["thread_guide"]["pitch_m_per_turn"])
        self.helix_k = self.pitch / (2 * math.pi)
        self.open_angle = 2 * math.pi * self.spec["thread_guide"]["opening_turns"]
        thread = self.spec["thread_guide"]
        # Explicit engineering assignments, not measured cap/thread parameters.
        # mu is the frozen LabBottle material; radius is the frozen bottle wall.
        self.passive_params = {
            "friction_static": .30, "friction_dynamic": .25,
            "effective_radius_m": float(self.spec["bottle"]["outer_radius_m"]),
            "bristle_stiffness_Nm_per_rad": float(thread["angular_stiffness_Nm_per_rad"]),
            "viscous_damping_Nms_per_rad": float(thread["angular_damping_Nms_per_rad"]),
            "seat_stiffness_Nm_per_rad": float(thread["angular_stiffness_Nm_per_rad"]),
            "seat_torque_limit_Nm": float(thread["angular_torque_limit_Nm"]),
        }
        if passive_params:
            if set(passive_params) - set(self.passive_params):
                raise ValueError("Unknown passive engineering parameter")
            self.passive_params.update(passive_params)
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in self.passive_params.values()):
            raise ValueError("Passive model coefficients must be finite and positive")
        if self.passive_params["friction_dynamic"] > self.passive_params["friction_static"]:
            raise ValueError("Dynamic friction must not exceed static friction")
        self.friction_deformation_rad = 0.
        self.friction_last_theta = 0.
        self.friction_plastic_dissipation_J = 0.
        self.rotation_measured = 0.0
        self.last_wrapped_angle = 0.0
        self.released = False
        self.release_receipt = None
        self.physics_steps = 0
        self._force_is_pending = False
        self._pending_diagnostics = None
        self.trace = []
        self.pose_at_initialization = pose or sapien.Pose([0, 0, .001])
        self.frozen_fingerprints = self._frozen_inputs()
        self._build(visual, int(seed))
        self._measure()

    def _frozen_inputs(self):
        paths = [self.spec_path, FROZEN / "lab_physics.py"]
        for asset_id in ("reagent_bottle_body", "reagent_bottle_cap"):
            paths.append(FROZEN / "prepared" / asset_id / "model_data0.json")
        return {str(path): sha(path) for path in paths}

    def _build(self, visual, seed):
        h, s = self.helpers, self.spec
        material = sapien.physx.PhysxMaterial(.30, .25, .02)
        builder = self.scene.create_actor_builder()
        h.add_hollow_collisions(builder, s["bottle"], material)
        if visual:
            builder.add_visual_from_file(str(FROZEN / "prepared/reagent_bottle_body/visual/base0.glb"))
        builder.initial_pose = self.pose_at_initialization
        self.body = builder.build(self.name_prefix + "_body")
        body_component = h.tune_contacts(self.body, s)
        body_data = json.loads((FROZEN / "prepared/reagent_bottle_body/model_data0.json").read_text())
        self.body_mass_kg = float(body_data["annotation_provenance"]["mass_kg"])
        body_component.mass = self.body_mass_kg

        cap = s["cap"]
        builder = self.scene.create_actor_builder()
        ring = {"inner_radius_m": cap["inner_radius_m"], "outer_radius_m": cap["outer_radius_m"],
                "height_m": cap["skirt_depth_m"], "wall_segments": 32}
        h.add_hollow_collisions(builder, ring, material, bottom=False, z_offset=-cap["skirt_depth_m"])
        builder.add_convex_collision_from_file(h._disc_file(cap["outer_radius_m"], cap["top_thickness_m"]), material=material)
        if visual:
            builder.add_visual_from_file(str(FROZEN / "prepared/reagent_bottle_cap/visual/base0.glb"))
        builder.initial_pose = self.pose_at_initialization * sapien.Pose([0, 0, self.closed_z])
        self.cap = builder.build(self.name_prefix + "_cap")
        cap_component = h.tune_contacts(self.cap, s)
        cap_component.mass = float(cap["mass_kg"])
        cap_component.linear_damping = .02
        cap_component.angular_damping = .02

        # PhysX twist axis is joint-frame X. Rotate X into the vessel's local Z.
        joint_rotation = h.qaxis([0, 1, 0], -math.pi / 2)
        self.guides = []
        # Two radial bearings remove lateral translation and tilt geometrically.
        # SAPIEN's cone/pyramid setters do not activate angular motion limits;
        # do not rely on them, and do not replace them with an angular servo.
        for offset in (0., -float(cap["skirt_depth_m"])):
            guide = self.scene.create_drive(self.body, sapien.Pose([0, 0, self.closed_z + offset], joint_rotation),
                                              self.cap, sapien.Pose([0, 0, offset], joint_rotation))
            guide.set_limit_y(0., 0.)
            guide.set_limit_z(0., 0.)
            # Every drive motor is off; hard radial constraints are mechanical.
            for axis in ("x", "y", "z", "twist", "swing", "slerp"):
                getattr(guide, "set_drive_property_" + axis)(0., 0., 0.)
            self.guides.append(guide)
        self.guide = self.guides[0]

        self.tablets = []
        rng = np.random.default_rng(seed)
        pellets = s["tablets"]
        for i in range(int(pellets["count"])):
            builder = self.scene.create_actor_builder()
            pellet_material = sapien.physx.PhysxMaterial(pellets["friction"], pellets["friction"], pellets["restitution"])
            builder.add_capsule_collision(radius=pellets["radius_m"], half_length=pellets["half_length_m"], material=pellet_material, density=1200)
            if visual:
                builder.add_capsule_visual(radius=pellets["radius_m"], half_length=pellets["half_length_m"], material=[.9, .85, .55, 1])
            angle = 2 * math.pi * (i % 3) / 3
            local = [.009 * math.cos(angle), .009 * math.sin(angle), .013 + .012 * (i // 3)]
            builder.initial_pose = self.pose_at_initialization * sapien.Pose(local, h.qaxis([0, 0, 1], float(rng.uniform(-math.pi, math.pi))))
            actor = builder.build(self.name_prefix + f"_tablet_{i:02d}")
            component = h.tune_contacts(actor, s)
            component.mass = float(pellets["mass_kg"])
            component.sleep_threshold = 0.0
            self.tablets.append(actor)

    def _measure(self):
        relative = self.body.pose.inv() * self.cap.pose
        if self.released:
            # A detached tumbling cap no longer has a threaded spin coordinate.
            self.relative_pose = relative
            return relative
        quaternion = np.asarray(relative.q, dtype=float)
        # Swing-twist decomposition around body Z; no commanded angle is read.
        wrapped = (2 * math.atan2(float(quaternion[3]), float(quaternion[0])) + math.pi) % (2 * math.pi) - math.pi
        delta = (wrapped - self.last_wrapped_angle + math.pi) % (2 * math.pi) - math.pi
        self.rotation_measured += delta
        self.last_wrapped_angle = wrapped
        self.relative_pose = relative
        return relative

    def before_step(self):
        if self._force_is_pending:
            raise RuntimeError("before_step must be paired with exactly one scene.step and after_step")
        self._measure()
        self._force_is_pending = True
        if self.released:
            self._pending_diagnostics = {"helix_axial_force_N": 0., "helix_reaction_torque_Nm": 0.}
            return
        t = self.spec["thread_guide"]
        body_R = quat2mat(self.body.pose.q)
        axis = np.asarray(body_R[:, 2], dtype=float)
        body_rigid, cap_rigid = rigid(self.body), rigid(self.cap)
        theta = self.rotation_measured
        clamped_theta = float(np.clip(theta, 0., self.open_angle))
        effective_k = self.helix_k if 0. < theta < self.open_angle else 0.
        target_lift = self.helix_k * clamped_theta
        lift = float(self.relative_pose.p[2] - self.closed_z)
        relative_omega = float((np.asarray(cap_rigid.angular_velocity) - np.asarray(body_rigid.angular_velocity)) @ axis)
        # Anchors are collinear under the radial D6 lock. Include rotating-frame
        # velocity at the actual cap origin when deriving relative axial speed.
        r_world = np.asarray(self.cap.pose.p) - np.asarray(self.body.pose.p)
        v_relative = (np.asarray(cap_rigid.linear_velocity) - np.asarray(body_rigid.linear_velocity)
                      - np.cross(np.asarray(body_rigid.angular_velocity), r_world))
        axial_speed = float(v_relative @ axis)
        error = lift - target_lift
        error_rate = axial_speed - effective_k * relative_omega
        force_limit = float(t["linear_force_limit_N"])
        spring = float(np.clip(-float(t["linear_stiffness_N_per_m"]) * error, -force_limit, force_limit))
        force = float(np.clip(spring - float(t["linear_damping_Ns_per_m"]) * error_rate, -force_limit, force_limit))
        torque = -effective_k * force
        passive = self._passive_reactions(force, relative_omega, axis)
        total_torque = torque + passive["friction_torque_Nm"] + passive["tightening_stop_torque_Nm"]
        # The paired torque is a constraint reaction. There is no desired spin,
        # cap-rotation servo, world anchor, or externally prescribed motion.
        cap_rigid.add_force_torque(axis * force, axis * total_torque)
        body_rigid.add_force_torque(-axis * force, -axis * total_torque)
        self._pending_diagnostics = {
            "helix_target_from_measured_spin_m": target_lift,
            "helix_constraint_error_m": error, "helix_constraint_error_rate_m_s": error_rate,
            "helix_axial_force_N": force, "helix_reaction_torque_Nm": torque,
            "relative_angular_speed_rad_s": relative_omega,
            "constraint_power_W": force * axial_speed + torque * relative_omega,
            **passive,
        }

    def _relative_axis_inertia(self, axis):
        inverse = 0.
        for actor in (self.body, self.cap):
            component = rigid(actor)
            rotation = quat2mat((actor.pose * component.cmass_local_pose).q)
            local_axis = rotation.T @ axis
            inertia = float(np.sum(np.asarray(component.inertia) * local_axis ** 2))
            if not math.isfinite(inertia) or inertia <= 0:
                raise RuntimeError("Invalid measured actor inertia")
            inverse += 1. / inertia
        return 1. / inverse

    def _passive_reactions(self, axial_force, omega, axis):
        """Elastoplastic dry friction plus a unilateral finite tightening stop.

        The friction reference slips at its load-derived Coulomb bound; it is
        not a requested cap angle. The seat reaction exists only for theta<0
        and derives from a finite conservative potential, not a position servo.
        """
        p = self.passive_params
        contact_load = 0.
        names = {self.body.name, self.cap.name}
        for contact in self.scene.get_contacts():
            if {body.entity.name for body in contact.bodies} != names:
                continue
            contact_load += sum(abs(float(np.asarray(point.impulse) @ axis)) / self.dt for point in contact.points)
        normal_load = max(abs(float(axial_force)), contact_load)
        static_limit = p["friction_static"] * p["effective_radius_m"] * normal_load
        kinetic_limit = p["friction_dynamic"] * p["effective_radius_m"] * normal_load
        stiffness = p["bristle_stiffness_Nm_per_rad"]
        trial = self.friction_deformation_rad + self.rotation_measured - self.friction_last_theta
        self.friction_last_theta = self.rotation_measured
        slipping = abs(stiffness * trial) > static_limit
        limit = kinetic_limit if slipping else static_limit
        deformation = float(np.clip(trial, -limit / stiffness, limit / stiffness))
        plastic_loss = max(0., .5 * stiffness * (trial * trial - deformation * deformation))
        self.friction_plastic_dissipation_J += plastic_loss
        self.friction_deformation_rad = deformation
        elastic_torque = -stiffness * deformation
        inertia = self._relative_axis_inertia(axis)
        raw_damping = p["viscous_damping_Nms_per_rad"]
        damping = raw_damping / (1. + raw_damping * self.dt / inertia)
        friction = float(np.clip(elastic_torque - damping * omega, -limit, limit))
        damping_power = max(0., -(friction - elastic_torque) * omega)
        compression = max(-self.rotation_measured, 0.)
        seat_k, seat_limit = p["seat_stiffness_Nm_per_rad"], p["seat_torque_limit_Nm"]
        stop_torque = min(seat_k * compression, seat_limit)
        transition = seat_limit / seat_k
        stop_potential = (.5 * seat_k * compression ** 2 if compression <= transition else
                          seat_limit * compression - .5 * seat_limit ** 2 / seat_k)
        return {
            "thread_normal_load_estimate_N": normal_load,
            "measured_body_cap_axial_contact_load_N": contact_load,
            "friction_static_limit_Nm": static_limit, "friction_dynamic_limit_Nm": kinetic_limit,
            "friction_torque_Nm": friction, "friction_power_W": friction * omega,
            "friction_bristle_deformation_rad": deformation,
            "friction_elastic_potential_J": .5 * stiffness * deformation ** 2,
            "friction_plastic_loss_this_step_J": plastic_loss,
            "friction_plastic_dissipation_accumulated_J": self.friction_plastic_dissipation_J,
            "friction_damping_dissipation_W": damping_power,
            "friction_sliding": slipping,
            "tightening_stop_torque_Nm": stop_torque,
            "tightening_stop_potential_J": stop_potential,
            "tightening_stop_power_W": stop_torque * omega,
        }

    def after_step(self):
        if not self._force_is_pending:
            raise RuntimeError("Call before_step before the actual PhysX step")
        self._measure()
        self.physics_steps += 1
        self._force_is_pending = False
        if self.auto_release and not self.released:
            self.try_release()
        state = self.state()
        state.update(self._pending_diagnostics or {})
        self.trace.append(state)
        return state

    def release_eligible(self):
        t = self.spec["thread_guide"]
        return (self.rotation_measured >= self.open_angle - float(t["release_angle_tolerance_rad"])
                and float(self.relative_pose.p[2] - self.closed_z) >= float(t["release_axial_m"]))

    def try_release(self):
        """Remove only the guide after measured turn/lift clear the engagement.

        The free cap stays dynamic at its actual state. No holder is created;
        the robot must continue physically grasping it or it falls naturally.
        """
        self._measure()
        if self.released:
            return True
        if not self.release_eligible():
            return False
        measured = self.state()
        for guide in self.guides:
            guide.disable()
        self.released = True
        released_friction_energy = .5 * self.passive_params["bristle_stiffness_Nm_per_rad"] * self.friction_deformation_rad ** 2
        self.friction_plastic_dissipation_J += released_friction_energy
        self.friction_deformation_rad = 0.
        self.release_receipt = {"released": True, "measured_before_release": measured,
                                "cap_pose_or_velocity_written": False, "cap_holder_created": False,
                                "released_microcontact_energy_dissipated_J": released_friction_energy}
        return True

    def state(self):
        relative = self.body.pose.inv() * self.cap.pose
        return {"physics_step": self.physics_steps, "time_s": self.physics_steps * self.dt,
                "rotation_measured_rad": float(self.rotation_measured),
                "turns_measured": float(self.rotation_measured / (2 * math.pi)),
                "lift_measured_m": float(relative.p[2] - self.closed_z),
                "radial_error_m": float(np.linalg.norm(relative.p[:2])),
                "axis_tilt_rad": float(np.arccos(np.clip(quat2mat(relative.q)[2, 2], -1, 1))),
                "released": self.released,
                "body_pose": np.r_[self.body.pose.p, self.body.pose.q].tolist(),
                "cap_pose": np.r_[self.cap.pose.p, self.cap.pose.q].tolist()}

    def tablet_local_positions(self):
        inv = self.body.pose.inv()
        return np.asarray([(inv * actor.pose).p for actor in self.tablets])

    def make_actor_wrappers(self):
        """Optional RoboTwin named point API; does not alter poses or dynamics."""
        from envs.utils.actor_utils import Actor
        registry = json.loads((ROOT / "metadata/registry.json").read_text())
        wrapped = {}
        for key, entity in (("body", self.body), ("cap", self.cap)):
            asset = registry["assets"]["reagent_bottle_" + key]
            data = json.loads(Path(asset["model_data"]).read_text())
            wrapped[key] = Actor(entity, data, mass=float(asset["mass_kg"]))
        return wrapped

    def contract(self):
        properties = {str(i): {axis: list(getattr(guide, "get_drive_property_" + axis)())
                              for axis in ("x", "y", "z", "twist", "swing", "slerp")}
                      for i, guide in enumerate(self.guides)}
        all_motors_zero = all(tuple(v[:3]) == (0., 0., 0.) for bearing in properties.values() for v in bearing.values())
        return {
            "component": "RobotCapAssembly", "body_dynamic": not rigid(self.body).kinematic,
            "cap_dynamic": not rigid(self.cap).kinematic, "tablet_count": len(self.tablets),
            "body_mass_readback_kg": float(rigid(self.body).mass),
            "cap_mass_readback_kg": float(rigid(self.cap).mass),
            "tablet_mass_readback_kg": [float(rigid(actor).mass) for actor in self.tablets],
            "physics_dt_s_actual": self.dt, "spec_sha256": sha(self.spec_path),
            "no_rotation_target_or_motor": all_motors_zero, "drive_property_readback": properties,
            "no_world_fixture_or_cap_holder": True,
            "states_written_after_initialization": False,
            "axial_model": "force-limited spring C=measured_lift-pitch*clipped_measured_spin/(2*pi), with paired reaction torque tau=-d(target_lift)/d(theta)*F",
            "guide_radial_and_tilt_constraints": "two coaxial separated D6 radial bearings; all angular motions and axial translation free; every drive motor off",
            "guide_bearing_count": len(self.guides),
            "guide_bearing_separation_m": float(self.spec["cap"]["skirt_depth_m"]),
            "passive_friction_and_seat_parameters": self.passive_params,
            "passive_parameter_status": "explicit engineering assignments requiring calibration; frozen mu/radius reused, former actuator stiffness/damping/limit assigned to passive microcontact/seat, no claim of measured threads",
            "release": "measured positive turn plus measured lift from frozen thresholds; guide disabled without changing free cap state",
            "frozen_inputs_unchanged": self._frozen_inputs() == self.frozen_fingerprints,
            "frozen_input_fingerprints": self.frozen_fingerprints,
            "boundary": "derived thread surrogate, not calibrated thread/friction, fluid, robot grasp, or complete unscrew/pour success",
        }


def run_no_actuation_probe(seed, inverted, steps=6000, dt_s=None, batch=None):
    spec = json.loads((FROZEN / "physics_spec.json").read_text())
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(float(spec["physics"]["dt_s"] if dt_s is None else dt_s))
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.5, .5, .01], material=sapien.physx.PhysxMaterial(.5, .5, .0))
    builder.initial_pose = sapien.Pose([0, 0, -.01])
    builder.build_static("ordinary_support_table")
    if inverted:
        # Initial condition only: cap rests on the support; no pose writes later.
        pose = sapien.Pose([0, 0, .0955], [0., 1., 0., 0.])
    else:
        pose = sapien.Pose([0, 0, .0005])
    assembly = RobotCapAssembly(scene, pose=pose, seed=seed, visual=False)
    assert assembly.contract()["body_dynamic"] and assembly.contract()["cap_dynamic"]
    assert assembly.try_release() is False
    escaped_ticks = 0
    max_radius = 0.
    min_z = math.inf
    max_z = -math.inf
    for _ in range(steps):
        assembly.before_step()
        scene.step()
        assembly.after_step()
        local = assembly.tablet_local_positions()
        radius = np.linalg.norm(local[:, :2], axis=1)
        max_radius = max(max_radius, float(radius.max()))
        min_z, max_z = min(min_z, float(local[:, 2].min())), max(max_z, float(local[:, 2].max()))
        # Centre containment smoke criterion only, not a full contact-stress or
        # watertightness certification. The capsules are real independent bodies.
        contained = (radius <= spec["bottle"]["inner_radius_m"] + .001) & (local[:, 2] >= -.001) & (local[:, 2] <= assembly.closed_z + .001)
        if not bool(contained.all()):
            escaped_ticks += 1
    final = assembly.state()
    result = {"date": "2026-10-22", "seed": seed, "initially_inverted": inverted,
              "scope": "actual CPU PhysX no-actuation static/closed-container negative smoke; no robot or opening trial",
              "passed": escaped_ticks == 0 and not assembly.released,
              "physics_steps": steps, "duration_s": steps * assembly.dt,
              "external_command_count": 0, "rotation_motor_count": 0,
              "tablet_center_outside_ticks": escaped_ticks,
              "tablet_local_center_radius_max_m": max_radius,
              "tablet_local_center_z_range_m": [min_z, max_z],
              "rotation_abs_max_rad": max(abs(x["rotation_measured_rad"]) for x in assembly.trace),
              "lift_abs_max_m": max(abs(x["lift_measured_m"]) for x in assembly.trace),
              "final": final, "contract": assembly.contract(),
              "executed_component_sha256": sha(Path(__file__)),
              "robot_unscrew_or_pour_verified": False}
    folder = PROBES
    if batch is not None:
        if not batch.isidentifier():
            raise ValueError("Batch must be one simple identifier")
        folder = folder / batch
    if dt_s is not None:
        folder = folder / ("dt_" + str(dt_s).replace(".", "p"))
    folder = folder / ("inverted" if inverted else "upright") / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    with gzip.open(folder / "trace.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(assembly.trace, stream, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["upright", "inverted"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--dt", type=float, help="Actual new scene timestep; frozen physics_spec is never edited")
    parser.add_argument("--batch", help="Optional new evidence group; previous outputs are never overwritten")
    args = parser.parse_args()
    result = run_no_actuation_probe(args.seed, args.probe == "inverted", args.steps, args.dt, args.batch)
    print(json.dumps({k: result[k] for k in ("passed", "initially_inverted", "seed", "physics_steps", "tablet_center_outside_ticks", "rotation_abs_max_rad", "lift_abs_max_m")}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
