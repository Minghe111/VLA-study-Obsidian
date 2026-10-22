"""Functional labware physics, SAPIEN 3.0.0b1 / PhysX CPU.

Derived hollow compound collisions and an EXPLICIT helical guide surrogate.
Imported visual meshes can be added independently; no source-asset functionality
is implied. No tablet state is written after initial placement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sapien
from transforms3d.quaternions import quat2mat

ROOT = Path(__file__).resolve().parent


def read_spec(path=None):
    return json.loads(Path(path or ROOT / "physics_spec.json").read_text())


def qaxis(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]


def rigid(actor):
    return actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)


def _disc_file(radius, thickness):
    """A convex flat cylinder; only bottoms/tops, NEVER the full hollow vessel."""
    import trimesh
    folder = ROOT / "physics_geometry"
    folder.mkdir(exist_ok=True)
    path = folder / f"disc_r{radius:.6f}_t{thickness:.6f}.obj"
    if not path.exists():
        trimesh.creation.cylinder(radius=radius, height=thickness, sections=48).export(path)
    return str(path)


def add_hollow_collisions(builder, cfg, material, *, bottom=True, z_offset=0):
    """Overlapping tangential convex boxes preserve the open central cavity."""
    ri, ro, h, n = (cfg[k] for k in ("inner_radius_m", "outer_radius_m", "height_m", "wall_segments"))
    radius = (ri + ro) / 2
    half_tangent = ro * math.tan(math.pi / n) * 1.03
    for i in range(n):
        a = 2 * math.pi * i / n
        builder.add_box_collision(
            sapien.Pose([radius * math.cos(a), radius * math.sin(a), z_offset + h / 2], qaxis([0,0,1], a)),
            half_size=[(ro-ri)/2, half_tangent, h/2], material=material, density=1100)
    if bottom:
        t = cfg["base_thickness_m"]
        builder.add_convex_collision_from_file(_disc_file(ro, t),
            pose=sapien.Pose([0,0,z_offset+t/2]), material=material, density=1100)


def tune_contacts(actor, spec):
    comp = actor.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
    for shape in comp.collision_shapes:
        shape.contact_offset = spec["physics"]["contact_offset_m"]
        shape.rest_offset = spec["physics"]["rest_offset_m"]
    if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
        comp.solver_position_iterations = spec["physics"]["solver_position_iterations"]
        comp.solver_velocity_iterations = spec["physics"]["solver_velocity_iterations"]
    return comp


def add_receiver(scene, spec=None, pose=None, visual=False, name="receiver"):
    spec = spec or read_spec()
    cfg = spec["receiver"]
    mat = sapien.physx.PhysxMaterial(.35, .30, .02)
    builder = scene.create_actor_builder()
    add_hollow_collisions(builder, cfg, mat)
    if visual:
        add_hollow_visuals(builder, cfg, [.68,.85,.91,.35])
    builder.initial_pose = pose or sapien.Pose()
    actor = builder.build_static(name)
    tune_contacts(actor, spec)
    return actor


def add_hollow_visuals(builder, cfg, color):
    """Readable collision-debug glass, not a photoreal imported visual asset."""
    ri, ro, h, n = (cfg[k] for k in ("inner_radius_m", "outer_radius_m", "height_m", "wall_segments"))
    radius = (ri + ro) / 2
    for i in range(n):
        a = 2 * math.pi * i / n
        builder.add_box_visual(sapien.Pose([radius*math.cos(a),radius*math.sin(a),h/2],qaxis([0,0,1],a)),
            half_size=[(ro-ri)/2, ro*math.tan(math.pi/n)*1.02, h/2], material=color)
    # cylinder primitive's longitudinal axis is X, rotate it into Z.
    builder.add_cylinder_visual(sapien.Pose([0,0,cfg["base_thickness_m"]/2],qaxis([0,1,0],math.pi/2)),
        radius=ro, half_length=cfg["base_thickness_m"]/2, material=color)


class LabBottle:
    """A kinematically held bottle, dynamically guided detachable cap and tablets.

    Actor origins: bottle bottom center, cap center at its top-disc midplane.
    `command_unscrew` sets an actuator target, never the cap's state. The
    simulator integrates the cap. `release_cap` checks measured axial position
    and accumulated measured rotation before removing the guide constraint.
    """
    def __init__(self, scene, spec=None, pose=None, visual=False, visuals=None, seed=0):
        self.scene, self.spec = scene, spec or read_spec()
        self.visuals = visuals or {}
        self.pose = pose or sapien.Pose([0,0,.16])
        s = self.spec
        mat = sapien.physx.PhysxMaterial(.30,.25,.02)
        b = scene.create_actor_builder()
        add_hollow_collisions(b,s["bottle"],mat)
        if self.visuals.get("body"):
            b.add_visual_from_file(**self.visuals["body"])
        elif visual:
            add_hollow_visuals(b,s["bottle"],[.67,.29,.08,.55])
        b.initial_pose = self.pose
        self.body = b.build_kinematic("functional_pill_bottle_body")
        tune_contacts(self.body,s)
        c = s["cap"]
        b = scene.create_actor_builder()
        ring = {"inner_radius_m":c["inner_radius_m"],"outer_radius_m":c["outer_radius_m"],
                "height_m":c["skirt_depth_m"],"wall_segments":32}
        add_hollow_collisions(b,ring,mat,bottom=False,z_offset=-c["skirt_depth_m"])
        b.add_convex_collision_from_file(_disc_file(c["outer_radius_m"],c["top_thickness_m"]),material=mat)
        if self.visuals.get("cap"):
            b.add_visual_from_file(**self.visuals["cap"])
        elif visual:
            b.add_cylinder_visual(sapien.Pose(q=qaxis([0,1,0],math.pi/2)),
                radius=c["outer_radius_m"],half_length=c["top_thickness_m"]/2,material=[.88,.88,.82,1])
            for i in range(32):
                a=2*math.pi*i/32
                b.add_box_visual(sapien.Pose([(c["inner_radius_m"]+c["outer_radius_m"])/2*math.cos(a),
                    (c["inner_radius_m"]+c["outer_radius_m"])/2*math.sin(a),-c["skirt_depth_m"]/2],qaxis([0,0,1],a)),
                    half_size=[(c["outer_radius_m"]-c["inner_radius_m"])/2,c["outer_radius_m"]*math.tan(math.pi/32),c["skirt_depth_m"]/2],material=[.88,.88,.82,1])
        self.closed_z=c["closed_center_z_m"]
        b.initial_pose = self.pose * sapien.Pose([0,0,self.closed_z])
        self.cap = b.build("functional_pill_bottle_cap")
        cp=tune_contacts(self.cap,s)
        cp.mass=c["mass_kg"]
        cp.linear_damping=.02
        cp.angular_damping=.02
        self.guide=scene.create_drive(self.body,sapien.Pose(),self.cap,sapien.Pose())
        self._configure_drive(self.guide)
        self.guide.set_drive_target(sapien.Pose([0,0,self.closed_z]))
        self.released=False
        self.turns_command=0.
        self.rotation_measured=0.
        self._last_angle=0.
        self.tablets=[]
        rng=np.random.default_rng(seed)
        for i in range(s["tablets"]["count"]):
            p=s["tablets"]
            b=scene.create_actor_builder()
            b.add_capsule_collision(radius=p["radius_m"],half_length=p["half_length_m"],
                material=sapien.physx.PhysxMaterial(p["friction"],p["friction"],p["restitution"]),density=1200)
            if visual:
                b.add_capsule_visual(radius=p["radius_m"],half_length=p["half_length_m"],
                    material=[.90,.85,.55,1] if i%2 else [.85,.31,.15,1])
            # Nonoverlapping initial four-layer fill. State never written again.
            a=2*math.pi*(i%3)/3
            r=.009
            local=[r*math.cos(a),r*math.sin(a),.013+.012*(i//3)]
            b.initial_pose=self.pose*sapien.Pose(local,qaxis([0,0,1],float(rng.uniform(-math.pi,math.pi))))
            actor=b.build(f"tablet_{i:02d}")
            comp=tune_contacts(actor,s)
            comp.mass=p["mass_kg"]
            # Tiny tablets must not remain asleep after a slowly moving fixture
            # loses contact with them. Disable sleeping; still integrate gravity.
            comp.sleep_threshold=0.0
            self.tablets.append(actor)
        self.cap_holder=None

    def _configure_drive(self, drive):
        t=self.spec["thread_guide"]
        for axis in ("x","y","z"):
            getattr(drive,"set_drive_property_"+axis)(t["linear_stiffness_N_per_m"],t["linear_damping_Ns_per_m"],t["linear_force_limit_N"])
        drive.set_drive_property_slerp(t["angular_stiffness_Nm_per_rad"],t["angular_damping_Nms_per_rad"],t["angular_torque_limit_Nm"])

    def set_fixture_pose(self, pose):
        """Declared kinematic holding fixture; not a robot grasp."""
        rigid(self.body).set_kinematic_target(pose)

    def command_unscrew(self, turns):
        if self.released:
            raise RuntimeError("Cap already detached")
        t=self.spec["thread_guide"]
        turns=float(np.clip(turns,0,t["opening_turns"]))
        self.turns_command=turns
        self._update_guide_target()

    def _update_guide_target(self):
        """Axial spring guide follows MEASURED twist, not requested twist.

        This is an explicitly simulated screw-guide surrogate: the actuator
        requests twist, while the virtual helical guide transmits axial force.
        Stalling the measured rotation therefore stalls the axial lift target.
        No actual cap pose/qpos is assigned.
        """
        t=self.spec["thread_guide"]
        measured_turns=float(np.clip(self.rotation_measured/(2*math.pi),0,t["opening_turns"]))
        self.guide.set_drive_target(sapien.Pose(
            [0,0,self.closed_z+t["pitch_m_per_turn"]*measured_turns],
            qaxis([0,0,1],2*math.pi*self.turns_command)))

    def after_step(self):
        """Measure actual cap rotation; no pose correction."""
        rel=self.body.pose.inv()*self.cap.pose
        R=quat2mat(rel.q)
        angle=math.atan2(R[1,0],R[0,0])
        self.rotation_measured+=(angle-self._last_angle+math.pi)%(2*math.pi)-math.pi
        self._last_angle=angle
        if not self.released:
            self._update_guide_target()

    def release_cap(self):
        t=self.spec["thread_guide"]
        measured=self.cap_state()
        if measured["lift_m"]<t["release_axial_m"] or measured["rotation_rad"]<2*math.pi*t["opening_turns"]-t["release_angle_tolerance_rad"]:
            raise RuntimeError("Measured cap has not cleared guide: "+str(measured))
        self.guide.disable()
        self.released=True
        # A separately declared gripper fixture takes the free cap at its actual pose.
        self.cap_holder=self.scene.create_drive(None,self.cap.pose,self.cap,sapien.Pose())
        self._configure_drive(self.cap_holder)
        self.cap_holder.set_drive_target(sapien.Pose())
        return {**measured,"released":True}

    def command_cap_holder(self, world_pose):
        if not self.released:
            raise RuntimeError("Cannot move the external cap holder before guide release")
        # Parent anchor stays fixed; use pose composition for the moving target.
        target=self.cap_holder.pose_in_parent.inv()*world_pose
        self.cap_holder.set_drive_target(target)

    def cap_state(self):
        rel=self.body.pose.inv()*self.cap.pose
        return {"lift_m":float(rel.p[2]-self.closed_z),"rotation_rad":float(self.rotation_measured),
            "xy_error_m":float(np.linalg.norm(rel.p[:2])),"released":self.released}

    def tablet_local_positions(self):
        inv=self.body.pose.inv()
        return np.asarray([(inv*p.pose).p for p in self.tablets])

    def tablet_states(self):
        return [{"name":p.name,"pose":np.r_[p.pose.p,p.pose.q].tolist(),
            "velocity_m_s":rigid(p).linear_velocity.tolist()} for p in self.tablets]


def create_cpu_scene(spec=None):
    spec=spec or read_spec()
    system=sapien.physx.PhysxCpuSystem()
    scene=sapien.Scene([system])
    scene.set_timestep(spec["physics"]["dt_s"])
    return scene
