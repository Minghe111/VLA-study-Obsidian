"""CPU-only geometric screening of the real RM65-B / Inspire cap pinch.

The clone has the vendor collisions, all 36 articulated joints, and the SDK
coupling. set_qpos is used only for static FK in this separate geometric clone;
this is not a robot rollout and never changes the working task or frozen asset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sapien
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "cap_geometry_probes"
VENDOR = Path("/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4")
NATIVE = Path("/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin")
FROZEN = ROOT.parent / "2026-10-22-labware-integration"
TEACHER = NATIVE / "integrations/realman_rm65b_gen4/teacher_legacy"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def matrix(pose):
    return np.asarray(pose.to_transformation_matrix(), dtype=float)


def pose(m):
    q = Rotation.from_matrix(m[:3, :3]).as_quat()
    return sapien.Pose(m[:3, 3], [q[3], *q[:3]])


def frame(axis, vertical):
    x = np.asarray(axis, dtype=float); x /= np.linalg.norm(x)
    z = np.asarray(vertical, dtype=float); z -= x * (x @ z); z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=1)


class Geometry:
    def __init__(self, output=OUTPUT, snapshot=None, height_extra=0., geometry_spec_path=None):
        from mplib import ArticulatedModel, Pose
        from mplib.collision_detection import fcl
        from mplib.sapien_utils.conversion import SapienPlanningWorld
        from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.height_extra=float(height_extra)
        self.geometry_spec_path=Path(geometry_spec_path or FROZEN/'physics_spec.json')
        if not 0<=self.height_extra<=.10:raise ValueError('Geometry-only height hypothesis must be explicit, 0..100mm')
        self.fcl, self.Pose = fcl, Pose
        self.fma = load("cap_geometry_original_fma", TEACHER / "full_mesh_path_audit_bounded.py")
        self.mapping = load("cap_geometry_continuous_sdk", TEACHER / "continuous_servo.py")
        physics = load("cap_geometry_native_filter", NATIVE / "envs/robot/realman_rm65b/physics_setup.py")
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        self.scene.set_timestep(.004)
        # Only render geometry is removed. Collisions, joints, mass, inertias and
        # joint limits stay byte-for-byte equivalent in their XML subtrees.
        source = ROOT / "task_probes/bottle_probe_v6/robot_head_down.urdf"
        tree = ET.parse(source)
        removed = 0
        for link in tree.getroot().findall("link"):
            for visual in list(link.findall("visual")):
                link.remove(visual); removed += 1
        self.derived = self.output / "collision_only_robot.urdf"
        tree.write(self.derived, encoding="utf-8", xml_declaration=True)
        loader = self.scene.create_urdf_loader(); loader.fix_root_link = True
        self.art = loader.parse(str(self.derived))[0][0].build()
        self.art.root_pose = sapien.Pose([0, -.65, .09539821919202804], [0, 0, 0, 1])
        self.links = {l.name: l for l in self.art.links}
        self.names = [j.name for j in self.art.active_joints]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.physics = physics.configure_collisions(self.art, VENDOR)
        manifest = json.loads((VENDOR / "source_manifest.json").read_text())
        self.maps = {s: self.mapping.ContinuousHandMapping(manifest["hands"][s]) for s in ("left", "right")}
        self.transitions_path = ROOT / "task_probes/bottle_probe_v6/transitions.json"
        self.trace_path = ROOT / "task_probes/bottle_probe_v6/physics_trace.json"
        transitions = json.loads(self.transitions_path.read_text())
        # Use a measured, airborne, opposing-contact hold, not commanded grasp.
        holds = [(i, x) for i, x in enumerate(transitions) if x["metrics"]["opposed_contact"]
                 and x["metrics"]["bottom_gap_m"] >= .025 and x["metrics"]["hold_seconds"] >= .5]
        self.reference_row, self.reference = min(holds, key=lambda pair: pair[1]["metrics"]["speed_m_s"])
        trace = json.loads(self.trace_path.read_text())
        self.bottle_matrix = np.asarray(next(x["pose"] for x in trace if x["tick"] == self.reference["tick_end"]), dtype=float)
        self.q0 = self.from_sdk24(self.reference["next_state"])
        self.snapshot = None; self.actual_cap_matrix = None
        if snapshot is not None:
            self.snapshot=json.loads(Path(snapshot).read_text())
            self.q0=self.from_sdk24(self.snapshot['state24'])
            if 'body_matrix'in self.snapshot:
                self.bottle_matrix=np.asarray(self.snapshot['body_matrix'],dtype=float);self.actual_cap_matrix=np.asarray(self.snapshot['cap_matrix'],dtype=float)
            else:
                state=self.snapshot['assembly_observation']['cap_state']
                self.bottle_matrix=matrix(sapien.Pose(state['body_pose'][:3],state['body_pose'][3:]))
                self.actual_cap_matrix=matrix(sapien.Pose(state['cap_pose'][:3],state['cap_pose'][3:]))
            if 'q36'in self.snapshot:
                positions=dict(zip(self.snapshot['joint_names'],self.snapshot['q36']))
                if set(positions)!=set(self.names): raise ValueError('Actual q36 name mismatch')
                self.q0=np.asarray([positions[n]for n in self.names])
            if 'actual_joint_positions' in self.snapshot:
                positions=self.snapshot['actual_joint_positions']
                if set(positions)!=set(self.names): raise ValueError('Actual snapshot q36 name mismatch')
                self.q0=np.asarray([positions[n]for n in self.names],dtype=float)
            if 'root_pose_matrix' in self.snapshot:self.art.root_pose=pose(np.asarray(self.snapshot['root_pose_matrix']))
            if 'root_matrix' in self.snapshot:self.art.root_pose=pose(np.asarray(self.snapshot['root_matrix']))
        self.art.set_qpos(self.q0)
        self.allowed = set()
        for group in self.physics["fixed_components"]:
            self.allowed.update(frozenset(p) for p in itertools.combinations(group, 2))
        for ex in self.physics["exemptions"]:
            self.allowed.update(frozenset((a, b)) for a in ex["components"][0] for b in ex["components"][1] if a != b)
        for link in self.art.links:
            if link.parent is not None and link.collision_shapes and link.parent.collision_shapes:
                self.allowed.add(frozenset((link.name, link.parent.name)))
        urdf = export_kinematic_chain_urdf(self.art)
        srdf = ET.Element("robot", name=ET.fromstring(urdf).get("name"))
        for pair in sorted(sorted(p) for p in self.allowed):
            ET.SubElement(srdf, "disable_collisions", link1=pair[0], link2=pair[1], reason="Existing native physical exclusion")
        self.convert = lambda comp: self.fma._convert_component_scaled(comp, SapienPlanningWorld.convert_physx_component, fcl, Pose)
        colliders = [c for l in self.art.links if (c := self.convert(l)) is not None]
        self.model = ArticulatedModel.create_from_urdf_string(urdf, ET.tostring(srdf, encoding="unicode"), colliders,
            link_names=list(self.links), joint_names=self.names, verbose=False)
        self.model.set_base_pose(Pose(self.art.root_pose)); self.model.set_qpos(self.q0, full=True)
        self.pk = self.model.get_pinocchio_model()
        assert list(self.pk.get_joint_names()) == self.names
        self.objects = {o.name: o for o in self.model.get_fcl_model().get_collision_objects()}
        self.vertices = {n: self.fma._link_vertices(self.links[n])[0] for n in self.objects}
        self.left_names = [n for n in self.objects if n.startswith("l_inspire_")]
        self.right_names = [n for n in self.objects if n.startswith("r_inspire_")]
        self.thumb_names = [n for n in self.left_names if "thumb" in n]
        self.index_names = [n for n in self.left_names if "index" in n]
        self.cr = fcl.CollisionRequest(); self.dr = fcl.DistanceRequest()
        self._build_labware()
        fk_checks = []
        self.pk.compute_forward_kinematics(self.q0)
        base = matrix(self.art.root_pose)
        for i, n in enumerate(self.pk.get_link_names()):
            a = base @ matrix(self.pk.get_link_pose(i)); b = matrix(self.links[n].entity.pose)
            fk_checks.append(float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
        self.receipt = {
            "kind": "CPU static geometric clone, no physics rollout", "renderer_created": False,
            "physics_steps": 0, "visual_elements_removed_only": removed,
            "active_joints": len(self.names), "collision_links": len(self.objects),
            "reference_episode": str(self.transitions_path), "reference_row": self.reference_row,
            "reference_tick": self.reference["tick_end"], "reference_metrics": self.reference["metrics"],
            "reference_bottle_matrix": self.bottle_matrix.tolist(),
            "primary_arm_hand_state_source": "measured next_state SDK24 from successful empty-bottle rollout",
            "mimic_state_boundary": "mimic positions reconstructed using vendor coupling; individual mimic deflection was not recorded",
            "cap_added_geometry_only": "same frozen cap at nominal closed relative pose; empty-bottle trace did not contain cap or pills",
            "max_fk_position_error_m": max(fk_checks),
            "source_sha256": {str(p): sha(p) for p in (source, VENDOR / "source_manifest.json", self.transitions_path, self.trace_path,
                ROOT / "robot_cap_assembly.py", FROZEN / "physics_spec.json", TEACHER / "full_mesh_path_audit_bounded.py")},
            "spec": {"bottle_height_m": self.spec["bottle"]["height_m"], "bottle_radius_m": self.spec["bottle"]["outer_radius_m"],
                "cap_outer_radius_m": self.spec["cap"]["outer_radius_m"], "cap_skirt_depth_m": self.spec["cap"]["skirt_depth_m"],
                "cap_top_thickness_m": self.spec["cap"]["top_thickness_m"], "opening_turns": self.spec["thread_guide"]["opening_turns"]},
        }
        if self.height_extra:
            self.receipt['geometry_only_variant']={'bottle_height_extra_m':self.height_extra,'cap_and_thread_axial_offset_m':self.height_extra,
                'changes':'hypothetical wall length and closed cap height only; source geometry is unchanged','mass_inertia_or_physics_validated':False,
                'body_height_m':self.spec['bottle']['height_m']}
        if snapshot is not None:
            self.receipt['reference_snapshot']=str(snapshot);self.receipt['reference_snapshot_sha256']=sha(snapshot)
            self.receipt['reference_episode']=self.snapshot.get('source',str(snapshot))
            self.receipt['reference_tick']=self.snapshot.get('physics_tick',self.snapshot.get('tick'))
            self.receipt['reference_metrics']=self.snapshot.get('metrics',{'snapshot_scope':self.snapshot.get('scope')})
            self.receipt['primary_arm_hand_state_source']='measured snapshot state24; actual named q36 takes precedence when present'
            self.receipt['reference_bottle_matrix']=self.bottle_matrix.tolist()
            self.receipt['cap_added_geometry_only']='cap pose taken directly from the actual snapshot'
            if 'actual_joint_positions' in self.snapshot or 'q36'in self.snapshot:self.receipt['mimic_state_boundary']='all 36 actual joint positions measured in deterministic preclose replay; later proposed closes use vendor coupling'
            if 'right_palm_matrix'in self.snapshot:
                expected=np.asarray(self.snapshot['right_palm_matrix']);actual=self.world_matrix('r_inspire_base_link')
                self.receipt['actual_snapshot_palm_position_error_m']=float(np.linalg.norm(actual[:3,3]-expected[:3,3]))
                self.receipt['actual_snapshot_palm_rotation_error_rad']=float(Rotation.from_matrix(actual[:3,:3]@expected[:3,:3].T).magnitude())
                if self.receipt['actual_snapshot_palm_position_error_m']>2e-5:raise RuntimeError('Actual replay FK clone differs')
        self.receipt['actual_collision_spec_path']=str(self.geometry_spec_path);self.receipt['actual_collision_spec_sha256']=sha(self.geometry_spec_path)

    def from_sdk24(self, state):
        q = np.zeros(len(self.names)); state = np.asarray(state)
        for side, prefix, offset in (("left", "l", 0), ("right", "r", 12)):
            for i in range(6): q[self.index[f"{prefix}_joint{i+1}"]] = state[offset+i]
            sdk = np.clip(state[offset+6:offset+12], 0, 1000)
            for n, v in self.maps[side].state(sdk)[0].items(): q[self.index[n]] = v
        return q

    def sdk24(self, q):
        out = []
        for side, prefix in (("left", "l"), ("right", "r")):
            out.extend(float(q[self.index[f"{prefix}_joint{i+1}"]]) for i in range(6))
            mp = self.maps[side]
            out.extend(float(1000*(1-(q[self.index[n]]-mp.limits[n]["lower"])/(mp.limits[n]["upper"]-mp.limits[n]["lower"]))) for n in mp.channels)
        return out

    def hand_q(self, sdk):
        q = self.q0.copy()
        for n, v in self.maps["left"].state(sdk)[0].items(): q[self.index[n]] = v
        return q

    def _build_labware(self):
        h = load("cap_geometry_frozen_collision_helpers", FROZEN / "lab_physics.py")
        h.ROOT = self.output / "collision_cache"; h.ROOT.mkdir(exist_ok=True)
        self.spec = json.loads(self.geometry_spec_path.read_text())
        self.spec['bottle']['height_m']+=self.height_extra;self.spec['cap']['closed_center_z_m']+=self.height_extra
        mat = sapien.physx.PhysxMaterial(.3, .25, .02)
        b = self.scene.create_actor_builder(); h.add_hollow_collisions(b, self.spec["bottle"], mat)
        b.initial_pose = pose(self.bottle_matrix); self.body = b.build_static("geometry_bottle")
        cap = self.spec["cap"]; b = self.scene.create_actor_builder()
        h.add_hollow_collisions(b, {"inner_radius_m": cap["inner_radius_m"], "outer_radius_m": cap["outer_radius_m"],
            "height_m": cap["skirt_depth_m"], "wall_segments": 32}, mat, bottom=False, z_offset=-cap["skirt_depth_m"])
        b.add_convex_collision_from_file(h._disc_file(cap["outer_radius_m"], cap["top_thickness_m"]), material=mat)
        self.cap_matrix = self.bottle_matrix.copy(); self.cap_matrix[:3, 3] += self.bottle_matrix[:3, :3] @ [0, 0, cap["closed_center_z_m"]]
        if self.actual_cap_matrix is not None:
            self.cap_matrix=self.actual_cap_matrix.copy();self.cap_matrix[:3,3]+=self.bottle_matrix[:3,2]*self.height_extra
        b.initial_pose = pose(self.cap_matrix); self.cap = b.build_static("geometry_cap")
        rb = sapien.physx.PhysxRigidBaseComponent
        self.body_obj = self.convert(self.body.find_component_by_type(rb)); self.cap_obj = self.convert(self.cap.find_component_by_type(rb))
        b = self.scene.create_actor_builder(); b.add_box_collision(half_size=[1., .35, .02]); b.initial_pose=sapien.Pose([0, 0, .72])
        self.table = b.build_static("geometry_table"); self.table_obj = self.convert(self.table.find_component_by_type(rb))

    def shift_held_reference(self, xyz):
        """Geometric hypothesis for a future right-arm reposition, no motion.

        Preserve the measured bottle-to-right-palm rigid transform. This tests
        endpoint IK/collisions only and cannot certify the move or continued
        grasp; both require an actual subsequent rollout.
        """
        xyz = np.asarray(xyz, dtype=float)
        if not np.any(xyz): return
        self.pk.compute_forward_kinematics(self.q0)
        target = self.world_matrix("r_inspire_base_link"); target[:3, 3] += xyz
        mask = np.ones(len(self.names), dtype=bool)
        mask[[self.index[f"r_joint{i}"] for i in range(1,7)]] = False
        link = list(self.pk.get_link_names()).index("r_inspire_base_link")
        root_inv = np.linalg.inv(matrix(self.art.root_pose))
        q, ok, err = self.pk.compute_IK_CLIK(link, self.Pose(root_inv@target), self.q0, mask.tolist(), eps=1e-6, max_iter=2000, dt=.1, damp=1e-6)
        q = np.asarray(q); limits=np.asarray(self.art.get_qlimits())
        legal=bool(np.all(q>=limits[:,0]-1e-6) and np.all(q<=limits[:,1]+1e-6))
        if not ok or not legal: raise RuntimeError(f"Right reposition endpoint IK failed: {ok}, {legal}, {err}")
        self.q0=q; self.art.set_qpos(q); self.model.set_qpos(q,full=True)
        self.bottle_matrix[:3,3]+=xyz; self.cap_matrix[:3,3]+=xyz
        self.body.pose=pose(self.bottle_matrix);self.cap.pose=pose(self.cap_matrix)
        rb=sapien.physx.PhysxRigidBaseComponent
        self.body_obj=self.convert(self.body.find_component_by_type(rb));self.cap_obj=self.convert(self.cap.find_component_by_type(rb))
        self.receipt["hypothetical_held_reposition_m"] = xyz.tolist()
        self.receipt["hypothetical_right_sdk24"] = self.sdk24(q)
        self.receipt["hypothetical_right_reposition_IK_only"] = True

    def distance(self, a, b):
        collision = bool(self.fcl.collide(a, b, self.cr).is_collision())
        gap = float(self.fcl.distance(a, b, self.dr).min_distance)
        if not math.isfinite(gap): raise RuntimeError("Non-finite mesh distance")
        return max(0., gap) if not collision else 0., collision

    def world_matrix(self, name):
        i = list(self.pk.get_link_names()).index(name)
        return matrix(self.art.root_pose) @ matrix(self.pk.get_link_pose(i))

    def screen(self, q):
        self.model.set_qpos(q, full=True)
        left_arm_hand = [n for n in self.objects if n.startswith("l_")]
        collisions = []; min_gaps = {"left_body": 10., "left_right": 10., "left_table": 10., "left_cap_noncontact": 10.}
        # The cap may contact thumb/index; body, table, opposite hand and all
        # other cap contacts remain hard collision failures. No guessed contact
        # exemptions are introduced for the right hand/body reference contacts.
        tips = set(self.thumb_names + self.index_names)
        for n in left_arm_hand:
            a = self.objects[n]
            for cat, obj in (("left_body", self.body_obj), ("left_table", self.table_obj), ("left_cap_noncontact", self.cap_obj)):
                d, c = self.distance(a, obj)
                if cat == "left_cap_noncontact" and n in tips: continue
                min_gaps[cat] = min(min_gaps[cat], d)
                if c: collisions.append([n, cat])
            for m, b in self.objects.items():
                if n == m or frozenset((n, m)) in self.allowed: continue
                if m.startswith("l_") and m < n: continue
                d, c = self.distance(a, b)
                if m.startswith("r_"): min_gaps["left_right"] = min(min_gaps["left_right"], d)
                if c: collisions.append([n, m])
        contact = {}
        for group, names in (("thumb", self.thumb_names), ("index", self.index_names)):
            vals = [(self.distance(self.objects[n], self.cap_obj)[0], n) for n in names]
            d, n = min(vals)
            contact[group] = {"gap_m": d, "link": n, "collision": self.distance(self.objects[n], self.cap_obj)[1]}
        return {"collision_free_except_intended_cap_contact": not collisions, "collisions": collisions,
            "minimum_gaps_m": min_gaps, "cap_nearest": contact}

    def cap_contact_depths(self):
        request=self.fcl.CollisionRequest(num_max_contacts=100,enable_contact=True)
        groups={};maximum=0.
        for label,names in (("thumb",self.thumb_names),("index",self.index_names)):
            contacts=[]
            for name in names:
                result=self.fcl.collide(self.objects[name],self.cap_obj,request)
                for c in result.get_contacts():
                    d=float(c.penetration_depth);maximum=max(maximum,d)
                    local=(np.linalg.inv(self.cap_matrix)@np.r_[np.asarray(c.pos),1.])[:3]
                    contacts.append({"link":name,"depth_m":d,"world_normal":np.asarray(c.normal).tolist(),"cap_local_position_m":local.tolist()})
            groups[label]=contacts
        dot=1.
        for a in groups['thumb']:
            for b in groups['index']:
                dot=min(dot,float(np.dot(a['world_normal'],b['world_normal'])))
        return {"groups":groups,"max_depth_m":maximum,"most_opposed_normal_dot":dot}

    def right_clearance_scan(self):
        self.pk.compute_forward_kinematics(self.q0)
        initial=self.world_matrix('r_inspire_base_link')
        mask=np.ones(len(self.names),dtype=bool);mask[[self.index[f'r_joint{i}'] for i in range(1,7)]]=False
        link=list(self.pk.get_link_names()).index('r_inspire_base_link')
        inv=np.linalg.inv(matrix(self.art.root_pose));records=[]
        request=self.fcl.CollisionRequest(num_max_contacts=200,enable_contact=True)
        for delta in (0.,-.005,-.010,-.015,-.020,-.025,-.030,-.040):
            target=initial.copy();target[:3,3]+=self.bottle_matrix[:3,2]*delta
            q,ok,err=self.pk.compute_IK_CLIK(link,self.Pose(inv@target),self.q0,mask.tolist(),eps=1e-6,max_iter=2000,dt=.1,damp=1e-6)
            q=np.asarray(q);limits=np.asarray(self.art.get_qlimits());legal=bool(np.all(q>=limits[:,0]-1e-6)and np.all(q<=limits[:,1]+1e-6))
            record={'palm_translation_along_bottle_Z_m':delta,'ik_in_limits':bool(ok and legal)}
            if ok and legal:
                self.model.set_qpos(q,full=True);record['sdk24_static_configuration']=self.sdk24(q)
                cap=[];body=[];table=[]
                for name in self.right_names:
                    for label,obj,out in (('cap',self.cap_obj,cap),('body',self.body_obj,body),('table',self.table_obj,table)):
                        d,c=self.distance(self.objects[name],obj)
                        if c or d<=.003:
                            contacts=self.fcl.collide(self.objects[name],obj,request).get_contacts()
                            out.append({'link':name,'gap_m':d,'collided':c,'maximum_depth_m':max([float(z.penetration_depth)for z in contacts],default=0.),
                                'contacts':[{'normal':np.asarray(z.normal).tolist(),'world_position':np.asarray(z.pos).tolist(),'depth_m':float(z.penetration_depth)}for z in contacts]})
                record.update(cap_contact_or_under_3mm=cap,body_contact_or_under_3mm=body,table_contact_or_under_3mm=table)
            records.append(record)
        return records

    def right_closing_sweep(self, prebend=1000.):
        self.pk.compute_forward_kinematics(self.q0)
        initial=self.world_matrix('r_inspire_base_link')
        mask=np.ones(len(self.names),dtype=bool);mask[[self.index[f'r_joint{i}']for i in range(1,7)]]=False
        link=list(self.pk.get_link_names()).index('r_inspire_base_link');inv=np.linalg.inv(matrix(self.art.root_pose))
        open_sdk=np.array([0.,0.,prebend,prebend,prebend,0.]);close_sdk=np.array([0.,0.,608.,610.,610.,0.])
        records=[]
        for delta in (0.,-.005,-.010,-.015,-.020,-.025,-.030,-.035,-.040):
            target=initial.copy();target[:3,3]+=self.bottle_matrix[:3,2]*delta
            q,ok,err=self.pk.compute_IK_CLIK(link,self.Pose(inv@target),self.q0,mask.tolist(),eps=1e-6,max_iter=2000,dt=.1,damp=1e-6)
            q=np.asarray(q);limits=np.asarray(self.art.get_qlimits());legal=bool(np.all(q>=limits[:,0]-1e-6)and np.all(q<=limits[:,1]+1e-6))
            record={'palm_translation_along_bottle_Z_m':delta,'ik_in_limits':bool(ok and legal),'close_samples':[],'palm_target_matrix':target.tolist(),'prebend_sdk6':open_sdk.tolist()}
            if ok and legal:
                for fraction in np.linspace(0,1,21):
                    sdk=open_sdk+(close_sdk-open_sdk)*fraction;sample_q=q.copy()
                    for n,v in self.maps['right'].state(sdk)[0].items():sample_q[self.index[n]]=v
                    self.model.set_qpos(sample_q,full=True)
                    categories={};minimum={}
                    for cat,obj in (('cap',self.cap_obj),('body',self.body_obj),('table',self.table_obj)):
                        rows=[(n,*self.distance(self.objects[n],obj))for n in self.right_names]
                        categories[cat]=[{'link':n,'gap_m':d,'collision':c}for n,d,c in rows if c or d<.001]
                        minimum[cat]=min(d for n,d,c in rows)
                    record['close_samples'].append({'fraction':float(fraction),'sdk6':sdk.tolist(),'near_or_contact':categories,'minimum_gaps_m':minimum,
                        'body_thumb_near':any('thumb'in z['link']for z in categories['body']),
                        'body_other_finger_near':any(('index'in z['link']or'middle'in z['link'])for z in categories['body']),
                        'sdk24_static_configuration':self.sdk24(sample_q)})
            records.append(record)
        return records

    def right_rotated_family(self,prebend=800.):
        self.pk.compute_forward_kinematics(self.q0);initial=self.world_matrix('r_inspire_base_link')
        mask=np.ones(len(self.names),dtype=bool);arm_idx=[self.index[f'r_joint{i}']for i in range(1,7)];mask[arm_idx]=False
        link=list(self.pk.get_link_names()).index('r_inspire_base_link');inv=np.linalg.inv(matrix(self.art.root_pose))
        axis=initial[:3,2];pivot=self.bottle_matrix[:3,3]+self.bottle_matrix[:3,2]*.086
        opened=np.array([0.,0.,prebend,prebend,prebend,0.]);closed=np.array([0.,0.,608.,610.,610.,0.]);records=[]
        for roll,delta in itertools.product((-60.,-45.,-30.,-15.,0.,15.),(-.010,-.015,-.020,-.025,-.030,-.035)):
            rot=Rotation.from_rotvec(axis*math.radians(roll)).as_matrix();target=initial.copy()
            target[:3,:3]=rot@initial[:3,:3];target[:3,3]=pivot+rot@(initial[:3,3]-pivot)+self.bottle_matrix[:3,2]*delta
            q,ok,err=self.pk.compute_IK_CLIK(link,self.Pose(inv@target),self.q0,mask.tolist(),eps=1e-6,max_iter=2000,dt=.1,damp=1e-6)
            q=np.asarray(q);limits=np.asarray(self.art.get_qlimits());legal=bool(np.all(q>=limits[:,0]-1e-6)and np.all(q<=limits[:,1]+1e-6))
            row={'id':len(records),'palm_forward_roll_deg':roll,'grasp_center_lower_m':delta,'palm_target_matrix':target.tolist(),'ik_in_limits':bool(ok and legal),'samples':[]}
            clean_so_far=True
            if ok and legal:
                for fraction in np.linspace(0,1,21):
                    hand=opened+(closed-opened)*fraction;v=q.copy()
                    for n,x in self.maps['right'].state(hand)[0].items():v[self.index[n]]=x
                    self.model.set_qpos(v,full=True);contacts={};gaps={}
                    for name,obj in (('cap',self.cap_obj),('table',self.table_obj),('body',self.body_obj)):
                        ds=[(n,*self.distance(self.objects[n],obj))for n in self.right_names]
                        contacts[name]=[n for n,d,c in ds if c or d<.001]
                        gaps[name]=min(d for n,d,c in ds)
                    prohibited_body=[n for n in contacts['body']if not any(x in n for x in ['thumb','index','middle'])]
                    clean_so_far=clean_so_far and not contacts['cap'] and not contacts['table'] and not prohibited_body
                    both=any('thumb'in n for n in contacts['body'])and any(('index'in n or'middle'in n)for n in contacts['body'])
                    rec={'fraction':float(fraction),'sdk6':hand.tolist(),'minimum_gaps_m':gaps,'near_or_contact':contacts,'clean_so_far':bool(clean_so_far),'both_body_sides_near':bool(both),'sdk24_static_configuration':self.sdk24(v)}
                    row['samples'].append(rec)
                    if clean_so_far and both:
                        row['potential_first_opposed_body_sample']=len(row['samples'])-1;row['q36']=v.tolist();break
            records.append(row)
        return records

    def right_cap_first(self):
        """One local cap-first pinch family; no cap pose or force is commanded."""
        self.pk.compute_forward_kinematics(self.q0);initial=self.world_matrix('r_inspire_base_link')
        mask=np.ones(len(self.names),dtype=bool);mask[[self.index[f'r_joint{i}']for i in range(1,7)]]=False
        link=list(self.pk.get_link_names()).index('r_inspire_base_link');inv=np.linalg.inv(matrix(self.art.root_pose))
        records=[];request=self.fcl.CollisionRequest(num_max_contacts=100,enable_contact=True)
        for dy,dz in itertools.product((0.,.005,.010),(.005,.010,.015)):
            target=initial.copy();target[:3,3]+=np.array([0,dy,dz])
            q,ok,err=self.pk.compute_IK_CLIK(link,self.Pose(inv@target),self.q0,mask.tolist(),eps=1e-6,max_iter=2000,dt=.1,damp=1e-6)
            q=np.asarray(q);limits=np.asarray(self.art.get_qlimits());legal=bool(np.all(q>=limits[:,0]-1e-6)and np.all(q<=limits[:,1]+1e-6))
            if not ok or not legal:
                records.append({'id':len(records),'dy_m':dy,'dz_m':dz,'ik_in_limits':False});continue
            for si,st in itertools.product((700.,725.,750.,775.,800.),(650.,675.,700.,725.,750.,775.,800.)):
                hand=[0.,0.,0.,si,st,0.];v=q.copy()
                for n,x in self.maps['right'].state(hand)[0].items():v[self.index[n]]=x
                self.model.set_qpos(v,full=True);collisions=[];minimum={'body':10.,'table':10.,'cap_other':10.};cap_groups={}
                for name in self.right_names:
                    for category,obj in (('body',self.body_obj),('table',self.table_obj)):
                        gap,col=self.distance(self.objects[name],obj);minimum[category]=min(minimum[category],gap)
                        if col or gap<.001:collisions.append([name,category])
                    if not ('index'in name or'thumb'in name):
                        gap,col=self.distance(self.objects[name],self.cap_obj);minimum['cap_other']=min(minimum['cap_other'],gap)
                        if col or gap<.001:collisions.append([name,'cap_other'])
                row={'id':len(records),'dy_m':dy,'dz_m':dz,'ik_in_limits':True,'sdk6':hand,'palm_target_matrix':target.tolist(),'prohibited_collisions':collisions,'minimum_gaps_m':minimum}
                if not collisions:
                    maximum_depth=0.;normals={}
                    for group in ('thumb','index'):
                        vals=[];contacts=[]
                        for name in self.right_names:
                            if group not in name:continue
                            gap,col=self.distance(self.objects[name],self.cap_obj);vals.append((gap,name))
                            result=self.fcl.collide(self.objects[name],self.cap_obj,request)
                            for c in result.get_contacts():
                                maximum_depth=max(maximum_depth,float(c.penetration_depth))
                                normal=np.asarray(c.normal);axisdot=abs(float(normal@self.bottle_matrix[:3,2]))
                                if axisdot<.5:contacts.append({'link':name,'normal':normal.tolist(),'depth_m':float(c.penetration_depth)})
                        cap_groups[group]={'nearest_gap_m':min(vals)[0],'contacts':contacts}
                    opposed=min((float(np.dot(a['normal'],b['normal']))for a in cap_groups['thumb']['contacts']for b in cap_groups['index']['contacts']),default=1.)
                    row.update(cap_groups=cap_groups,maximum_cap_penetration_m=maximum_depth,most_opposed_side_normal_dot=opposed,
                        shallow_opposed_cap_candidate=bool(opposed<-.5 and maximum_depth<=.0015),sdk24_static_configuration=self.sdk24(v),q36=v.tolist())
                records.append(row)
        return records

    def hand_samples(self):
        """Pad-centre geometry from original fixed sensor meshes, not TCP guess."""
        palm_name = "l_inspire_base_link"
        records = []
        for index, thumb, rotation in itertools.product((200., 400., 600., 800., 1000.), (0., 200., 400., 600., 800., 1000.), (0., 500., 1000.)):
            sdk = [0., 0., 0., index, thumb, rotation]
            q = self.hand_q(sdk); self.pk.compute_forward_kinematics(q)
            palm = self.world_matrix(palm_name); inv = np.linalg.inv(palm)
            pts = []
            for name in ("l_inspire_thumb_force_sensor_3", "l_inspire_index_force_sensor_3"):
                v = self.vertices[name]; local = .5*(v.min(0)+v.max(0))
                world = self.world_matrix(name) @ np.r_[local, 1.]
                pts.append((inv @ world)[:3])
            sep = float(np.linalg.norm(pts[1]-pts[0]))
            records.append({"sdk6": sdk, "thumb_pad_local": pts[0].tolist(), "index_pad_local": pts[1].tolist(), "pad_center_separation_m": sep})
        return records


def main():
    p = argparse.ArgumentParser(); p.add_argument("--limit", type=int, default=1536); p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--hands",type=int,default=8);p.add_argument("--shift",type=float,nargs=3,default=[0,0,0])
    p.add_argument("--refine",type=Path);p.add_argument("--candidate-id",type=int,default=1244)
    p.add_argument("--right-clearance",action='store_true')
    p.add_argument("--right-sweep",action='store_true');p.add_argument('--snapshot',type=Path)
    p.add_argument('--prebend',type=float,default=1000.)
    p.add_argument('--right-family',action='store_true')
    p.add_argument('--height-extra',type=float,default=0.)
    p.add_argument('--cap-first',action='store_true')
    args = p.parse_args(); t0 = time.time(); g = Geometry(args.output,snapshot=args.snapshot,height_extra=args.height_extra)
    g.shift_held_reference(args.shift)
    if args.cap_first:
        records=g.right_cap_first();valid=[x['id']for x in records if x.get('shallow_opposed_cap_candidate')]
        result={**g.receipt,'probe':'original90 cap-first local pinch family','records':records,'candidate_ids':valid,'executed':False}
        (args.output/'cap_first.json').write_text(json.dumps(result,indent=2));print(json.dumps({'candidate_ids':valid,'elapsed_s':time.time()-t0}));return
    if args.right_family:
        records=g.right_rotated_family(args.prebend);valid=[x['id']for x in records if 'potential_first_opposed_body_sample'in x]
        result={**g.receipt,'probe':'finite right palm forward-axis roll plus lower contact-centre family','prebend':args.prebend,'records':records,'potential_candidate_ids':valid,'executed':False}
        (args.output/'right_family.json').write_text(json.dumps(result,indent=2));print(json.dumps({'potential_candidate_ids':valid,'elapsed_s':time.time()-t0}));return
    if args.right_sweep:
        result={**g.receipt,'probe':'right-hand lower grasp discrete closing geometry, no dynamics','records':g.right_closing_sweep(args.prebend)}
        (args.output/'right_sweep.json').write_text(json.dumps(result,indent=2));print(json.dumps({'saved':str(args.output/'right_sweep.json'),'elapsed_s':time.time()-t0}));return
    if args.right_clearance:
        records=g.right_clearance_scan()
        result={**g.receipt,'probe':'right-hand lower grasp endpoint geometry, not execution','records':records}
        (args.output/'right_clearance.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:len(v)if isinstance(v,list)else v for k,v in result.items()},indent=2));return
    if args.refine:
        original=json.loads(args.refine.read_text())[args.candidate_id]
        base=np.asarray(original['q36']);sdk=np.asarray(original['left_sdk6']);records=[]
        for di,dt,dr in itertools.product((-150.,-100.,-50.,-25.,0.,25.,50.,100.,150.),(-150.,-100.,-50.,0.,50.,100.,150.),(-100.,0.,100.)):
            hand=sdk.copy();hand[3:6]+=np.array([di,dt,dr])
            if np.any(hand<0) or np.any(hand>1000):continue
            q=base.copy()
            for n,v in g.maps['left'].state(hand)[0].items():q[g.index[n]]=v
            record={"id":len(records),"left_sdk6":hand.tolist(),"sdk24_static_configuration":g.sdk24(q),"q36":q.tolist(),**g.screen(q)}
            if record['collision_free_except_intended_cap_contact']:
                record['cap_contacts']=g.cap_contact_depths()
                d=record['cap_contacts']
                record['shallow_opposing_cap_contact']=bool(d['groups']['thumb'] and d['groups']['index'] and d['max_depth_m']<=.0015 and d['most_opposed_normal_dot']<-.5)
            records.append(record)
        valid=[r for r in records if r.get('shallow_opposing_cap_contact')]
        summary={**g.receipt,"refinement_source":str(args.refine),"refinement_source_id":args.candidate_id,"tested":len(records),
            "collision_free_static":sum(r['collision_free_except_intended_cap_contact'] for r in records),"shallow_opposing_contact_count":len(valid),
            "candidate_ids":[r['id'] for r in valid],"physical_execution":False,"elapsed_s":time.time()-t0}
        (args.output/'refinement.json').write_text(json.dumps(records,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return
    samples = g.hand_samples()
    (args.output / "finger_pad_samples.json").write_text(json.dumps(samples, indent=2))
    # Match the two pad centres to opposite sides of the frozen cap. The true
    # full-mesh test below determines whether their finite thickness clashes.
    diameter = 2*g.spec["cap"]["outer_radius_m"]
    chosen = sorted(samples, key=lambda x: abs(x["pad_center_separation_m"]-(diameter+.006)))[:args.hands]
    candidates = []; arm_idx = [g.index[f"l_joint{i}"] for i in range(1,7)]
    mask = np.ones(len(g.names), dtype=bool); mask[arm_idx] = False
    palm_index = list(g.pk.get_link_names()).index("l_inspire_base_link")
    root_inv = np.linalg.inv(matrix(g.art.root_pose)); cap_z = g.bottle_matrix[:3, 2]
    center = g.cap_matrix[:3, 3] - cap_z*.004
    for hand, yaw_deg, roll_deg, zshift in itertools.product(chosen, (0., 45., 90., 135., 180., 225., 270., 315.), (0., 45., 90., 135., 180., 225., 270., 315.), (0., .006, .012)):
        if len(candidates) >= args.limit: break
        qbase = g.hand_q(hand["sdk6"])
        a, b = np.asarray(hand["thumb_pad_local"]), np.asarray(hand["index_pad_local"])
        local_axis = b-a; local_up = np.array([0., 0., -1.])
        local = frame(local_axis, local_up)
        yaw = math.radians(yaw_deg); axis = g.bottle_matrix[:3,:3] @ [math.cos(yaw), math.sin(yaw), 0.]
        rot = frame(axis, cap_z) @ Rotation.from_rotvec([math.radians(roll_deg), 0., 0.]).as_matrix() @ local.T
        target = np.eye(4); target[:3,:3]=rot; target[:3,3]=center + zshift*cap_z - rot@(.5*(a+b))
        rng=np.random.default_rng(1022+len(candidates))
        seeds=[qbase,np.zeros(len(qbase))]
        for _ in range(2):
            seed=qbase.copy();seed[arm_idx]=rng.uniform(-2.,2.,6);seeds.append(seed)
        sol = None; err_best = None
        for seed in seeds:
            seed = qbase.copy() if seed is qbase else np.where(mask,qbase,seed)
            q, ok, err = g.pk.compute_IK_CLIK(palm_index, g.Pose(root_inv@target), seed, mask.tolist(), eps=1e-6, max_iter=1000, dt=.1, damp=1e-6)
            q=np.asarray(q); err_best=float(np.linalg.norm(err))
            # A solver's success flag alone is not a joint-limit certificate.
            limits=np.asarray(g.art.get_qlimits()); legal=bool(np.all(q>=limits[:,0]-1e-6) and np.all(q<=limits[:,1]+1e-6))
            if ok and legal: sol=q;break
        record = {"id":len(candidates),"left_sdk6":hand["sdk6"],"yaw_deg":yaw_deg,"roll_deg":roll_deg,
            "palm_target_matrix":target.tolist(),"pad_center_separation_m":hand["pad_center_separation_m"],"pad_height_shift_m":zshift,
            "ik_solved_in_limits":sol is not None,"last_ik_error_norm":err_best}
        if sol is not None:
            record.update(g.screen(sol));record["sdk24_static_configuration"]=g.sdk24(sol);record["q36"]=sol.tolist()
        candidates.append(record)
        if len(candidates)%16==0:
            print(json.dumps({"candidates":len(candidates),"ik":sum(x['ik_solved_in_limits'] for x in candidates),"elapsed_s":time.time()-t0}),flush=True)
    valid=[x for x in candidates if x.get("collision_free_except_intended_cap_contact")]
    near=sorted(valid,key=lambda x:max(v['gap_m'] for v in x['cap_nearest'].values()))
    summary={**g.receipt,"candidate_count":len(candidates),"ik_solutions":sum(x['ik_solved_in_limits'] for x in candidates),
        "collision_free_static_count":len(valid),"best_candidate_ids":[x['id'] for x in near[:10]],"elapsed_s":time.time()-t0,
        "geometry_success_is_not_robot_execution":True,"opening_executed":False,"pouring_executed":False,
        "limits":"Finite pose family; no impossibility proof. No path/regrasp/force closure/contact dynamics verification."}
    (args.output/"candidates.json").write_text(json.dumps(candidates,indent=2));(args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
