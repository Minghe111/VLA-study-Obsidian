"""Construct left cap pinches in the hand's own geometry, before arm IK.

Original sensor collision meshes supply exposed surface centroids/normals.
Only the hand's intrinsic FK is sampled. No right-hand pose, held-bottle height,
arm trajectory, physical contact, or successful manipulation is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sapien
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

PIPE=Path(__file__).resolve().parents[1]
OUT=PIPE/'cap_geometry_probes/left_intrinsic'
VENDOR=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4')
NATIVE=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin')
LEGACY=NATIVE/'integrations/realman_rm65b_gen4/teacher_legacy'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def matrix(pose):return np.asarray(pose.to_transformation_matrix(),dtype=float)


class IntrinsicHand:
    def __init__(self,out=OUT):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.fma=load('intrinsic_original_collision_geometry',LEGACY/'full_mesh_path_audit_bounded.py')
        self.mapping=load('intrinsic_original_sdk_mapping',LEGACY/'continuous_servo.py')
        self.scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);self.scene.set_timestep(.004)
        source=VENDOR/'robot_sapien.urdf';tree=ET.parse(source)
        for link in tree.getroot().findall('link'):
            for visual in list(link.findall('visual')):link.remove(visual)
        for mesh in tree.getroot().iter('mesh'):
            original=Path(mesh.get('filename'));mesh.set('filename',str(original if original.is_absolute() else (VENDOR/original).resolve()))
        self.derived=self.out/'intrinsic_collision_only_robot.urdf';tree.write(self.derived,encoding='utf-8',xml_declaration=True)
        loader=self.scene.create_urdf_loader();loader.fix_root_link=True;self.art=loader.parse(str(self.derived))[0][0].build()
        self.names=[j.name for j in self.art.active_joints];self.joint_index={n:i for i,n in enumerate(self.names)}
        self.links={l.name:l for l in self.art.links};self.link_index={l.name:i for i,l in enumerate(self.art.links)}
        self.pk=self.art.create_pinocchio_model()
        self.map=self.mapping.ContinuousHandMapping(json.loads((VENDOR/'source_manifest.json').read_text())['hands']['left'])
        self.base_q=np.zeros(len(self.names));self.art.set_qpos(self.base_q);self.pk.compute_forward_kinematics(self.base_q)
        self.pads={}
        for name in ('l_inspire_thumb_force_sensor_3','l_inspire_thumb_force_sensor_4','l_inspire_index_force_sensor_2','l_inspire_index_force_sensor_3'):
            self.pads[name]=self.exposed_pad_surfaces(name)
        fk=[]
        for name in self.links:
            fk.append(float(np.linalg.norm(matrix(self.pk.get_link_pose(self.link_index[name]))[:3,3]-matrix(self.links[name].entity.pose)[:3,3])))
        if max(fk)>2e-5:raise RuntimeError('Intrinsic CPU FK does not match original articulation')
        self.receipt={'scope':'intrinsic left-hand surface/FK construction only','right_hand_or_held_height_bound':False,'arm_IK_tested':False,
            'robot_rollout_tested':False,'renderer_created':False,'physics_steps':0,'cap_geometry':{'outer_radius_m':.030,'local_z_bounds_m':[-.012,.002],'mid_height_m':-.005},
            'joint_count':len(self.names),'link_count':len(self.links),'initial_FK_max_position_error_m':max(fk),
            'normal_method':'Actual convex sensor hull facet normals, selected near the thin PCA axis; faces occluded by the parent phalanx convex collision hull are rejected',
            'surface_exposure_test':'Actual facet centroid is outside the real parent convex collision hull; no outward inflation or sensor-frame sign assumed',
            'geometry_to_physics_boundary':'Facet means/normals are geometric candidate contact patches; actual contact location, forces, friction and force closure remain unverified',
            'source_sha256':{str(p):sha(p)for p in [source,VENDOR/'source_manifest.json',LEGACY/'continuous_servo.py',LEGACY/'full_mesh_path_audit_bounded.py',Path(__file__)]}}

    def link_T(self,name):return matrix(self.pk.get_link_pose(self.link_index[name]))

    def exposed_pad_surfaces(self,name):
        link=self.links[name];vertices=self.fma._link_vertices(link)[0]
        center=vertices.mean(0);eigen,basis=np.linalg.eigh((vertices-center).T@(vertices-center)/len(vertices));axis=basis[:,0]
        sensor_hull=ConvexHull(vertices);parent=link.parent
        parent_vertices=self.fma._link_vertices(parent)[0];parent_hull=ConvexHull(parent_vertices)
        sensor_to_parent=np.linalg.inv(self.link_T(parent.name))@self.link_T(name)
        groups={-1:[],1:[]};occluded=0;other=0
        for triangle,equation in zip(sensor_hull.simplices,sensor_hull.equations):
            normal=equation[:3];alignment=float(normal@axis)
            if abs(alignment)<.6:other+=1;continue
            tri=vertices[triangle];area=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2)
            point=tri.mean(0);outside=sensor_to_parent@np.r_[point,1.]
            inside=bool(np.all(parent_hull.equations[:,:3]@outside[:3]+parent_hull.equations[:,3]<=1e-8))
            if inside:occluded+=1;continue
            groups[1 if alignment>0 else -1].append((area,point,normal,triangle))
        surfaces=[]
        for sign,faces in groups.items():
            if not faces:continue
            total=sum(x[0]for x in faces)
            surface=sum(x[0]*x[1]for x in faces)/total;normal=sum(x[0]*x[2]for x in faces);normal/=np.linalg.norm(normal)
            candidate_point=sensor_to_parent@np.r_[surface,1.]
            source='area-weighted exposed facet centroid'
            # The union of exterior patches need not be convex. Its average
            # could be covered by the parent even when each facet is exposed.
            # In that case use one real exposed facet, never inflate the point.
            if np.all(parent_hull.equations[:,:3]@candidate_point[:3]+parent_hull.equations[:,3]<=1e-8):
                largest=max(faces,key=lambda x:x[0]);total=largest[0];surface=largest[1];normal=largest[2];source='largest actual exposed facet centroid'
            surfaces.append({'surface_id':f'{name}:pca_sign{sign:+d}','sensor_link':name,'surface_local_m':surface.tolist(),'normal_local':normal.tolist(),
                'area_m2':total,'hull_faces':[x[3].tolist()for x in faces],'exposed_facet_count':len(faces),'pca_sign':sign,'patch_point_source':source})
        return {'sensor_link':name,'parent_link':parent.name,'collision_vertices_local':vertices.tolist(),'pca_eigenvalues_m2':eigen.tolist(),
            'thin_normal_unsigned_local':axis.tolist(),'occluded_facets_rejected':occluded,'side_facets_excluded':other,'surfaces':surfaces}

    def sample(self,sdk,group):
        q=self.base_q.copy()
        for n,x in self.map.state(sdk)[0].items():q[self.joint_index[n]]=x
        self.pk.compute_forward_kinematics(q);inv=np.linalg.inv(self.link_T('l_inspire_base_link'));out=[]
        for name,data in self.pads.items():
            if group not in name:continue
            T=inv@self.link_T(name)
            for surface in data['surfaces']:
                point=(T@np.r_[surface['surface_local_m'],1.])[:3];normal=T[:3,:3]@np.asarray(surface['normal_local']);normal/=np.linalg.norm(normal)
                out.append({'sdk6':list(sdk),'surface_id':surface['surface_id'],'sensor_link':name,'surface_in_palm_m':point.tolist(),
                    'normal_in_palm':normal.tolist(),'q_left_hand':{n:float(q[self.joint_index[n]])for n in self.map.joint_names}})
        return out


def canonical_transform(a,b,roll_deg=0.):
    a,b=np.asarray(a),np.asarray(b);mid=.5*(a+b);x=b-a;x/=np.linalg.norm(x)
    toward_palm=-mid;toward_palm-=x*(toward_palm@x)
    if np.linalg.norm(toward_palm)<1e-8:raise RuntimeError('Degenerate hand approach basis')
    z=toward_palm/np.linalg.norm(toward_palm);y=np.cross(z,x);basis=np.stack([x,y,z],axis=1)
    R=Rotation.from_rotvec([math.radians(roll_deg),0,0]).as_matrix()@basis.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=np.array([0,0,-.005])-R@mid
    # Jaw line is horizontal by construction; equal pad heights are not learned.
    return T


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--count',type=int,default=12);args=p.parse_args();t0=time.time()
    g=IntrinsicHand(args.output);index_samples=[];thumb_samples=[]
    for index in range(0,1001,20):index_samples.extend(g.sample([0,0,0,index,1000,0],'index'))
    for flex in range(0,1001,20):
        for opposition in range(0,1001,50):thumb_samples.extend(g.sample([0,0,0,1000,flex,opposition],'thumb'))
    (args.output/'pad_surfaces.json').write_text(json.dumps(g.pads,indent=2))
    (args.output/'index_samples.json').write_text(json.dumps(index_samples))
    (args.output/'thumb_samples.json').write_text(json.dumps(thumb_samples))
    pairs=[];near_count=0;max_facing=-1.;best_relaxed=[]
    for t in thumb_samples:
        a=np.asarray(t['surface_in_palm_m']);na=np.asarray(t['normal_in_palm'])
        for i in index_samples:
            b=np.asarray(i['surface_in_palm_m']);nb=np.asarray(i['normal_in_palm']);line=b-a;distance=float(np.linalg.norm(line));direction=line/distance
            if not .058<=distance<=.066:continue
            near_count+=1;facing=min(float(na@direction),float(-nb@direction));opposed=float(na@nb);max_facing=max(max_facing,facing)
            basic={'thumb':t,'index':i,'surface_separation_m':distance,'min_facing_cosine':facing,'normal_dot':opposed}
            orbital_min=abs(float((.5*(a+b))@direction))
            basic['minimum_geometric_palm_orbit_radius_m']=orbital_min
            basic['score']=abs(distance-.0608)/.001+.5*(1-facing)+.25*(1+opposed)+.6*orbital_min/.01
            best_relaxed.append(basic)
            if facing<.7 or opposed>-.7 or distance<.060:continue
            pairs.append(basic)
    selected=[]
    for candidate in sorted(pairs,key=lambda x:x['score']):
        sdk=[0,0,0,candidate['index']['sdk6'][3],candidate['thumb']['sdk6'][4],candidate['thumb']['sdk6'][5]]
        if any(np.linalg.norm(np.asarray(sdk[3:])-np.asarray(x['left_sdk6'][3:]))<65 and
               candidate['thumb']['sensor_link']==x['thumb']['sensor_link'] and candidate['index']['sensor_link']==x['index']['sensor_link']for x in selected):continue
        item={**candidate,'candidate_id':len(selected),'left_sdk6':sdk,'q_left_hand':g.map.state(sdk)[0],
            'T_cap_from_palm_roll0':canonical_transform(candidate['thumb']['surface_in_palm_m'],candidate['index']['surface_in_palm_m']).tolist(),
            'future_transform':'T_world_palm = T_world_cap @ Rz(cap_yaw) @ T_cap_palm(jaw_roll). Roll is about the horizontal jaw line through cap-local midpoint [0,0,-0.005], not about cap actor origin. Bind actual held cap pose only.',
            'precontact_geometry_only':True}
        # Verify the claimed elimination of pad height mismatch numerically.
        T=np.asarray(item['T_cap_from_palm_roll0']);pa=T@np.r_[candidate['thumb']['surface_in_palm_m'],1.];pb=T@np.r_[candidate['index']['surface_in_palm_m'],1.]
        item['canonical_pad_points_m']=[pa[:3].tolist(),pb[:3].tolist()];item['canonical_pad_height_difference_m']=abs(float(pa[2]-pb[2]))
        roll_family=[]
        for roll in range(-30,31,5):
            Tc=canonical_transform(candidate['thumb']['surface_in_palm_m'],candidate['index']['surface_in_palm_m'],roll)
            normals=[Tc[:3,:3]@np.asarray(candidate[side]['normal_in_palm'])for side in ('thumb','index')]
            roll_family.append({'jaw_line_roll_deg':roll,'T_cap_from_palm':Tc.tolist(),'palm_orbit_radius_m':float(np.linalg.norm(Tc[:2,3])),
                'palm_height_above_cap_origin_m':float(Tc[2,3]),'max_abs_contact_normal_Z':max(abs(float(n[2]))for n in normals)})
        item['preferred_roll_family']=sorted(roll_family,key=lambda x:x['palm_orbit_radius_m']+.06*x['max_abs_contact_normal_Z'])[:3]
        selected.append(item)
        if len(selected)>=args.count:break
    summary={**g.receipt,'index_surface_samples':len(index_samples),'thumb_surface_samples':len(thumb_samples),'pairs_in58to66mm_band':near_count,
        'strict_opposing_pairs':len(pairs),'max_min_facing_cosine_in_distance_band':max_facing,'selected_count':len(selected),'elapsed_s':time.time()-t0,
        'criteria':{'precontact_surface_gap_m':[.060,.066],'target_gap_m':.0608,'minimum_facing_cosine':.7,'maximum_normal_dot':-.7},
        'pending':'Bind actual115 held q36/root/body/cap pose; enumerate canonical yaw/roll; full hand/body/cap/right-hand collision and arm IK; actual contact pinch and regrasp rollout.',
        'opening_arc_boundary':'Each eventual 45-90degree opening arc must move the palm around measured body-Z and account for measured axial cap rise. A static endpoint is not a swept path, and grasp is not assumed permanently rigid.',
        'minimum_selected_palm_orbit_m':min((x['minimum_geometric_palm_orbit_radius_m']for x in selected),default=None)}
    (args.output/'intrinsic_candidates.json').write_text(json.dumps(selected,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'best_relaxed_pairs.json').write_text(json.dumps(sorted(best_relaxed,key=lambda x:(-x['min_facing_cosine'],x['score']))[:20],indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
