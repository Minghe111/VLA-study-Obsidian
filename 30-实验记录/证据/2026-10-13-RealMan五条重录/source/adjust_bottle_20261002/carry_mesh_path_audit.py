"""CPU full native carry-path mesh audit; no SAPIEN writes or physics steps.
Bottle vertices follow a frozen measured palm^-1*bottle relation. Only grasp-side
hand/bottle pairs with current nonzero normal impulse are excluded. Full robot
self/table and bottle/other robot/table geometry remain checked. A rigid hold is
a hypothesis: actual force support/slip and tracking need physics guards.
"""
from __future__ import annotations
import math
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace
import numpy as np
import full_mesh_path_audit as fma


def _matrix(value):
    if hasattr(value,'to_transformation_matrix'):value=value.to_transformation_matrix()
    m=fma._array(value)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-8):
        raise ValueError('Expected finite homogeneous 4x4 pose')
    if not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=2e-5) or np.linalg.det(m[:3,:3])<.9999:
        raise ValueError('Expected SO(3) rotation')
    return m.copy()


def rigid_bottle_pose(palm_world,palm_to_bottle):
    return _matrix(palm_world)@_matrix(palm_to_bottle)


def attached_motion_coefficients(urdf_tree,anchor_name,joint_names,attached_vertices_in_anchor):
    vertices=fma._array(attached_vertices_in_anchor)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Expected finite nonempty N x 3 attached vertices')
    names=list(joint_names);coeff=np.zeros(len(names));ancestors=set();parent={}
    for j in urdf_tree.findall('joint'):
        origin=j.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError('Invalid joint origin')
        child=j.find('child').get('link')
        if child in parent:raise ValueError('Multiple parents in kinematic tree')
        parent[child]=(j.find('parent').get('link'),j.get('name'),j.get('type'),float(np.linalg.norm(xyz)))
    radius=float(np.linalg.norm(vertices,axis=1).max());node=anchor_name;visited=set()
    while node in parent:
        if node in visited:raise ValueError('Kinematic cycle')
        visited.add(node);next_node,joint,kind,length=parent[node]
        if joint in names:
            index=names.index(joint);ancestors.add(index)
            if kind not in ('revolute','continuous'):raise ValueError('Carry bound requires revolute active joints: '+kind)
            coeff[index]=radius
        radius+=length;node=next_node
    return coeff,ancestors


def relative_motion_coefficients(coeff_a,ancestors_a,coeff_b,ancestors_b):
    # Common ancestors move both objects by exactly the same rigid transform.
    # Distance is invariant to it; only relative branch joints need a bound.
    result=np.asarray(coeff_a,dtype=float)+np.asarray(coeff_b,dtype=float)
    result=result.copy()
    for index in set(ancestors_a)&set(ancestors_b):result[index]=0.
    return result


def carry_path_samples(path,actual_q,arm_indices,coefficients,*,vertex_step_m=.001,max_samples=20000):
    if not math.isfinite(vertex_step_m) or not 0.<vertex_step_m<=.001:raise ValueError('vertex_step_m must be in (0,1mm]')
    if not isinstance(max_samples,int) or max_samples<2:raise ValueError('Invalid max_samples')
    raw=fma._array(path['position'] if isinstance(path,dict) else path);q=fma._array(actual_q)
    if q.ndim!=1 or not np.isfinite(q).all() or raw.ndim!=2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Expected finite nonempty 2D command path and finite full actual q')
    indices=list(arm_indices)
    if len(set(indices))!=len(indices) or any(i<0 or i>=len(q) for i in indices):raise ValueError('Invalid arm indices')
    if raw.shape[1]==len(indices):
        full=np.tile(q,(len(raw),1));full[:,indices]=raw;command=indices
    elif raw.shape[1]==len(q):full=raw.copy();command=list(range(len(q)))
    else:raise ValueError('Path width must be arm DOF or all actual DOFs')
    coeff=np.asarray(list(coefficients.values()),dtype=float)
    if coeff.ndim!=2 or coeff.shape[1]!=len(q) or not np.isfinite(coeff).all() or np.any(coeff<0):raise ValueError('Invalid motion coefficients')
    full=np.vstack([q,full]);samples=[(full[0].copy(),-1,0.)];peak=0.
    for row,(q0,q1) in enumerate(zip(full[:-1],full[1:])):
        bound=float(np.max(coeff@np.abs(q1-q0)));pieces=max(1,int(math.ceil(bound/vertex_step_m)))
        if len(samples)+pieces>max_samples:raise ValueError('Adaptive carry subdivision exceeded max_samples; incomplete audit refused')
        peak=max(peak,bound/pieces)
        for k in range(1,pieces+1):samples.append((q0+(q1-q0)*k/pieces,row,k/pieces))
    return full,samples,command,peak


def real_grasp_contact_evidence(contacts,bottle_name,eligible_hand_links,*,normal_impulse_threshold_Ns=1e-10):
    if not math.isfinite(normal_impulse_threshold_Ns) or normal_impulse_threshold_Ns<0:raise ValueError('Invalid normal impulse threshold')
    eligible=set(eligible_hand_links);allowed={};zero=[];foreign=[]
    for c in contacts:
        names=[z.entity.name for z in c.bodies]
        if len(names)!=2 or bottle_name not in names:continue
        other=names[1] if names[0]==bottle_name else names[0]
        points=[];total=0.
        for p in c.points:
            impulse=fma._array(p.impulse);normal=fma._array(p.normal)
            if impulse.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(impulse).all() or not np.isfinite(normal).all():
                raise ValueError('Nonfinite contact impulse/normal')
            nn=float(np.linalg.norm(normal))
            if nn<1e-10:raise ValueError('Zero contact normal')
            magnitude=abs(float(impulse@normal/nn))
            if magnitude>normal_impulse_threshold_Ns:
                total+=magnitude;points.append({'normal_impulse_Ns':magnitude,'impulse_Ns':impulse.tolist(),'normal':normal.tolist()})
        entry={'hand_link':other,'normal_impulse_Ns':total,'active_point_count':len(points),'points':points}
        if other not in eligible:
            if points:foreign.append(entry)
        elif not points:zero.append(entry)
        else:
            if other not in allowed:allowed[other]=entry
            else:
                allowed[other]['normal_impulse_Ns']+=total;allowed[other]['active_point_count']+=len(points);allowed[other]['points']+=points
    return {'allowed_exact_hand_links':sorted(allowed),'active_contact_evidence':list(allowed.values()),
            'zero_impulse_near_contacts_not_excluded':zero,'foreign_bottle_contacts_not_excluded':foreign,
            'normal_impulse_threshold_Ns':normal_impulse_threshold_Ns}


class CarryMeshPathAuditor(fma.FullMeshPathAuditor):
    def __init__(self,task,side='left',*,palm_to_bottle=None,palm_link_name=None,
                 max_reference_position_error_m=.001,max_reference_rotation_error_rad=math.radians(.5),
                 require_real_grasp_contact=True,normal_impulse_threshold_Ns=1e-10,check_table=True):
        if side not in ('left','right'):raise ValueError('side must be left/right')
        for value in (max_reference_position_error_m,max_reference_rotation_error_rad):
            if not math.isfinite(value) or value<0:raise ValueError('Invalid rigid reference tolerance')
        import preflight_place_goals as ppg
        self.ppg=ppg;self.physical_before=ppg._snapshot(task)
        super().__init__(task,check_table=check_table)
        self.side=side;self.palm_name=palm_link_name or ('l_' if side=='left' else 'r_')+'inspire_base_link'
        if self.palm_name not in self.objects:raise ValueError('Missing actual grasp palm collision link '+self.palm_name)
        measured_palm=self.initial_scene_poses[self.palm_name]
        measured_bottle=_matrix(task.bottle.get_pose());measured=np.linalg.inv(measured_palm)@measured_bottle
        self.reference=measured if palm_to_bottle is None else _matrix(palm_to_bottle)
        error_position=float(np.linalg.norm(self.reference[:3,3]-measured[:3,3]))
        dr=self.reference[:3,:3]@measured[:3,:3].T;u,_,vt=np.linalg.svd(dr)
        error_rotation=float(math.acos(float(np.clip((np.trace(u@vt)-1)/2,-1,1))))
        if error_position>max_reference_position_error_m or error_rotation>max_reference_rotation_error_rad:
            raise ValueError(f'Frozen grip reference disagrees with measured rigid relation: position {error_position}m, rotation {error_rotation}rad')
        from sapien.physx import PhysxRigidBaseComponent
        component=task.bottle.actor.find_component_by_type(PhysxRigidBaseComponent)
        self.bottle_vertices,_=fma._link_vertices(component)
        in_palm=self.bottle_vertices@self.reference[:3,:3].T+self.reference[:3,3]
        self.bottle_coefficients,self.bottle_ancestors=attached_motion_coefficients(self.urdf_tree,self.palm_name,self.sim_joints,in_palm)
        self.ancestors={}
        for name,v in self.vertices.items():_,self.ancestors[name]=attached_motion_coefficients(self.urdf_tree,name,self.sim_joints,v)
        eligible=[]
        for name in self.objects:
            node=self.links[name]
            while node is not None and node.name!=self.palm_name:node=node.parent
            if node is not None:eligible.append(name)
        self.contact_evidence=real_grasp_contact_evidence(task.scene.get_contacts(),component.entity.name,eligible,
            normal_impulse_threshold_Ns=normal_impulse_threshold_Ns)
        self.allowed_bottle_hand=set(self.contact_evidence['allowed_exact_hand_links'])
        if require_real_grasp_contact and not self.allowed_bottle_hand:raise ValueError('No current real grasp-hand/bottle normal impulse; rigid carry audit refused')
        self.self_pair_coeff={(a,b):relative_motion_coefficients(self.motion_coefficients[a],self.ancestors[a],self.motion_coefficients[b],self.ancestors[b]) for a,b in self.self_pairs}
        self.bottle_pair_coeff={n:relative_motion_coefficients(self.bottle_coefficients,self.bottle_ancestors,c,self.ancestors[n])
                                for n,c in self.motion_coefficients.items() if n not in self.allowed_bottle_hand}
        self.original_bottle_pose=self.Pose(self.static['bottle'].pose)
        self.metadata.update(rigid_carry_reference=self.reference.tolist(),measured_palm_to_bottle=measured.tolist(),
            reference_position_error_m=error_position,reference_rotation_error_rad=error_rotation,
            max_reference_position_error_m=max_reference_position_error_m,max_reference_rotation_error_rad=max_reference_rotation_error_rad,
            palm_link=self.palm_name,real_grasp_contact_evidence=self.contact_evidence,
            allowed_bottle_robot_pairs=[[n,'bottle'] for n in sorted(self.allowed_bottle_hand)],
            checked_bottle_robot_links=len(self.bottle_pair_coeff),
            attached_bottle_vertex_radius_in_palm_m=float(np.linalg.norm(in_palm,axis=1).max()),
            bottle_motion_coefficients_m_per_rad={n:float(v) for n,v in zip(self.sim_joints,self.bottle_coefficients) if v},
            relative_pair_bound='Common ancestor rigid transforms cancel exactly in pairwise distance. All noncommon branch displacement bounds retained.')
        predicted=rigid_bottle_pose(_matrix(self.objects[self.palm_name].pose),self.reference)
        self.metadata['predicted_start_bottle_vs_measured_position_m']=float(np.linalg.norm(predicted[:3,3]-measured_bottle[:3,3]))
        ppg._assert_snapshot(task,self.physical_before)

    def audit(self,path,side=None,*,vertex_step_m=.001,min_bottle_clearance_m=0.,
              min_table_clearance_m=0.,min_self_clearance_m=0.,max_samples=20000,include_samples=False):
        started=time.monotonic();side=self.side if side is None else side
        if side!=self.side:raise ValueError('Carry side differs from frozen grip side')
        for value in (min_bottle_clearance_m,min_table_clearance_m,min_self_clearance_m):
            if not math.isfinite(value) or value<0:raise ValueError('Clearance thresholds must be finite nonnegative')
        self.ppg._assert_snapshot(self.task,self.physical_before)
        names=list(getattr(self.task.robot,side+'_arm_joints_name'));indices=[self.sim_joints.index(n) for n in names]
        coefficients={**self.motion_coefficients,'__carried_bottle__':self.bottle_coefficients}
        full,samples,command,peak=carry_path_samples(path,self.initial_sim_q,indices,coefficients,
            vertex_step_m=vertex_step_m,max_samples=max_samples)
        limits=np.concatenate([fma._array(j.limits).reshape(-1,2) for j in self.art.active_joints])
        limit_audit=fma.joint_limit_audit(full,limits,self.sim_joints,command)
        endpoint={-1:0}
        for i,(_,row,frac) in enumerate(samples):
            if frac==1.:endpoint[row]=i
        for metric in ('commanded','frozen_measured','start_measured','overall'):
            worst=limit_audit[metric]['worst']
            if worst is not None:worst['sample_index']=endpoint[worst['path_index']]
        categories=['bottle_robot','self']
        if 'table' in self.static:categories+=['bottle_table','robot_table']
        thresholds={'bottle_robot':min_bottle_clearance_m,'self':min_self_clearance_m,
                    'bottle_table':min_table_clearance_m,'robot_table':min_table_clearance_m}
        best={n:{'distance_m':math.inf} for n in categories}
        certified={n:{'lower_bound_m':math.inf} for n in categories}
        events=[];seen=set();occurrences=0;cache={};sample_records=[]
        deltas=np.abs(full-full[0]);object_moving={n:bool(np.any(deltas@c>1e-12)) for n,c in coefficients.items()}
        self_moving={pair:bool(np.any(deltas@c>1e-12)) for pair,c in self.self_pair_coeff.items()}
        bottle_pair_moving={n:bool(np.any(deltas@c>1e-12)) for n,c in self.bottle_pair_coeff.items()}
        source_bottle=self.static['bottle']
        try:
            for sample_index,(q,path_index,fraction) in enumerate(samples):
                self.model.set_qpos(q,full=True)
                outgoing=np.abs(samples[sample_index+1][0]-q) if sample_index+1<len(samples) else np.zeros_like(q)
                bottle_pose=rigid_bottle_pose(_matrix(self.objects[self.palm_name].pose),self.reference)
                # Installed FCLObject.pose is read-only; constructor updates only
                # independent auxiliary CollisionObjects, never SAPIEN shapes.
                bottle=self.fcl.FCLObject(source_bottle.name,self.Pose(bottle_pose),source_bottle.shapes,source_bottle.shape_poses)
                boxes={}
                for n,obj in self.objects.items():
                    t=_matrix(obj.pose);v=self.vertices[n]@t[:3,:3].T+t[:3,3]
                    boxes[n]=np.stack([v.min(0),v.max(0)])
                bv=self.bottle_vertices@bottle_pose[:3,:3].T+bottle_pose[:3,3]
                bottle_box=np.stack([bv.min(0),bv.max(0)])
                stamp={'sample_index':sample_index,'path_index':path_index,'interval_fraction':fraction}
                row_min={n:math.inf for n in categories}
                def evaluate(category,link,other,first_obj,second_obj,first_box,second_box,motion,invariant):
                    nonlocal occurrences
                    lower=fma._aabb_gap(first_box,second_box)
                    if lower>0. and lower>best[category]['distance_m'] and lower-motion>certified[category]['lower_bound_m']:return
                    key=(category,link,other)
                    if invariant and key in cache:gap,collision=cache[key]
                    else:
                        gap,collision=self._distance(first_obj,second_obj)
                        if invariant:cache[key]=(gap,collision)
                    row_min[category]=min(row_min[category],gap);item={**stamp,'link':link,'other':other}
                    if gap<best[category]['distance_m']:best[category]={**item,'distance_m':gap}
                    if gap-motion<certified[category]['lower_bound_m']:
                        certified[category]={**item,'lower_bound_m':gap-motion,'relative_vertex_bound_m':motion}
                    if collision:
                        occurrences+=1
                        if key not in seen:
                            seen.add(key)
                            if len(events)<64:events.append({**item,'category':category})
                for name,coeff in self.bottle_pair_coeff.items():
                    evaluate('bottle_robot',name,'bottle',self.objects[name],bottle,boxes[name],bottle_box,
                             float(coeff@outgoing),not bottle_pair_moving[name])
                if 'table' in self.static:
                    evaluate('bottle_table','bottle','table',bottle,self.static['table'],bottle_box,self.static_boxes['table'],
                             float(self.bottle_coefficients@outgoing),not object_moving['__carried_bottle__'])
                    for name,obj in self.objects.items():
                        evaluate('robot_table',name,'table',obj,self.static['table'],boxes[name],self.static_boxes['table'],
                                 float(self.motion_coefficients[name]@outgoing),not object_moving[name])
                for (first,second),coeff in self.self_pair_coeff.items():
                    evaluate('self',first,second,self.objects[first],self.objects[second],boxes[first],boxes[second],
                             float(coeff@outgoing),not self_moving[first,second])
                if include_samples:
                    sample_records.append({**stamp,'predicted_bottle_pose_world':bottle_pose.tolist(),
                        'evaluated_minima_m':{n:(value if math.isfinite(value) else None) for n,value in row_min.items()}})
            passed=limit_audit['passed'] and not seen and all(certified[n]['lower_bound_m']>=thresholds[n] for n in categories)
            result={'passed':bool(passed),'side':side,'metadata':self.metadata,
                'path_rows':len(full)-1,'audited_samples':len(samples),'max_vertex_displacement_bound_m':peak,
                'max_joint_limit_excess_rad':limit_audit['overall']['excess_rad'],'joint_limit_audit':limit_audit,
                'commanded_excess_rad':limit_audit['commanded']['excess_rad'],
                'frozen_excess_rad':limit_audit['frozen_measured']['excess_rad'],
                'start_excess_rad':limit_audit['start_measured']['excess_rad'],
                'sampled_minimum_gap':best,'continuous_gap_lower_bound':certified,
                'thresholds_m':{n:thresholds[n] for n in categories},'collisions_first_64':events,
                'unique_collision_pairs':len(seen),'collision_pair_sample_occurrences':occurrences,
                'invariant_pair_cache_entries':len(cache),'physical_state_unchanged':True,
                'elapsed_s':time.monotonic()-started,
                'scope':'Actual full native meshes, frozen measured/provided rigid grip, linear joint interpolation with robot AND bottle vertex displacement <=1mm. No proof of force support or actual servo tracking.'}
            if include_samples:result['samples']=sample_records
            return result
        finally:
            self.model.set_qpos(self.old_model_q,full=True)
            self.static['bottle']=self.fcl.FCLObject(source_bottle.name,self.original_bottle_pose,source_bottle.shapes,source_bottle.shape_poses)
            self.ppg._assert_snapshot(self.task,self.physical_before)


def audit_carry_path(task,path,side='left',*,palm_to_bottle=None,palm_link_name=None,
                     max_reference_position_error_m=.001,max_reference_rotation_error_rad=math.radians(.5),
                     require_real_grasp_contact=True,normal_impulse_threshold_Ns=1e-10,check_table=True,**kwargs):
    """Fresh CPU auditor; physical scene cannot change during planning preflight.
    palm_to_bottle is a frozen REAL measured 4x4 reference. The default rejects
    reference drift >1mm/0.5deg; no pose clipping, snapping or attachments occur.
    """
    auditor=CarryMeshPathAuditor(task,side,palm_to_bottle=palm_to_bottle,palm_link_name=palm_link_name,
        max_reference_position_error_m=max_reference_position_error_m,
        max_reference_rotation_error_rad=max_reference_rotation_error_rad,
        require_real_grasp_contact=require_real_grasp_contact,normal_impulse_threshold_Ns=normal_impulse_threshold_Ns,
        check_table=check_table)
    return auditor.audit(path,side,**kwargs)


def run_cpu_tests():
    tree=ET.fromstring('''<robot name="test"><link name="base"/><link name="common"/><link name="pa"/><link name="pb"/>
      <joint name="root" type="revolute"><parent link="base"/><child link="common"/><origin xyz="0 0 0"/></joint>
      <joint name="a" type="revolute"><parent link="common"/><child link="pa"/><origin xyz=".2 0 0"/></joint>
      <joint name="b" type="revolute"><parent link="common"/><child link="pb"/><origin xyz=".3 0 0"/></joint></robot>''')
    names=['root','a','b'];va=np.array([[.4,0,0],[.6,.01,0]]);vb=np.array([[.2,0,0]])
    ca,aa=attached_motion_coefficients(tree,'pa',names,va);cb,ab=attached_motion_coefficients(tree,'pb',names,vb)
    assert np.allclose(ca,[.2+np.linalg.norm(va,axis=1).max(),np.linalg.norm(va,axis=1).max(),0])
    relative=relative_motion_coefficients(ca,aa,cb,ab)
    assert relative[0]==0. and relative[1]==ca[1] and relative[2]==cb[2]
    def rz(theta):
        c,s=math.cos(theta),math.sin(theta);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    rng=np.random.default_rng(7);worst_ratio=0.
    for _ in range(500):
        q=rng.uniform(-2,2,3);dq=rng.uniform(-.04,.04,3)
        aw=(va@rz(q[1]).T+[.2,0,0])@rz(q[0]).T
        bw=(vb@rz(q[2]).T+[.3,0,0])@rz(q[0]).T
        aw2=(va@rz(q[1]+dq[1]).T+[.2,0,0])@rz(q[0]+dq[0]).T
        bw2=(vb@rz(q[2]+dq[2]).T+[.3,0,0])@rz(q[0]+dq[0]).T
        displacement=float(np.max(np.linalg.norm(aw2-aw,axis=1)))
        assert displacement<=ca@np.abs(dq)+1e-12
        d0=np.min(np.linalg.norm(aw[:,None]-bw[None,:],axis=2));d1=np.min(np.linalg.norm(aw2[:,None]-bw2[None,:],axis=2))
        assert abs(d1-d0)<=relative@np.abs(dq)+1e-12
        worst_ratio=max(worst_ratio,displacement/(ca@np.abs(dq)))
    palm=np.eye(4);palm[:3,:3]=rz(.6);palm[:3,3]=[.1,-.2,.3]
    ref=np.eye(4);ref[:3,3]=[.2,.1,.05];b=rigid_bottle_pose(palm,ref)
    assert np.allclose(np.linalg.inv(palm)@b,ref)
    full,samples,cmd,peak=carry_path_samples([[.1,.2,0]],np.zeros(3),[0,1,2],{'bottle':ca,'other':cb})
    assert peak<=.001 and samples[0][1]==-1 and samples[-1][1:]==(0,1.)
    assert len(samples)>50
    for (q,_,_),(qn,_,_) in zip(samples,samples[1:]):assert max(ca@abs(qn-q),cb@abs(qn-q))<=.001+1e-12
    raw=np.zeros((2,6));raw[1,0]=.001
    q=np.zeros(8);q[7]=1.+3.7e-5
    full,_,cmd,_=carry_path_samples(raw,q,list(range(6)),{'bottle':np.ones(8)})
    limits=np.array([[-1.,1.]]*8)
    check=fma.joint_limit_audit(full,limits,['j'+str(i) for i in range(8)],cmd)
    assert check['passed'] and check['frozen_measured']['excess_rad']>1e-5
    full2,_,cmd2,_=carry_path_samples(full[1:],q,list(range(6)),{'bottle':np.ones(8)})
    assert not fma.joint_limit_audit(full2,limits,['j'+str(i) for i in range(8)],cmd2)['passed']
    def contact(link,impulse):
        return SimpleNamespace(bodies=[SimpleNamespace(entity=SimpleNamespace(name='bottle')),SimpleNamespace(entity=SimpleNamespace(name=link))],
            points=[SimpleNamespace(impulse=np.array(impulse),normal=np.array([0.,0.,1.]))])
    evidence=real_grasp_contact_evidence([contact('l_finger',[0,0,1e-6]),contact('l_near',[0,0,0]),
        contact('l_tangent',[.01,0,0]),contact('r_finger',[0,0,1e-6])],'bottle',{'l_finger','l_near','l_tangent'})
    assert evidence['allowed_exact_hand_links']==['l_finger']
    assert len(evidence['zero_impulse_near_contacts_not_excluded'])==2
    assert evidence['foreign_bottle_contacts_not_excluded'][0]['hand_link']=='r_finger'
    invalid=[lambda:carry_path_samples([[0,0,0]],np.zeros(3),[0,1,2],{'b':ca},vertex_step_m=.0011),
        lambda:carry_path_samples([[float('nan'),0,0]],np.zeros(3),[0,1,2],{'b':ca}),
        lambda:carry_path_samples([[0,0]],np.zeros(3),[0,1,2],{'b':ca}),
        lambda:carry_path_samples([[.5,.5,0]],np.zeros(3),[0,1,2],{'b':ca},max_samples=2),
        lambda:attached_motion_coefficients(tree,'pa',names,np.empty((0,3))),
        lambda:rigid_bottle_pose(np.zeros((4,4)),np.eye(4)),
        lambda:real_grasp_contact_evidence([], 'b', [], normal_impulse_threshold_Ns=-1)]
    for fn in invalid:
        try:fn()
        except ValueError:pass
        else:raise AssertionError('Invalid carry input accepted')
    return {'passed':True,'checks':['rigid_relation_roundtrip','attached_bottle_radius_includes_long_body',
        '500_random_FK_vertex_motion_bounds','common_ancestor_relative_distance_bound',
        'adaptive_robot_and_bottle_vertex_bound_1mm','actual_start_included','command6_vs_frozen36_limits',
        'normal_impulse_only_exact_grasp_pair_exclusions','zero_and_tangential_impulse_not_excluded',
        'other_arm_contact_not_excluded','7_invalid_inputs_rejected'],
        'max_observed_to_conservative_displacement_ratio':worst_ratio,'GPU':False,'hardware_io':False}


if __name__=='__main__':
    import json
    print(json.dumps(run_cpu_tests(),indent=2))


def run_fcl_cpu_tests():
    """Independent synthetic FCL objects, not a SAPIEN scene or GPU worker."""
    from mplib import Pose
    from mplib.collision_detection import fcl
    signs=np.array([[x,y,z] for x in (-.05,.05) for y in (-.05,.05) for z in (-.05,.05)])
    def rotation(q):
        c,s=math.cos(q),math.sin(q);m=np.eye(4);m[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
        m[:3,3]=[c,s,0];return m
    def box(name,pose,size):return fcl.FCLObject(name,Pose(pose),[fcl.CollisionObject(fcl.Box(size))],[Pose()])
    class Guard:
        def _assert_snapshot(self,task,before):
            if not np.array_equal(task.actual_q,before):raise AssertionError('Synthetic physical state changed')
    outputs=[]
    for collision_case in (True,False):
        a=object.__new__(CarryMeshPathAuditor);a.side='left';a.sim_joints=['l_joint'];a.initial_sim_q=np.array([0.]);a.old_model_q=a.initial_sim_q.copy()
        a.Pose=Pose;a.fcl=fcl;a.palm_name='palm';a.reference=np.eye(4);a.reference[0,3]=.5
        a.bottle_vertices=signs.copy();a.vertices={'palm':signs.copy(),'other':signs.copy()}
        a.objects={'palm':box('palm',rotation(0),[.1,.1,.1]),
                   'other':box('other',np.eye(4),[.1,.1,.1])}
        obstacle=np.eye(4);obstacle[:3,3]=[1.5/math.sqrt(2)]*2+[0.] if collision_case else [4.,4.,0.]
        a.objects['other']=box('other',obstacle,[.1,.1,.1])
        bottle_pose=rotation(0)@a.reference;table_pose=np.eye(4);table_pose[2,3]=-1.
        a.static={'bottle':box('bottle',bottle_pose,[.1,.1,.1]),'table':box('table',table_pose,[6.,6.,.1])}
        a.original_bottle_pose=Pose(a.static['bottle'].pose)
        a.static_boxes={'table':np.array([[-3.,-3.,-1.05],[3.,3.,-.95]])}
        a.motion_coefficients={'palm':np.array([1.1]),'other':np.array([0.])};a.bottle_coefficients=np.array([1.7])
        a.bottle_pair_coeff={'palm':np.array([0.]),'other':np.array([1.7])}
        a.self_pair_coeff={('palm','other'):np.array([1.1])};a.metadata={'synthetic_FCL_only':True}
        a.distance_request=fcl.DistanceRequest(enable_signed_distance=False);a.collision_request=fcl.CollisionRequest()
        a.art=SimpleNamespace(active_joints=[SimpleNamespace(limits=np.array([[-math.pi,math.pi]]))])
        a.task=SimpleNamespace(actual_q=np.array([0.]),robot=SimpleNamespace(left_arm_joints_name=['l_joint']))
        a.ppg=Guard();a.physical_before=a.task.actual_q.copy()
        class Model:
            def set_qpos(self,q,full=True):
                original=a.objects['palm'];a.objects['palm']=fcl.FCLObject('palm',Pose(rotation(float(q[0]))),original.shapes,original.shape_poses)
        a.model=Model();source=a.static['bottle']
        distance_before=float(fcl.distance(source,a.objects['other'],a.distance_request).min_distance)
        result=a.audit(np.array([[math.pi/2]]),include_samples=False,max_samples=4000)
        distance_after=float(fcl.distance(source,a.objects['other'],a.distance_request).min_distance)
        assert abs(distance_before-distance_after)<1e-12
        assert np.allclose(fma._pose_matrix(a.objects['palm'].pose),rotation(0),atol=1e-12)
        assert result['max_vertex_displacement_bound_m']<=.001
        if collision_case:
            assert not result['passed'] and any(e['category']=='bottle_robot' and e['link']=='other' for e in result['collisions_first_64'])
        else:assert result['passed'] and not result['collisions_first_64']
        outputs.append({'midpath_bottle_obstacle':collision_case,'audit_passed':result['passed'],
            'audited_samples':result['audited_samples'],'collision_categories':[e['category'] for e in result['collisions_first_64']],
            'auxiliary_pose_and_physical_state_restored':True})
    return {'passed':True,'cases':outputs,'SAPIEN_scene_created':False,'GPU':False,'hardware_io':False}
