"""v25: official-tempo five-finger profile; earlier attempts preserved; faster manufacturer-bounded force feedback without extra hold."""
from integrations.realman_rm65b_gen4.native_robot import RealManRobot
from types import MethodType
from envs.robot.robot import Robot
from envs._base_task import Base_Task

# Physically couple serial joints while retaining all vendor assets and active DOFs.
def install_hand_coupling(robot):
    entity=robot.left_entity; q=robot._measured_positions(); records=[]
    for side in ('left','right'):
        for name,m in robot._manifest['hands'][side]['mimic'].items():
            child=entity.find_joint_by_name(name); master=entity.find_joint_by_name(m['joint'])
            a=float(m.get('multiplier',1)); b=float(m.get('offset',0))
            assert child.parent_link==master.child_link
            assert abs(q[name]-a*q[m['joint']]-b)<1e-4
            child.set_drive_property(0,0,0,mode='force')
            entity.create_fixed_tendon([child.parent_link.parent,child.parent_link,child.child_link],
                [0,-a,1],[0,-1/a,1],rest_length=b,stiffness=100,damping=.02)
            records.append({'joint':name,'master':m['joint'],'multiplier':a,'stiffness':100,'follower_drive':0})
    robot.hand_coupling_audit=records
    return records

if 'original_initialize_hands' not in globals():original_initialize_hands=RealManRobot._initialize_hands

def initialize_coupled_hands(self):
    original_initialize_hands(self)
    # Set the stationary ready initial condition before the scene advances or objects spawn.
    # All task motions remain physical drive execution.
    q=np.asarray(self.left_entity.get_qpos()).copy()
    for side in ['left','right']:
        for n,v in zip(getattr(self,side+'_arm_joints_name'),ready[side+'_joint_rad']):q[self._joint_index[n]]=v
        initial=dict(zip(self._hand_joints,q))
        for n,v in self._hand_maps[side].from_sdk([1000]*5+[0],initial).items():q[self._joint_index[n]]=v
    self.left_entity.set_qpos(q);self.left_entity.set_qvel(np.zeros_like(q))
    for n,j in self._hand_joints.items():j.set_drive_target(float(q[self._joint_index[n]]));j.set_drive_velocity_target(0)
    install_hand_coupling(self)
    for h in self._hand_maps.values():
        for name in h.limits:self._hand_joints[name].set_friction(.001)
    from integrations.realman_rm65b_gen4.hand_coupling import set_hand_material
    self.hand_material_audit=set_hand_material(self.left_entity,self._manifest,(.5,.5,0))
    self.hand_friction_audit={n:float(self._hand_joints[n].friction) for n in self._hand_joints}
    for side in ['left','right']:self.set_hand_angle(side,[1000]*5+[0]);setattr(self,side+'_gripper_val',1.0)
    initial_q=self.left_entity.get_qpos().copy(); original_scene_step=task.scene.step; count=[0];init_trace=[]
    def initialization_step():
        before=self.left_entity.get_qpos().copy();original_scene_step();count[0]+=1;after=self.left_entity.get_qpos().copy()
        if count[0]<=5 or not np.isfinite(after).all():
            contacts=[]
            for c in task.scene.get_contacts():
                impulse=sum(float(np.linalg.norm(p.impulse)) for p in c.points)
                if impulse>1e-8:contacts.append({'names':[z.entity.name for z in c.bodies],'impulse':impulse})
            init_trace.append({'step':count[0],'before':before.tolist(),'after':after.tolist(),'contacts':contacts})
        if not np.isfinite(after).all():
            (BASE/'initialization_nan_trace_20261003.json').write_text(json.dumps(init_trace));raise RuntimeError(f'Initialization NaN step {count[0]}')
    task.scene.step=initialization_step

RealManRobot._initialize_hands=initialize_coupled_hands

def velocity_limited_hand_plan(self,side,now_val,target_val):
    h=self._hand_maps[side]; current=self._measured_positions()
    def positions(t):
        sdk=np.rint(self._synergy[side]['close']+t*(self._synergy[side]['open']-self._synergy[side]['close'])).astype(int).tolist()
        return h.from_sdk(sdk,current)
    q0=positions(now_val); q1=positions(target_val)
    span=max(abs(q1[n]-q0[n])/h.limits[n]['velocity'] for n in q0)
    # Smooth endpoint velocities; peak quintic derivative = 1.875.
    steps=max(200,int(math.ceil(1.875*span/.004))+2)
    t=np.linspace(0,1,steps); ramp=10*t**3-15*t**4+6*t**5
    vals=now_val+(target_val-now_val)*ramp
    increment=(target_val-now_val)/(steps-1)*1.9
    return {'num_step':steps,'per_step':increment,'result':vals}
RealManRobot.left_plan_grippers=lambda self,n,t:velocity_limited_hand_plan(self,'left',n,t)
RealManRobot.right_plan_grippers=lambda self,n,t:velocity_limited_hand_plan(self,'right',n,t)

def side_entry_grasp(self,actor,arm_tag,pre_dis=.1,target_dis=0,contact_point_id=None):
    M=actor.get_pose().to_transformation_matrix();px=M[:3,1].copy()
    if px[0]<0:px=-px
    px/=np.linalg.norm(px); angle=math.radians(grasp_setting['pitch_deg']);pz=np.array([0,math.cos(angle),-math.sin(angle)])
    pz-=px*np.dot(px,pz);pz/=np.linalg.norm(pz);py=np.cross(pz,px); R=np.stack([pz,py,-px],axis=1)
    center=actor.get_contact_point(0,'matrix')[:3,3]
    g=center-(grasp_setting['hand_z']-.06)*pz-grasp_setting['hand_y']*py-grasp_setting['hand_x']*px
    return [*(g-(pre_dis+.095-grasp_setting['hand_y'])*py),*mat2quat(R)],[*(g-target_dis*py),*mat2quat(R)]

import importlib, full_mesh_path_audit as fma
importlib.reload(fma)

def precise_path(self,target_pose,constraint_pose=None,**kw):
    current_R=t3d.quaternions.quat2mat(self.get_left_ee_pose()[-4:]);target_R=t3d.quaternions.quat2mat(target_pose[-4:])
    requested_turn_deg=float(np.degrees(np.arccos(np.clip((np.trace(current_R.T@target_R)-1)/2,-1,1))))
    if dense_calls[0]>=3:
        import full_native_carry_planner_v21 as ncp;importlib.reload(ncp)
        planning_reference=np.asarray(grip_measure(task)['relative_bottle_palm'],dtype=float)
        res=ncp.plan_carry_to_task_pose(task,target_pose,'left',palm_to_bottle=planning_reference,time_step=.004,max_joint_velocity=1.,max_joint_acceleration=1.,min_duration_s=0.,planning_time_s=10,ik_attempts=24,support_departure=(dense_calls[0]==3))
        if res['status']!='Success':
            report.setdefault('full_native_planning_failures',[]).append(res);write();return res
        np.savez(OUT/f'planned_carry_path_{dense_calls[0]}.npz',position=res['position'],velocity=res['velocity'],planning_reference=planning_reference)
        report.setdefault('segment_planning_references',{})[str(dense_calls[0])]=planning_reference.tolist()
        (OUT/f'carry_mesh_path_{dense_calls[0]}.json').write_text(json.dumps(res.get('audit'),default=str,indent=2))
        (OUT/f'carry_planning_{dense_calls[0]}.json').write_text(json.dumps(res.get('planning_metadata'),default=str,indent=2))
        model=fma.FullMeshPathAuditor(task);palms=[];q=model.initial_sim_q.copy();indices=[model.sim_joints.index(n) for n in self.left_arm_joints_name]
        try:
            for arm_q in res['position']:
                q[indices]=arm_q;model.model.set_qpos(q,full=True);palms.append(fma._pose_matrix(model.objects['l_inspire_base_link'].pose))
        finally:model.model.set_qpos(model.old_model_q,full=True)
        from planned_carry_wrench import planned_carry_wrench,preflight_contact_lp
        component=next(c for c in task.bottle.actor.get_components() if isinstance(c,sapien.physx.PhysxRigidDynamicComponent))
        schedule=planned_carry_wrench(palms,planning_reference,cmass_local_matrix=component.cmass_local_pose.to_transformation_matrix(),mass_kg=float(component.mass),inertia_kg_m2=component.inertia,dt=.004,endpoint_mode='hold')
        raw=grip_measure(task)['contacts'];inputs=[]
        for c in raw:
            token=next((z for z in ['middle','index','thumb'] if z in c['link']),None)
            if token:inputs.append({'channel':token,'point_world_m':c['point'],'force_on_bottle_normal_world':(-np.asarray(c['normal_bottle_to_hand'])).tolist()})
        check=preflight_contact_lp(schedule,inputs,initial_bottle_world_matrix=task.bottle.get_pose().to_transformation_matrix(),sample_stride=1,previous_allocation_N=feedback['force_target_N'],lp_options={'min_main_force_N':float(grasp_setting.get('lp_min_main_force_N',.1))})
        check['lp_options']={'min_main_force_N':float(grasp_setting.get('lp_min_main_force_N',.1))}
        (OUT/f'carry_wrench_preflight_{dense_calls[0]}.json').write_text(json.dumps(check,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x),indent=2))
        print('CARRY_WRENCH_PREFLIGHT',dense_calls[0],{k:v for k,v in check.items() if k not in ['checked_indices','added_peak_indices','failed_cases']},flush=True)
        if not check['passed']:return {'status':'Fail','expected_wrench_preflight_failed':True}
        np.savez(OUT/f'planned_wrench_{dense_calls[0]}.npz',force_target_world_N=schedule['force_target_world_N'],torque_target_world_Nm=schedule['torque_target_world_Nm'])
        feedback['wrench_schedule']=schedule;feedback['wrench_index']=0;feedback['last_lp_phase']=None
        return res
    if constraint_pose is not None:
        palm_pose=self._trans_from_gripper_to_endlink(target_pose,arm_tag='left')
        res=self.left_mplib_planner.planner.plan_screw(goal_pose=palm_pose,current_qpos=self.left_entity.get_qpos(),time_step=.004,qpos_step=.005)
        if res['status']=='Success':
            print('OFFICIAL_SCREW_TIMING',len(res['position']),flush=True)
    else:
        if dense_calls[0]==0 and grasp_setting['pitch_deg']==10 and grasp_setting['hand_x']==0:
            data=np.load(BASE/'coupled_side_entry_v4_B10_20261003_seed7/planned_path_1.npz')
            res={'status':'Success','position':data['position'],'velocity':data['velocity']}
        else:res=Robot.left_plan_path(self,target_pose,constraint_pose=constraint_pose,**kw)
    if res['status']=='Success':
        a=fma.audit_path(task,res,'left',min_bottle_clearance_m=.0005,min_table_clearance_m=0,min_self_clearance_m=0)
        (OUT/f'mesh_path_{dense_calls[0]}.json').write_text(json.dumps(a,default=str,indent=2));np.savez(OUT/f'planned_path_{dense_calls[0]}.npz',position=res['position'],velocity=res['velocity'])
        print('PATH_CLEARANCE',dense_calls[0],a['passed'],a['continuous_gap_lower_bound'],flush=True)
        if not a['passed']:res={'status':'Fail','full_mesh_audit_failed':True}
    return res
def realman_place_pose(self,actor,arm_tag,target_pose,**kw):
    if arm_tag=='left':
        live=self.robot.get_left_ee_pose();axis=t3d.quaternions.quat2mat(live[-4:])[:,2]
        kw.update(constrain='align',actor_axis=axis.tolist(),actor_axis_type='world',align_axis=[-1.,0.,0.])
    return Base_Task.get_place_pose(self,actor,arm_tag,target_pose,**kw)
def realman_lift_displacement(self,arm_tag,x=0.,y=0.,z=0.,quat=None,move_axis='world'):
    selected='world' if grasp_setting.get('lift_axis')=='world' and arm_tag=='left' and move_axis=='arm' and x==0 and y==0 and z==.1 else move_axis
    return Base_Task.move_by_displacement(self,arm_tag,x=x,y=y,z=z,quat=quat,move_axis=selected)
exec((BASE/'contact_capture_control_v22.py').read_text(),globals())
strict_hold=[False];hold_samples=[]

def grip_measure(self):
    groups={'thumb':{'normal_N':0.,'normal_vector':np.zeros(3)},'fingers':{'normal_N':0.,'normal_vector':np.zeros(3)}}
    table_N=0.;raw=[]
    for c in self.scene.get_contacts():
        names=[z.entity.name for z in c.bodies]
        if '001_bottle' not in names:continue
        other=names[1] if names[0]=='001_bottle' else names[0]
        for point in c.points:
            normal=np.asarray(point.normal)*(-1 if names[0]=='001_bottle' else 1)
            impulse=np.asarray(point.impulse);force=abs(float(np.dot(impulse,normal)))/self.scene.get_timestep()
            if 'table' in other.lower():table_N+=force
            if not other.startswith('l_') or force<1e-6:continue
            group='thumb' if 'thumb' in other else ('fingers' if any(z in other for z in ['index','middle','ring','little']) else None)
            if group:
                groups[group]['normal_N']+=force;groups[group]['normal_vector']+=force*normal
                raw.append({'link':other,'normal_force_N':force,'normal_bottle_to_hand':normal.tolist(),'impulse_Ns':impulse.tolist(),'point':point.position.tolist()})
    a=groups['thumb']['normal_vector'];b=groups['fingers']['normal_vector'];dot=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>1e-12 else 1.
    palm=self.robot.left_entity.find_link_by_name('l_inspire_base_link').entity.pose.to_transformation_matrix()
    bottle=self.bottle.get_pose().to_transformation_matrix();relative=np.linalg.inv(palm)@bottle
    return {'thumb_normal_N':groups['thumb']['normal_N'],'fingers_normal_N':groups['fingers']['normal_N'],'opposing_normals_dot':dot,'table_normal_N':table_N,'relative_bottle_palm':relative.tolist(),'contacts':raw}

def configure_support_departure(self):
    """Fresh actual native signed-height monitor. No physics steps or masks changed."""
    import carry_mesh_path_audit as cma
    import preflight_place_goals as ppg
    from sapien.physx import PhysxRigidBaseComponent
    before=ppg._snapshot(self)
    a=cma.CarryMeshPathAuditor(self,'left',palm_to_bottle=np.asarray(grip_measure(self)['relative_bottle_palm']))
    try:
        actors=[x for x in self.scene.get_all_actors() if x.name=='table']
        if len(actors)!=1:raise RuntimeError('Expected one native support table')
        comp=actors[0].find_component_by_type(PhysxRigidBaseComponent)
        signs=np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],dtype=float)
        boxes=[]
        for shape in comp.collision_shapes:
            if not hasattr(shape,'half_size'):raise RuntimeError('Original table collision shapes must be boxes')
            W=cma._matrix(comp.entity.pose)@cma._matrix(shape.local_pose)
            if not np.allclose(np.abs(W[:3,:3]),np.eye(3),atol=1e-7):raise RuntimeError('Support table boxes must be axis-aligned')
            v=(signs*np.asarray(shape.half_size))@W[:3,:3].T+W[:3,3]
            boxes.append(np.stack([v.min(0),v.max(0)]))
        bounds=max(boxes,key=lambda q:float(q[1,2])).copy();top=float(bounds[1,2])
        if abs(top-.74)>1e-6 or abs(top-float(a.static_boxes['table'][1,2]))>1e-6:raise RuntimeError('Incorrect original tabletop world height')
        if not np.allclose(bounds[1,:2]-bounds[0,:2],[1.2,.7],atol=1e-6):raise RuntimeError('Incorrect original tabletop support footprint')
        vertices=a.bottle_vertices.copy()
        M=self.bottle.get_pose().to_transformation_matrix();world=vertices@M[:3,:3].T+M[:3,3]
        baseline=float(world[:,2].min()-top)
        if baseline < -.0005 or baseline>=.001:raise RuntimeError('Actual first-lift support baseline outside allowed support phase')
        support_monitor.clear();support_monitor.update(active=True,vertices=vertices,table_bounds=bounds,table_top_z=top,
            baseline_signed_gap_m=baseline,floor_signed_gap_m=baseline-.000020,minimum_signed_gap_m=baseline,
            maximum_additional_depth_m=0.,release_step=None,release_gap_m=.001,checked_steps=0,
            physical_table_collisions_enabled=True,configured_at_step=steps[0])
    finally:
        a.model.set_qpos(a.old_model_q,full=True);ppg._assert_snapshot(self,before)
    return {k:v for k,v in support_monitor.items() if k not in ('vertices','table_bounds')}

def check_support_departure(self):
    """Monitor real executed vertices through support exit and all later carries."""
    if not support_monitor.get('active'):return None
    M=self.bottle.get_pose().to_transformation_matrix();vertices=support_monitor['vertices']@M[:3,:3].T+M[:3,3]
    gap=float(vertices[:,2].min()-support_monitor['table_top_z']);support_monitor['checked_steps']+=1
    support_monitor['minimum_signed_gap_m']=min(support_monitor['minimum_signed_gap_m'],gap)
    support_monitor['maximum_additional_depth_m']=max(support_monitor['maximum_additional_depth_m'],support_monitor['baseline_signed_gap_m']-gap)
    released=support_monitor['release_step'] is not None
    violation=None
    if not np.isfinite(vertices).all():violation='Nonfinite actual support vertices'
    elif np.any(vertices[:,:2]<support_monitor['table_bounds'][0,:2]) or np.any(vertices[:,:2]>support_monitor['table_bounds'][1,:2]):violation='Actual bottle leaves table footprint'
    elif gap < (0. if released else support_monitor['floor_signed_gap_m']):violation='Actual support depth or released clearance violated'
    if not released and gap>=support_monitor['release_gap_m']:support_monitor['release_step']=steps[0]
    summary={k:v for k,v in support_monitor.items() if k not in ('vertices','table_bounds')};summary.update(actual_signed_gap_m=gap,violation=violation)
    if violation:raise RuntimeError(violation+': '+str(summary))
    return summary

def strict_official_episode(self,phase):
    before=bottle_state(self)
    info=self.play_once();immediate=bottle_state(self);hold_samples.clear()
    if self.plan_success:
        strict_hold[0]=True
        try:hold_current_targets(self,1.)
        finally:strict_hold[0]=False
    after=bottle_state(self);drift=float(np.linalg.norm(np.asarray(after['functional_point0'][:3])-np.asarray(immediate['functional_point0'][:3])));accepted=False;max_position=None;max_angle=None;paired_fraction=0.
    if hold_samples:
        ref=np.asarray(hold_samples[0]['relative_bottle_palm']);pos=[np.linalg.norm(np.asarray(x['relative_bottle_palm'])[:3,3]-ref[:3,3]) for x in hold_samples]
        angles=[np.degrees(np.arccos(np.clip((np.trace(ref[:3,:3].T@np.asarray(x['relative_bottle_palm'])[:3,:3])-1)/2,-1,1))) for x in hold_samples]
        max_position=float(max(pos));max_angle=float(max(angles))
        paired_fraction=sum(x['thumb_normal_N']>.05 and x['fingers_normal_N']>.05 and x['opposing_normals_dot']<-.3 and x['table_normal_N']<1e-5 for x in hold_samples)/len(hold_samples)
        accepted=bool(self.plan_success and immediate['official_success'] and after['official_success'] and before['selected_hand_active_points']==0
            and immediate['upright_angle_deg']<5 and after['upright_angle_deg']<5 and drift<.002 and max_position<.002 and max_angle<2 and paired_fraction>=.99 and len(hold_samples)>=250)
    audit={'phase':phase,'before':before,'immediate':immediate,'after_hold':after,'accepted':accepted,'hold_validation_s':1.,'hold_drift_m':drift,'hold_samples':len(hold_samples),'paired_contact_fraction':paired_fraction,'max_hand_relative_translation_drift_m':max_position,'max_hand_relative_rotation_drift_deg':max_angle,'requirements':'unmodified official task pass; upright<5deg; paired true normal forces>.05N; opposing dot<-.3; table normal≈0; relative drift<2mm/2deg for1s'}
    report.setdefault('bottle_attempts',[]).append(audit);info['realman_bottle_audit']=audit;write();return info,accepted
run_official_episode=strict_official_episode
for label,geometry in [('B085',{'lift_axis':'world','pitch_deg':10,'hand_x':.015,'hand_y':.085,'hand_z':.165,'close_sdk':[650,670,610,610,887,0],'ready_near_sdk':[650,670,610,610,887,0],'finger_force_allocation_N':[0,0,.12,.12,.24],'lp_min_main_force_N':1.15,'feedback_joint_rate_cap_rad_s':.16})]:
    OUT=BASE/f'coupled_side_entry_v25_{label}_20261003_seed7';OUT.mkdir(exist_ok=False)
    grasp_setting=dict(geometry,open_sdk=[1000]*5+[0])
    runtime_paths=[BASE/'try_coupled_side_entry_v25.py',BASE/'contact_capture_control_v22.py',BASE/'full_native_carry_planner_v21.py',BASE/'support_departure_certificate_v20.py']
    report={'runtime_source_hashes':{str(p):sha(p) for p in runtime_paths},'timing_profile':'Official arm TOPP; vendor-speed-limited one-stage five-finger close; official 50pct gripper padding; no additional pre-lift hold','result':'running','task':'adjust_bottle','seed':7,'attempts':[],'grasp_setting':grasp_setting,'hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected}}
    for side in ['left','right']:
        args[side+'_embodiment_config'].update(hand_synergy={'open':grasp_setting['open_sdk'],'close':grasp_setting['close_sdk']},hand_joint_stiffness=5,hand_joint_damping=.1)
        task.robot._hand_config[side].update(args[side+'_embodiment_config'])
    for method_name in ['set_gripper','set_arm_joints','left_plan_grippers','right_plan_grippers']:
        if method_name in task.robot.__dict__:del task.robot.__dict__[method_name]
    args.update(seed=7,save_path=str(OUT),save_data=False,save_freq=None,need_plan=True)
    for name in ['move','take_dense_action','choose_grasp_pose']:
        if name in task.__dict__:del task.__dict__[name]
    task.close_env();task.setup_demo(**args)
    for side in ['left','right']:
        for suffix in ['plan_path','plan_multi_path']:
            name=side+'_'+suffix
            setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
    task.choose_grasp_pose=MethodType(side_entry_grasp,task)
    task.get_place_pose=MethodType(realman_place_pose,task)
    task.move_by_displacement=MethodType(realman_lift_displacement,task)
    feedback,contact_update=install_contact_capture(task,grasp_setting)
    task.robot.left_plan_grippers=MethodType(feedback['plan'],task.robot)
    steps=[0]; dense_calls=[0]; traces=[]; pre_retreat_samples=[]; support_monitor={}; original_step=task.scene.step
    actual_self_auditor=fma.FullMeshPathAuditor(task);actual_self_allowed=actual_self_auditor.allowed;actual_robot_link_names=set(actual_self_auditor.links)
    report['full_qpos_joint_names']=actual_self_auditor.sim_joints
    hand_ids=[task.robot._joint_index[n] for h in task.robot._hand_maps.values() for n in h.limits]
    dynamic_hand_audit={'max_actual_fd_rad_s':0.,'max_engine_qvel_rad_s':0.,'max_mimic_residual_rad':0.};carry_reference=[None];carry_audit={'max_translation_drift_m':0.,'max_rotation_drift_deg':0.}
    initial_fp=np.asarray(task.bottle.get_functional_point(0)[:3]);initial_R=task.bottle.get_pose().to_transformation_matrix()[:3,:3]
    stage={'name':'ready','gripper':False}
    def trace_step():
        q_before=task.robot.left_entity.get_qpos().copy();contact_update();original_step();steps[0]+=1
        unexpected_self_contacts=[]
        for contact in task.scene.get_contacts():
            names=[b.entity.name for b in contact.bodies]
            if all(n in actual_robot_link_names for n in names) and frozenset(names) not in actual_self_allowed:
                normal_impulse=sum(abs(float(np.dot(p.impulse,p.normal))) for p in contact.points)
                if normal_impulse>1e-7:unexpected_self_contacts.append({'names':names,'normal_impulse_Ns':normal_impulse,'points':[{'position':p.position.tolist(),'normal':p.normal.tolist(),'impulse_Ns':p.impulse.tolist(),'separation_m':getattr(p,'separation',None)} for p in contact.points]})
        fd=(task.robot.left_entity.get_qpos()-q_before)/task.scene.get_timestep();engine_v=task.robot.left_entity.get_qvel()
        dynamic_hand_audit['max_actual_fd_rad_s']=max(dynamic_hand_audit['max_actual_fd_rad_s'],float(np.max(np.abs(fd[hand_ids]))))
        dynamic_hand_audit['max_engine_qvel_rad_s']=max(dynamic_hand_audit['max_engine_qvel_rad_s'],float(np.max(np.abs(engine_v[hand_ids]))))
        q=task.robot._measured_positions();state=bottle_state(task);
        if not np.isfinite(task.robot.left_entity.get_qpos()).all():raise RuntimeError('Nonfinite articulation')
        if max(abs(fd[task.robot._joint_index[n]])/lim['velocity'] for h in task.robot._hand_maps.values() for n,lim in h.limits.items())>1.05:raise RuntimeError('Actual hand FD speed limit failed')
        if stage['name']=='after_grasp' and carry_reference[0] is not None:
            actual=np.asarray(grip_measure(task)['relative_bottle_palm']);ref=carry_reference[0]
            slip=float(np.linalg.norm(actual[:3,3]-ref[:3,3]));tilt=float(np.degrees(np.arccos(np.clip((np.trace(ref[:3,:3].T@actual[:3,:3])-1)/2,-1,1))))
            carry_audit['max_translation_drift_m']=max(carry_audit['max_translation_drift_m'],slip)
            carry_audit['max_rotation_drift_deg']=max(carry_audit['max_rotation_drift_deg'],tilt)
            if slip>.002 or tilt>2:raise RuntimeError(f'Carry slip guard: translation={slip}, rotation_deg={tilt}')
        if strict_hold[0]:hold_samples.append(grip_measure(task))
        if stage['name'] in ['close','close_hold']:pre_retreat_samples.append(grip_measure(task))
        fp=np.asarray(state['functional_point0'][:3]);rot=np.asarray(state['pose'])[:3,:3]
        delta=float(np.linalg.norm(fp-initial_fp));angle=float(np.degrees(np.arccos(np.clip((np.trace(initial_R.T@rot)-1)/2,-1,1))))
        contacts=[c for c in state['hand_contacts'] if c['active_point_count']>0]
        forces=[]
        for c in task.scene.get_contacts():
            names=[z.entity.name for z in c.bodies]
            if '001_bottle' not in names:continue
            other=names[1] if names[0]=='001_bottle' else names[0]
            if not other.startswith('l_'):continue
            for point in c.points:
                impulse=float(np.linalg.norm(point.impulse))
                if impulse<1e-8:continue
                normal=np.asarray(point.normal)*(-1 if names[0]=='001_bottle' else 1)
                forces.append({'link':other,'force_N':impulse/task.scene.get_timestep(),'normal_force_N':abs(float(np.dot(point.impulse,point.normal)))/task.scene.get_timestep(),'impulse_Ns':point.impulse.tolist(),'normal_bottle_to_hand':normal.tolist(),'point':point.position.tolist()})
        residual=max(abs(q[n]-float(m['multiplier'])*q[m['joint']]) for h in task.robot._manifest['hands'].values() for n,m in h['mimic'].items())
        dynamic_hand_audit['max_mimic_residual_rad']=max(dynamic_hand_audit['max_mimic_residual_rad'],residual)
        if True:
            traces.append({'step':steps[0],'phase':dict(stage),'fp_delta_m':delta,'rotation_delta_deg':angle,'bottle':state,'actual_self_contacts':unexpected_self_contacts,'full_qpos_actual_rad':task.robot.left_entity.get_qpos().tolist(),'arm_qpos':{s:[q[n] for n in getattr(task.robot,s+'_arm_joints_name')] for s in ['left','right']},'sdk_measured':{s:task.robot.get_hand_sdk(s) for s in ['left','right']},'sdk_target':{s:task.robot.get_hand_sdk(s,False) for s in ['left','right']},'max_mimic_residual_rad':residual,'contact_forces':forces,'capture_latched':list(feedback['latched']),'capture_filtered_N':feedback['filtered_N'].tolist(),'capture_actual_pair_streak':feedback['actual_pair_streak'],'hand_actual_fd_max_rad_s':float(np.max(np.abs(fd[hand_ids]))),'hand_engine_qvel_max_rad_s':float(np.max(np.abs(engine_v[hand_ids])))})
        support_state=check_support_departure(task)
        if support_state is not None:traces[-1]['support_departure']=support_state
        if unexpected_self_contacts:raise RuntimeError('Actual full-body self contact: '+str(unexpected_self_contacts))
        if (stage['name']=='approach' and (delta>.001 or angle>1 or contacts)) or (stage['name'] in ['close','close_hold'] and (delta>.002 or angle>2)):
            raise RuntimeError(f'No-push approach guard: delta={delta}, angle={angle}, contacts={contacts}')
    task.scene.step=trace_step
    original_dense=Base_Task.take_dense_action
    def dense_trace(self,seq,**kw):
        dense_calls[0]+=1;stage['gripper']=seq.get('left_gripper') is not None
        stage['name']='close' if stage['gripper'] else ('approach' if dense_calls[0]<=2 else 'after_grasp')
        stage['dense_call']=dense_calls[0]
        if stage['gripper']:pre_retreat_samples.clear()
        if dense_calls[0]==4:report['support_departure_initial']=configure_support_departure(self)
        first_step=steps[0]
        result=original_dense(self,seq,**kw)
        report.setdefault('action_timing',[]).append({'dense_call':dense_calls[0],'phase':stage['name'],'start_step':first_step,'end_step':steps[0],'duration_s':(steps[0]-first_step)*self.scene.get_timestep()})
        if stage['gripper']:
            stage['name']='close_hold';stage['gripper']=False
            samples=pre_retreat_samples[-50:]
            if len(samples)!=50:raise RuntimeError('Missing last 50 fresh close samples')
            reference=np.asarray(samples[0]['relative_bottle_palm'])
            displacement=max(np.linalg.norm(np.asarray(z['relative_bottle_palm'])[:3,3]-reference[:3,3]) for z in samples)
            rotation=max(np.degrees(np.arccos(np.clip((np.trace(reference[:3,:3].T@np.asarray(z['relative_bottle_palm'])[:3,:3])-1)/2,-1,1))) for z in samples)
            actual=grip_measure(self)
            last_lp=feedback['lp_history'][-1]['allocation'] if feedback['lp_history'] else {'usable':False}
            gate=bool(last_lp['usable'] and feedback['lp_failure_steps']==0 and all(feedback['normal_N'][i]>.05 for i in [2,3,4]) and all(sum(c['normal_force_N'] for c in z['contacts'] if token in c['link'])>.05 for z in samples for token in ['middle','index','thumb']) and all(z['thumb_normal_N']>.05 and z['fingers_normal_N']>.05 and z['opposing_normals_dot']<-.3 for z in samples) and feedback['actual_pair_streak']>=50 and actual['opposing_normals_dot']<-.3 and displacement<.002 and rotation<2)
            report['pre_retreat_grip']={'passed':gate,'reference_step':steps[0],'actual_pair_streak_steps':feedback['actual_pair_streak'],'relative_drift_m':float(displacement),'relative_rotation_deg':float(rotation),'actual':actual}
            print('PRE_RETREAT_GRIP',report['pre_retreat_grip'],flush=True)
            if not gate:raise RuntimeError('No stable physical paired grip before official retreat')
            carry_reference[0]=np.asarray(actual['relative_bottle_palm'])
            snapshot(self,'closed_grip')
            print('CLOSE_HOLD',json.dumps(bottle_state(self)),flush=True)
        return result
    task.take_dense_action=MethodType(dense_trace,task)
    task.robot.left_plan_path=MethodType(precise_path,task.robot)
    phase=instrument(task,'plan')
    try:
        info,accepted=run_official_episode(task,'plan');report['final_bottle']=bottle_state(task)
        legal,excess=task.planned_joints_legal();report.update(result='physics_success' if accepted and legal else 'physics_failed',accepted=accepted and all(sha(p)==report['runtime_source_hashes'][str(p)] for p in runtime_paths),plan_success=bool(task.plan_success),joint_limit_excess_rad=float(excess))
        if accepted and legal:
            accepted_grasp_setting=copy.deepcopy(grasp_setting);accepted_seed=7;task.save_traj_data(0)
    except Exception:
        report.update(result='exception',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    finally:
        task.scene.step=original_step
        report['support_departure_actual']={k:v for k,v in support_monitor.items() if k not in ('vertices','table_bounds')}
        report.update(runtime_source_hashes_final={str(p):sha(p) for p in runtime_paths},runtime_sources_unchanged=all(sha(p)==report['runtime_source_hashes'][str(p)] for p in runtime_paths),carry_audit=carry_audit,force_feedback_simulation_parameters=feedback.get('feedback_simulation_parameters'),adaptive_force_allocations=feedback['lp_history'],dynamic_hand_audit=dynamic_hand_audit,continuous_hand_profile=feedback['profile_segments'],runtime_scope='hand-only friction0.001 and material0.5/0.5/0 simulation parameters; no real-hand calibration; continuous joint drive and PhysX normal contact feedback',hand_material=task.robot.hand_material_audit,hand_friction=task.robot.hand_friction_audit,tendons=task.robot.hand_coupling_audit,canonical_sources_unchanged=all(sha(p)==report['source_hashes'][str(p)] for p in protected));write()
        (OUT/'physics_trace.json').write_text(json.dumps(traces,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
        (OUT/'final_hold_contact_trace.json').write_text(json.dumps(hold_samples));print('COUPLED_SIDE_RESULT',label,report['result'],flush=True)
    if 'accepted_grasp_setting' in globals():break