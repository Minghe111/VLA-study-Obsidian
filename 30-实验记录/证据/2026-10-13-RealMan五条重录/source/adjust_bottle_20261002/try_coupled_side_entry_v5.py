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
    return [*(g-pre_dis*py),*mat2quat(R)],[*(g-target_dis*py),*mat2quat(R)]

import importlib, full_mesh_path_audit as fma
importlib.reload(fma)

def precise_path(self,target_pose,constraint_pose=None,**kw):
    if constraint_pose is not None and dense_calls[0]<4:
        palm_pose=self._trans_from_gripper_to_endlink(target_pose,arm_tag='left')
        res=self.left_mplib_planner.plan_screw(self.left_entity.get_qpos(),palm_pose,arms_tag='left',log=True)
        if res['status']=='Success':
            old=np.asarray(res['position']);peak=np.max(np.abs(np.diff(old,axis=0)))/.004
            n=max(len(old),int(math.ceil(len(old)*max(1,peak/.25)*1.875)))
            t=np.linspace(0,1,n);u=10*t**3-15*t**4+6*t**5
            pos=np.stack([np.interp(u,np.linspace(0,1,len(old)),old[:,j]) for j in range(6)],axis=1)
            vel=np.gradient(pos,.004,axis=0);vel[0]=0;vel[-1]=0
            res.update(position=pos,velocity=vel)
            print('PRECISE_CARTESIAN_PATH',len(pos),'max_vel',float(np.max(np.abs(vel))),flush=True)
    else:
        if dense_calls[0]==1 and grasp_setting['pitch_deg']==10:
            data=np.load(BASE/'coupled_side_entry_v4_B10_20261003_seed7/planned_path_1.npz')
            res={'status':'Success','position':data['position'],'velocity':data['velocity']}
        else:res=Robot.left_plan_path(self,target_pose,constraint_pose=constraint_pose,**kw)
    if res['status']=='Success' and dense_calls[0]<4:
        a=fma.audit_path(task,res,'left',min_bottle_clearance_m=.0005,min_table_clearance_m=0,min_self_clearance_m=0)
        (OUT/f'mesh_path_{dense_calls[0]}.json').write_text(json.dumps(a,default=str,indent=2));np.savez(OUT/f'planned_path_{dense_calls[0]}.npz',position=res['position'],velocity=res['velocity'])
        print('PATH_CLEARANCE',dense_calls[0],a['passed'],a['continuous_gap_lower_bound'],a['collisions_first_64'][:4],flush=True)
        if not a['passed']:res={'status':'Fail','full_mesh_audit_failed':True}
    return res
for label,geometry in [('B10',{'pitch_deg':10,'hand_x':0,'hand_y':.095,'hand_z':.165,'close_sdk':[515,590,590,590,915,0]})]:
    OUT=BASE/f'coupled_side_entry_v5_{label}_20261003_seed7';OUT.mkdir(exist_ok=False)
    grasp_setting=dict(geometry,open_sdk=[1000]*5+[0])
    report={'result':'running','task':'adjust_bottle','seed':7,'attempts':[],'grasp_setting':grasp_setting,'hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected}}
    for side in ['left','right']:
        args[side+'_embodiment_config'].update(hand_synergy={'open':grasp_setting['open_sdk'],'close':grasp_setting['close_sdk']},hand_joint_stiffness=5,hand_joint_damping=.1)
        task.robot._hand_config[side].update(args[side+'_embodiment_config'])
    args.update(seed=7,save_path=str(OUT),save_data=False,save_freq=None,need_plan=True)
    for name in ['move','take_dense_action','choose_grasp_pose']:
        if name in task.__dict__:del task.__dict__[name]
    task.close_env();task.setup_demo(**args)
    for side in ['left','right']:
        for suffix in ['plan_path','plan_multi_path']:
            name=side+'_'+suffix
            setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
    task.choose_grasp_pose=MethodType(side_entry_grasp,task)
    steps=[0]; dense_calls=[0]; traces=[]; original_step=task.scene.step
    initial_fp=np.asarray(task.bottle.get_functional_point(0)[:3]);initial_R=task.bottle.get_pose().to_transformation_matrix()[:3,:3]
    stage={'name':'ready','gripper':False}
    def trace_step():
        original_step();steps[0]+=1
        q=task.robot._measured_positions();state=bottle_state(task);fp=np.asarray(state['functional_point0'][:3]);rot=np.asarray(state['pose'])[:3,:3]
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
                normal=np.asarray(point.normal)*(1 if names[0]=='001_bottle' else -1)
                forces.append({'link':other,'force_N':impulse/task.scene.get_timestep(),'normal_bottle_to_hand':normal.tolist(),'point':point.position.tolist()})
        residual=max(abs(q[n]-float(m['multiplier'])*q[m['joint']]) for h in task.robot._manifest['hands'].values() for n,m in h['mimic'].items())
        if steps[0]%10==0 or contacts:
            traces.append({'step':steps[0],'phase':dict(stage),'fp_delta_m':delta,'rotation_delta_deg':angle,'bottle':state,'arm_qpos':{s:[q[n] for n in getattr(task.robot,s+'_arm_joints_name')] for s in ['left','right']},'sdk_measured':{s:task.robot.get_hand_sdk(s) for s in ['left','right']},'sdk_target':{s:task.robot.get_hand_sdk(s,False) for s in ['left','right']},'max_mimic_residual_rad':residual,'contact_forces':forces})
        if (stage['name']=='approach' and (delta>.001 or angle>1 or contacts)) or (stage['name'] in ['close','close_hold'] and (delta>.002 or angle>2)):
            raise RuntimeError(f'No-push approach guard: delta={delta}, angle={angle}, contacts={contacts}')
    task.scene.step=trace_step
    original_dense=Base_Task.take_dense_action
    def dense_trace(self,seq,**kw):
        dense_calls[0]+=1;stage['gripper']=seq.get('left_gripper') is not None
        stage['name']='close' if stage['gripper'] else ('approach' if dense_calls[0]<=3 else 'after_grasp')
        stage['dense_call']=dense_calls[0]
        result=original_dense(self,seq,**kw)
        if stage['gripper']:
            stage['name']='close_hold';stage['gripper']=False
            controls={s+'_arm':{'position':np.tile(np.asarray([j.get_drive_target()[0] for j in getattr(self.robot,s+'_arm_joints')]),(125,1)),'velocity':np.zeros((125,6))} for s in ['left','right']}
            controls.update(left_gripper=None,right_gripper=None)
            original_dense(self,controls,save_freq=None)
            print('CLOSE_HOLD',json.dumps(bottle_state(self)),flush=True)
        return result
    task.take_dense_action=MethodType(dense_trace,task)
    task.robot.left_plan_path=MethodType(precise_path,task.robot)
    phase=instrument(task,'plan')
    try:
        info,accepted=run_official_episode(task,'plan');report['final_bottle']=bottle_state(task)
        legal,excess=task.planned_joints_legal();report.update(result='physics_success' if accepted and legal else 'physics_failed',accepted=accepted,plan_success=bool(task.plan_success),joint_limit_excess_rad=float(excess))
        if accepted and legal:
            accepted_grasp_setting=copy.deepcopy(grasp_setting);accepted_seed=7;task.save_traj_data(0)
    except Exception:
        report.update(result='exception',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    finally:
        task.scene.step=original_step
        report.update(tendons=task.robot.hand_coupling_audit,canonical_sources_unchanged=all(sha(p)==report['source_hashes'][str(p)] for p in protected));write()
        (OUT/'physics_trace.json').write_text(json.dumps(traces,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
        print('COUPLED_SIDE_RESULT',label,report['result'],flush=True)
    if 'accepted_grasp_setting' in globals():break