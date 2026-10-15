"""v26: explicit v22 capture using numerically checked LP fallback solver.
The close profile, independent capture, force floor, limits and timing remain
v22. Only the explicit LP module binding changes; its feasible set is unchanged.
The gain/rate are simulation parameters, not calibrated hardware force control.
Target slew bounds do not certify actual qvel or position finite-difference speed.
No scene stepping, qpos teleportation, official/model changes or hardware IO.
"""
import numpy as np
from types import MethodType
from continuous_servo import ContinuousHandMapping, SDKWaypointProfile

def _feedback_rate_limits(mapping, configured_cap_rad_s):
    """Fold each follower speed through the full signed mimic derivative chain."""
    cap=float(configured_cap_rad_s)
    if isinstance(configured_cap_rad_s,(bool,np.bool_)) or not np.isfinite(cap) or cap<=0:
        raise ValueError('Feedback rate cap must be positive finite rad/s')
    root_and_derivative={n:(n,1.) for n in mapping.channels}
    vendor={n:float(mapping.limits[n]['velocity']) for n in mapping.channels}
    for n,parent,multiplier,offset in mapping.mimic_order:
        root,derivative=root_and_derivative[parent]
        derivative*=multiplier
        root_and_derivative[n]=(root,derivative)
        if abs(derivative)>0:
            vendor[root]=min(vendor[root],float(mapping.limits[n]['velocity'])/abs(derivative))
    return vendor,{n:min(cap,vendor[n]) for n in mapping.channels}

def install_contact_capture(task,setting):
    robot=task.robot; h=robot._hand_maps['left'];dt=task.scene.get_timestep()
    mapping=ContinuousHandMapping(robot._manifest['hands']['left'])
    feedback_gain=float(setting.get('feedback_force_gain',.3))
    if isinstance(setting.get('feedback_force_gain',.3),(bool,np.bool_)) or not np.isfinite(feedback_gain) or feedback_gain<=0:
        raise ValueError('Feedback force gain must be positive finite rad/(N s)')
    feedback_cap=setting.get('feedback_joint_rate_cap_rad_s',.08)
    vendor_rate_limits,feedback_rate_limits=_feedback_rate_limits(mapping,feedback_cap)
    feedback_cap=float(feedback_cap)
    lp_minimum=float(setting.get('lp_min_main_force_N',.1))
    if isinstance(setting.get('lp_min_main_force_N',.1),(bool,np.bool_)) or not np.isfinite(lp_minimum) or not 0<=lp_minimum<=1.5:
        raise ValueError('Simulation LP normal-force floor must be finite in [0,1.5] N under unchanged channel caps')
    lp_options={'min_main_force_N':lp_minimum}
    profile=SDKWaypointProfile(mapping,[setting['open_sdk'],setting['close_sdk']],dt=dt,segment_duration_s=[float(setting.get('close_requested_duration_s',.8))],post_padding_steps=0)
    passive_limit=float(setting.get('passive_contact_force_limit_N',.10))
    if not np.isfinite(passive_limit) or passive_limit<=0:raise ValueError('Passive contact guard must be positive finite N')
    state={'active':False,'scalar':1.,'index':-1,'latched':[False]*5,'latched_q':[None]*5,'free_q':None,'filtered_N':np.zeros(5),'normal_N':np.zeros(5),'actual_pair_streak':0,'lost_pair_steps':0,'had_pair':False,'profile_segments':profile.segments,'control_steps':0,'lp_history':[],'lp_failure_steps':0,'last_lp_phase':None,'force_target_N':np.asarray(setting['finger_force_allocation_N'],float).copy(),'wrench_schedule':None,'wrench_index':0,'last_wrench_mode':None,'passive_contact_force_limit_N':passive_limit,'profile_motion_steps':profile.motion_steps,'profile_padding_steps':profile.padding_steps}
    state.update({'feedback_force_gain':feedback_gain,'feedback_joint_rate_cap_rad_s':feedback_cap,
                  'feedback_vendor_primary_rate_limit_rad_s':vendor_rate_limits,
                  'feedback_effective_primary_rate_limit_rad_s':feedback_rate_limits,
                  'feedback_applied_primary_target_rate_rad_s':{n:0. for n in mapping.channels},
                  'feedback_simulation_parameters':{'force_gain_rad_per_N_s':feedback_gain,
                    'configured_primary_rate_cap_rad_s':feedback_cap,
                    'effective_primary_rate_cap_rad_s':feedback_rate_limits,
                    'vendor_primary_rate_limits_after_all_mimic_folding_rad_s':vendor_rate_limits,
                    'hardware_force_calibrated':False,
                    'bound_scope':'latched target slew; actual qvel and position FD require separate physical guards'}})
    state['lp_min_main_force_N']=lp_minimum
    state['lp_options']=dict(lp_options)
    state['feedback_simulation_parameters']['lp_options']=dict(lp_options)
    tokens=['little','ring','middle','index','thumb']
    def plan(self,now_val,target_val):
        assert now_val>.99 and target_val==0
        n=profile.num_steps;t=np.linspace(0,1,n);vals=now_val+(target_val-now_val)*(10*t**3-15*t**4+6*t**5)
        return {'num_step':n,'per_step':1.9*(target_val-now_val)/(n-1),'result':vals}
    state['plan']=plan
    def forces():
        f=np.zeros(5)
        for c in task.scene.get_contacts():
            names=[x.entity.name for x in c.bodies]
            if '001_bottle' not in names:continue
            name=names[1] if names[0]=='001_bottle' else names[0]
            if not name.startswith('l_'):continue
            for i,token in enumerate(tokens):
                if token in name:
                    f[i]+=sum(abs(float(np.dot(p.impulse,p.normal))) for p in c.points)/dt;break
        return f
    def command(self,value,arm_tag,gripper_eps=.1):
        side=self._side(arm_tag)
        if side!='left':return RealManRobot.set_gripper(self,value,arm_tag,gripper_eps)
        state['active']=True;state['scalar']=float(np.clip(value,0,1));state['index']+=1;self.left_gripper_val=state['scalar']
    robot.set_gripper=MethodType(command,robot)
    def update():
        if not state['active']:return
        measured=robot._measured_positions();f=forces();state['normal_N']=f;state['control_steps']+=1
        state['filtered_N']+=dt/(.04+dt)*(f-state['filtered_N'])
        if np.any(f[:2]>passive_limit):
            raise RuntimeError('Passive little/ring contact exceeds conservative simulation force guard: '+str(f[:2].tolist()))
        for i,n in enumerate(h.channels[:5]):
            if f[i]>.025 and not state['latched'][i]:
                state['latched'][i]=True;state['latched_q'][i]=measured[n]
                print('CONTACT_CAPTURE_LATCH',i,robot.get_hand_sdk('left')[i],f[i],flush=True)
        paired=state['latched'][4] and any(state['latched'][2:4])
        actual_pair=bool(np.all(f[2:5]>.05))
        state['actual_pair_streak']=state['actual_pair_streak']+1 if actual_pair else 0
        state['had_pair']=state['had_pair'] or actual_pair
        state['lost_pair_steps']=0 if actual_pair else state['lost_pair_steps']+1
        if stage['name']=='after_grasp' and state['had_pair'] and state['lost_pair_steps']*dt>.10:
            raise RuntimeError('Contact support lost during official grasp retreat/place')
        wrench_mode='planned' if stage['name']=='after_grasp' and state['wrench_schedule'] is not None and state['wrench_index']<len(state['wrench_schedule']['force_target_world_N']) else 'static_mg'
        lp_phase_ready=stage['name'] in ['close_hold','after_grasp'] or (stage['name']=='close' and state['actual_pair_streak']>=50)
        if lp_phase_ready and (state['last_lp_phase']!=stage['name'] or state['control_steps']%50==0 or state['last_wrench_mode']!=wrench_mode):
            state['last_wrench_mode']=wrench_mode
            state['last_lp_phase']=stage['name']
            from purecontact_lp_v26 import allocate_contact_forces
            inputs=[]
            for contact in task.scene.get_contacts():
                names=[b.entity.name for b in contact.bodies]
                if '001_bottle' not in names:continue
                other=names[1] if names[0]=='001_bottle' else names[0]
                channel=next((t for t in ['middle','index','thumb'] if other.startswith('l_') and t in other),None)
                if channel is None:continue
                for p in contact.points:
                    if abs(float(np.dot(p.impulse,p.normal)))/dt<1e-5:continue
                    inputs.append({'channel':channel,'point_world_m':p.position.tolist(),'force_on_bottle_normal_world':(np.asarray(p.normal)*(1 if names[0]=='001_bottle' else -1)).tolist()})
            component=next(c for c in task.bottle.actor.get_components() if isinstance(c,sapien.physx.PhysxRigidDynamicComponent))
            com=(task.bottle.get_pose().to_transformation_matrix()@component.cmass_local_pose.to_transformation_matrix())[:3,3]
            target_wrench={}
            if stage['name']=='after_grasp' and state['wrench_schedule'] is not None and state['wrench_index']<len(state['wrench_schedule']['force_target_world_N']):
                index=state['wrench_index']
                target_wrench={'target_force_world_N':state['wrench_schedule']['force_target_world_N'][index],'target_torque_world_Nm':state['wrench_schedule']['torque_target_world_Nm'][index]}
            lp=allocate_contact_forces(inputs,com_world_m=com,mass_kg=float(component.mass),previous_allocation_N=state['force_target_N'],**lp_options,**target_wrench)
            state['lp_history'].append({'step':state['control_steps'],'phase':dict(stage),'wrench_mode':wrench_mode,'wrench_index':state['wrench_index'],'planned_timestamp_s':state['wrench_index']*dt if wrench_mode=='planned' else None,'contacts':inputs,'passive_normal_N':f[:2].tolist(),'passive_reaction_not_in_three_contact_lp':bool(np.any(f[:2]>1e-5)),'lp_options':dict(lp_options),'allocation':lp})
            if lp['usable']:
                state['force_target_N']=np.asarray(lp['target_normal_forces_N'],float);state['lp_failure_steps']=0
            else:
                state['lp_failure_steps']+=50
                if state['lp_failure_steps']*dt>=.4:raise RuntimeError('No feasible current physical contact wrench: '+lp['reason'])
        if stage['name']=='after_grasp':state['wrench_index']+=1
        sample=profile.sample(max(state['index'],0));output={};velocity={}
        state['feedback_applied_primary_target_rate_rad_s']={n:0. for n in mapping.channels}
        for i,n in enumerate(h.channels):
            lim=h.limits[n];q=sample['joint_position_rad'][n];qd=sample['primary_velocity_rad_s'][n]
            if i<5 and state['latched'][i]:
                if i in (2,3,4) and paired:
                    desired=state['force_target_N'][i]
                    previous_q=float(state['latched_q'][i])
                    rate_limit=feedback_rate_limits[n]
                    q_step=float(np.clip(feedback_gain*(desired-state['filtered_N'][i]),-rate_limit,rate_limit))*dt
                    state['latched_q'][i]=float(np.clip(previous_q+q_step,lim['lower'],lim['upper']))
                    state['feedback_applied_primary_target_rate_rad_s'][n]=(state['latched_q'][i]-previous_q)/dt
                # Passive little/ring never squeeze further or follow LP's zero loads.
                q=state['latched_q'][i];qd=0.
            # Every uncontacted primary retains its original common quintic phase.
            output[n]=q;velocity[n]=qd
        targets=h.coupled(output)
        for n,q in targets.items():
            robot._hand_joints[n].set_drive_target(float(q));robot._hand_joints[n].set_drive_velocity_target(float(velocity.get(n,0)))
        robot._hand_sdk_target['left']=h.to_sdk(targets);robot.left_gripper_val=state['scalar'];robot._entity_qf(robot.left_entity)
    return state,update
