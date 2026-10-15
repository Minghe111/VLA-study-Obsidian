"""Simulation-only continuous position servo with PhysX contact capture feedback.
Does not represent calibrated RealMan/Inspire electronic tactile or force control.
"""
from continuous_servo import ContinuousHandMapping, SDKWaypointProfile

def install_contact_capture(task,setting):
    robot=task.robot; h=robot._hand_maps['left'];dt=task.scene.get_timestep()
    mapping=ContinuousHandMapping(robot._manifest['hands']['left'])
    profile=SDKWaypointProfile(mapping,[setting['open_sdk'],setting['ready_near_sdk'],setting['close_sdk']],dt=dt,segment_duration_s=[3.,6.],post_padding_steps=500)
    state={'active':False,'scalar':1.,'index':-1,'latched':[False]*5,'latched_q':[None]*5,'free_q':None,'filtered_N':np.zeros(5),'normal_N':np.zeros(5),'actual_pair_streak':0,'lost_pair_steps':0,'had_pair':False,'profile_segments':profile.segments}
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
        measured=robot._measured_positions();f=forces();state['normal_N']=f
        state['filtered_N']+=dt/(.04+dt)*(f-state['filtered_N'])
        for i,n in enumerate(h.channels[:5]):
            if f[i]>.025 and not state['latched'][i]:
                if state['free_q'] is None:state['free_q']={j:measured[j] for j in h.channels}
                state['latched'][i]=True;state['latched_q'][i]=measured[n]
                print('CONTACT_CAPTURE_LATCH',i,robot.get_hand_sdk('left')[i],f[i],flush=True)
        paired=state['latched'][4] and any(state['latched'][:4])
        actual_pair=f[4]>.05 and sum(f[:4])>.05
        state['actual_pair_streak']=state['actual_pair_streak']+1 if actual_pair else 0
        state['had_pair']=state['had_pair'] or actual_pair
        state['lost_pair_steps']=0 if actual_pair else state['lost_pair_steps']+1
        if stage['name']=='after_grasp' and state['had_pair'] and state['lost_pair_steps']*dt>.10:
            raise RuntimeError('Contact support lost during official grasp retreat/place')
        sample=profile.sample(max(state['index'],0));output={};velocity={}
        for i,n in enumerate(h.channels):
            lim=h.limits[n];q=sample['joint_position_rad'][n];qd=sample['primary_velocity_rad_s'][n]
            if state['free_q'] is not None and i<5:
                if state['latched'][i]:
                    if paired:
                        desired=setting['finger_force_allocation_N'][i]
                        q_step=float(np.clip(.03*(desired-state['filtered_N'][i]),-.02,.02))*dt
                        state['latched_q'][i]=float(np.clip(state['latched_q'][i]+q_step,lim['lower'],lim['upper']))
                    q=state['latched_q'][i];qd=0.
                else:
                    goal=profile.sample(profile.motion_steps-1)['joint_position_rad'][n]
                    state['free_q'][n]+=float(np.clip(goal-state['free_q'][n],-.004*dt,.004*dt))
                    q=state['free_q'][n];qd=0.
            output[n]=q;velocity[n]=qd
        targets=h.coupled(output)
        for n,q in targets.items():
            robot._hand_joints[n].set_drive_target(float(q));robot._hand_joints[n].set_drive_velocity_target(float(velocity.get(n,0)))
        robot._hand_sdk_target['left']=h.to_sdk(targets);robot.left_gripper_val=state['scalar'];robot._entity_qf(robot.left_entity)
    return state,update