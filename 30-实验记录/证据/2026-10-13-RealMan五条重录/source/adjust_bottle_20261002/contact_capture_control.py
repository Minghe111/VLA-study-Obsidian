"""Simulation contact-feedback position controller; no electronics or hardware IO.
Maintains the official scalar-command observation and exposes true SDK drive targets.
"""
def install_contact_capture(task):
    robot=task.robot; h=robot._hand_maps['left'];dt=task.scene.get_timestep()
    state={'active':False,'scalar':1.,'latched':[False]*5,'latched_q':[None]*5,'filtered_N':np.zeros(5),'normal_N':np.zeros(5),'samples':[]}
    tokens=['little','ring','middle','index','thumb']
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
        if side!='left' or value>=.999999:
            if side=='left':state.update(active=False,scalar=float(value),latched=[False]*5,latched_q=[None]*5)
            return RealManRobot.set_gripper(self,value,arm_tag,gripper_eps)
        state['active']=True;state['scalar']=float(np.clip(value,0,1));self.left_gripper_val=state['scalar']
    robot.set_gripper=MethodType(command,robot)
    def update():
        if not state['active']:return
        measured=robot._measured_positions();f=forces();state['normal_N']=f
        state['filtered_N']+=dt/(.04+dt)*(f-state['filtered_N'])
        for i,n in enumerate(h.channels[:5]):
            if f[i]>.025 and not state['latched'][i]:
                state['latched'][i]=True;state['latched_q'][i]=measured[n]
                print('CONTACT_CAPTURE_LATCH',i,robot.get_hand_sdk('left')[i],f[i],flush=True)
        paired=state['latched'][4] and any(state['latched'][:4])
        nominal=robot._synergy['left']['close']+state['scalar']*(robot._synergy['left']['open']-robot._synergy['left']['close'])
        output={}
        for i,n in enumerate(h.channels):
            lim=h.limits[n];q=lim['lower']+(1-float(nominal[i])/1000)*(lim['upper']-lim['lower'])
            if i<5 and state['latched'][i]:
                if paired:
                    desired=.3 if i<4 else .3*sum(state['latched'][:4])
                    q_step=float(np.clip(.03*(desired-state['filtered_N'][i]),-.02,.02))*dt
                    state['latched_q'][i]=float(np.clip(state['latched_q'][i]+q_step,lim['lower'],lim['upper']))
                q=state['latched_q'][i]
            output[n]=q
        targets=h.coupled(output)
        for n,q in targets.items():
            robot._hand_joints[n].set_drive_target(float(q));robot._hand_joints[n].set_drive_velocity_target(0)
        robot._hand_sdk_target['left']=h.to_sdk(targets);robot.left_gripper_val=state['scalar'];robot._entity_qf(robot.left_entity)
    return state,update