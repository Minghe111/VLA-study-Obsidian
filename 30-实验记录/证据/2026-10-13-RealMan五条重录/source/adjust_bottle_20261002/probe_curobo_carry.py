from preflight_place_goals import _snapshot,_assert_snapshot
import carry_mesh_path_audit as cma
before=_snapshot(task)
pose=np.asarray(task.robot.get_left_ee_pose(),float);goal=pose.copy();goal[:3]-=.1*t3d.quaternions.quat2mat(pose[-4:])[:,0]
res=Robot.left_plan_path(task.robot,goal.tolist())
probe={'planner_status':res.get('status'),'task_goal':goal.tolist(),'physical_state_unchanged':False}
if res['status']=='Success':
 old=np.asarray(res['position']);peak=np.max(np.abs(np.diff(old,axis=0)))/.004
 n=max(len(old),int(math.ceil(len(old)*max(1,peak/.15)*1.875)),3001)
 t=np.linspace(0,1,n);u=10*t**3-15*t**4+6*t**5
 pos=np.stack([np.interp(u,np.linspace(0,1,len(old)),old[:,j]) for j in range(6)],axis=1)
 vel=np.gradient(pos,.004,axis=0);vel[0]=0;vel[-1]=0
 res.update(position=pos,velocity=vel)
 np.savez(BASE/'curobo_retreat_probe_20261003.npz',position=pos,velocity=vel)
 probe['audit']=cma.audit_carry_path(task,res,'left',palm_to_bottle=carry_reference[0],min_table_clearance_m=0,min_self_clearance_m=0)
_assert_snapshot(task,before);probe['physical_state_unchanged']=True
(BASE/'curobo_retreat_probe_20261003.json').write_text(json.dumps(probe,indent=2,default=str))
print('CUROBO_RETREAT_PROBE',probe.get('planner_status'),probe.get('audit',{}).get('passed'),probe.get('audit',{}).get('collisions_first_64'),flush=True)
