import importlib, full_mesh_path_audit as fma
importlib.reload(fma)
v=fma.validate_scene_fk(task);print('FULL_MESH_FK',json.dumps(v,default=str),flush=True);(BASE/'full_mesh_fk_validation_20261003.json').write_text(json.dumps(v,default=str,indent=2))
pre,g=side_entry_grasp(task,task.bottle,'left');p=task.robot._trans_from_gripper_to_endlink(g,arm_tag='left')
m=task.robot.left_mplib_planner;res=m.plan_screw(task.robot.left_entity.get_qpos(),p,arms_tag='left',log=True)
print('MPLIB_NEAR_RESULT',res['status'],np.asarray(res.get('position',[])).shape,flush=True)
if res['status']=='Success':
    np.savez(BASE/'mplib_near_probe_20261003.npz',position=res['position'],velocity=res['velocity']);a=fma.audit_path(task,res,'left',clearance_m=0);(BASE/'mplib_near_mesh_probe_20261003.json').write_text(json.dumps(a,default=str,indent=2));print('MPLIB_NEAR_MESH',json.dumps({k:v for k,v in a.items() if k not in ['samples','sample_records']},default=str),flush=True)