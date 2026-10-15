friction_readback={j.name:float(j.friction) for j in task.robot.left_entity.get_active_joints()}
print('TASK_FRICTION_READBACK',friction_readback,flush=True)
(BASE/'task_friction_readback_20261003.json').write_text(json.dumps(friction_readback,indent=2))
