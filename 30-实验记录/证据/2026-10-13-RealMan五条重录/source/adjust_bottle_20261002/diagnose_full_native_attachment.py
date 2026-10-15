"""Only root queues: read-only attachment frame/index diagnostic, pure CPU."""
import json
from pathlib import Path
import numpy as np
import carry_mesh_path_audit as cma
import preflight_place_goals as ppg


def error(reference,value):
    r=cma._matrix(reference);v=cma._matrix(value);u,_,vt=np.linalg.svd(r[:3,:3]@v[:3,:3].T)
    return {'position_m':float(np.linalg.norm(r[:3,3]-v[:3,3])),
        'rotation_rad':float(np.arccos(np.clip((np.trace(u@vt)-1)/2,-1,1)))}


def run_attachment_diagnostic(task,base,reference):
    from mplib import PlanningWorld
    before=ppg._snapshot(task);a=cma.CarryMeshPathAuditor(task,'left',palm_to_bottle=reference)
    model=a.model;oldq=model.get_qpos().copy()
    try:
        model.set_move_group(a.palm_name);model.set_qpos(a.initial_sim_q,True)
        user=list(model.get_user_link_names());pk=list(a.pk.get_link_names())
        fclnames=[obj.name for obj in model.get_fcl_model().get_collision_objects()]
        actualp=a.initial_scene_poses[a.palm_name];actualb=cma._matrix(task.bottle.get_pose());predicted=actualp@a.reference
        table=a.static['table'];bottle=a.static['bottle']
        tableobj=a.fcl.FCLObject('realman_table',a.Pose(table.pose),table.shapes,table.shape_poses)
        bottleobj=a.fcl.FCLObject('held_bottle',a.Pose(bottle.pose),bottle.shapes,bottle.shape_poses)
        tableposes=[cma._matrix(shape.get_pose()) for shape in tableobj.shapes]
        world=PlanningWorld([model],[tableobj,bottleobj]);acm=world.get_allowed_collision_matrix()
        for pair in a.allowed:
            first,second=sorted(pair);acm.set_entry(first,second,True)
        armids=[a.sim_joints.index(n) for n in task.robot.left_arm_joints_name]
        candidates={}
        for label,ordering in [('pinocchio',pk),('model_user',user),('fcl_collision_objects',fclnames)]:
            if a.palm_name in ordering:candidates.setdefault(ordering.index(a.palm_name),[]).append(label)
        rows=[]
        for index,labels in candidates.items():
            try:
                world.attach_object('held_bottle',model.get_name(),int(index),a.Pose(a.reference),sorted(a.allowed_bottle_hand))
                world.set_qpos_all(a.initial_sim_q[armids]);attached=world.get_attached_object('held_bottle');attached.update_pose()
                anchor=cma._matrix(attached.get_attached_link_global_pose());globalb=cma._matrix(attached.get_global_pose())
                nearest=min(a.initial_scene_poses,key=lambda n:error(a.initial_scene_poses[n],anchor)['position_m']+error(a.initial_scene_poses[n],anchor)['rotation_rad'])
                row={'candidate_index':index,'ordering_labels':labels,'attached_link_id':attached.get_attached_link_id(),
                    'user_name_at_index':user[index] if index<len(user) else None,'pk_name_at_index':pk[index] if index<len(pk) else None,
                    'fcl_name_at_index':fclnames[index] if index<len(fclnames) else None,
                    'attached_anchor_matrix_world':anchor.tolist(),'attached_bottle_matrix_world':globalb.tolist(),
                    'attached_relative_matrix':cma._matrix(attached.get_pose()).tolist(),
                    'anchor_vs_actual_palm':error(actualp,anchor),'bottle_vs_actual_actor':error(actualb,globalb),
                    'bottle_vs_predicted_palm_ref':error(predicted,globalb),'closest_actual_scene_link':nearest,
                    'closest_actual_scene_link_error':error(a.initial_scene_poses[nearest],anchor),
                    'world_collisions':[{k:str(getattr(c,k)) for k in ('link_name1','link_name2','object_name1','object_name2')} for c in world.check_collision()]}
                if any(not np.allclose(cma._matrix(shape.get_pose()),old,atol=1e-10,rtol=0) for shape,old in zip(tableobj.shapes,tableposes)):
                    raise RuntimeError('Auxiliary table shape moved')
            except Exception as e:row={'candidate_index':index,'ordering_labels':labels,'error':str(e),'exception_type':type(e).__name__}
            rows.append(row);print('ATTACHMENT_DIAG',index,labels,'anchor',row.get('anchor_vs_actual_palm'),
                'bottle',row.get('bottle_vs_actual_actor'),'nearest_actual_link',row.get('closest_actual_scene_link'),'error',row.get('error'),flush=True)
        report={'model_name':model.get_name(),'palm_name':a.palm_name,'model_user_link_names':user,'pinocchio_link_names':pk,
            'fcl_collision_object_names':fclnames,'actual_palm_matrix_world':actualp.tolist(),'actual_bottle_matrix_world':actualb.tolist(),
            'predicted_bottle_matrix_world':predicted.tolist(),'frozen_relative_matrix':a.reference.tolist(),
            'actual_model_root_matrix_world':cma._matrix(a.art.root_pose).tolist(),'aux_model_base_matrix_world':cma._matrix(model.get_base_pose()).tolist(),
            'cases':rows,'physical_state_unchanged':True,'scope':'CPU auxiliary attachment diagnostic only, no physics or action.'}
        output=Path(base)/'full_native_attachment_diagnostic_20261003.json'
        with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
        print('ATTACHMENT_DIAG_REPORT',str(output),flush=True);return report
    finally:
        model.set_qpos(oldq,True);source=a.static['bottle']
        a.static['bottle']=a.fcl.FCLObject(source.name,a.original_bottle_pose,source.shapes,source.shape_poses)
        ppg._assert_snapshot(task,before)


if 'task' in globals():attachment_diagnostic=run_attachment_diagnostic(task,BASE,carry_reference[0])
