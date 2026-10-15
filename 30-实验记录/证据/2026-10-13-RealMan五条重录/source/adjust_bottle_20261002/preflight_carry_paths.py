"""Read-only worker preflight. Root alone queues; no scene or GPU operations.
Optional globals: carry_paths_for_audit={label:path}, carry_reference_for_audit=4x4,
carry_audit_options={...}. Default is only a measured holding-state mesh audit.
"""
import hashlib
import importlib
import json
import time
from pathlib import Path
import numpy as np
import carry_mesh_path_audit as cma


def measured_table_support_geometry(task):
    from sapien.physx import PhysxRigidBaseComponent
    fma=cma.fma;bc=task.bottle.actor.find_component_by_type(PhysxRigidBaseComponent)
    v,_=fma._link_vertices(bc);t=cma._matrix(bc.entity.pose);bv=v@t[:3,:3].T+t[:3,3]
    tables=[x for x in task.scene.get_all_actors() if x.name=='table']
    if len(tables)!=1:raise ValueError('Expected exactly one table')
    tc=tables[0].find_component_by_type(PhysxRigidBaseComponent)
    v,_=fma._link_vertices(tc);t=cma._matrix(tc.entity.pose);tv=v@t[:3,:3].T+t[:3,3]
    bottom=float(bv[:,2].min());top=float(tv[:,2].max())
    return {'bottle_collision_vertex_count':len(bv),'bottle_min_world_z_m':bottom,
        'table_max_world_z_m':top,'bottle_min_z_minus_table_max_z_m':bottom-top,
        'bottle_world_aabb_m':[bv.min(0).tolist(),bv.max(0).tolist()],
        'table_world_aabb_m':[tv.min(0).tolist(),tv.max(0).tolist()],
        'scope':'Actual scaled PhysX vertices. Highest-table-Z halfspace is sufficient for clearance, not a complete signed mesh penetration measure. No clipping.'}


def run_carry_preflight(task,base,paths=None,*,side='left',palm_to_bottle=None,options=None):
    importlib.reload(cma);base=Path(base)
    import preflight_place_goals as ppg
    before=ppg._snapshot(task)
    try:
        art=getattr(task.robot,side+'_entity');names=[j.name for j in art.active_joints]
        q=np.asarray(art.get_qpos()).copy();indices=[names.index(n) for n in getattr(task.robot,side+'_arm_joints_name')]
        palmname=('l_' if side=='left' else 'r_')+'inspire_base_link'
        palm=cma._matrix(art.find_link_by_name(palmname).entity.pose);bottle=cma._matrix(task.bottle.get_pose())
        report={'scope':'CPU read-only carry mesh preflight; no execution.','side':side,
            'actual_joint_names':names,'actual_joint_rad':q.tolist(),'actual_arm_joint_rad':q[indices].tolist(),
            'actual_palm_matrix_world':palm.tolist(),'actual_bottle_matrix_world':bottle.tolist(),
            'measured_palm_to_bottle':(np.linalg.inv(palm)@bottle).tolist(),
            'table_support_geometry':measured_table_support_geometry(task),
            'helper_sha256':hashlib.sha256(Path(cma.__file__).read_bytes()).hexdigest(),
            'base_mesh_helper_sha256':hashlib.sha256(Path(cma.fma.__file__).read_bytes()).hexdigest()}
        report['snapshot_only']=paths is None
        if paths is None:paths={'measured_hold_snapshot':q[indices][None,:]}
        if not isinstance(paths,dict) or not paths:raise ValueError('Expected nonempty named paths')
        outcomes={}
        for label,path in paths.items():
            try:outcome=cma.audit_carry_path(task,path,side,palm_to_bottle=palm_to_bottle,**dict(options or {}))
            except Exception as error:outcome={'passed':False,'status':'Exception','exception_type':type(error).__name__,'message':str(error)}
            outcomes[str(label)]=outcome
            print('CARRY_PREFLIGHT',label,'passed',outcome['passed'],'samples',outcome.get('audited_samples'),
                'gaps',json.dumps(outcome.get('continuous_gap_lower_bound')),'collisions',json.dumps(outcome.get('collisions_first_64')),
                'error',outcome.get('message'),flush=True)
            ppg._assert_snapshot(task,before)
        report.update(paths=outcomes,physical_state_unchanged=True)
        output=base/('carry_mesh_preflight_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000)+'.json')
        with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
        print('CARRY_PREFLIGHT_REPORT',str(output),'initial_table_Z_gap_m',report['table_support_geometry']['bottle_min_z_minus_table_max_z_m'],flush=True)
        return report
    finally:ppg._assert_snapshot(task,before)


if 'task' in globals():
    carry_mesh_preflight_report=run_carry_preflight(task,BASE,globals().get('carry_paths_for_audit'),
        side=globals().get('carry_audit_side','left'),palm_to_bottle=globals().get('carry_reference_for_audit'),
        options=globals().get('carry_audit_options'))
elif __name__=='__main__':print(json.dumps(cma.run_cpu_tests(),indent=2))
