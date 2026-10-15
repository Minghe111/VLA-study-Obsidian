import carry_mesh_path_audit as cc
from preflight_place_goals import _snapshot,_assert_snapshot
before_read=_snapshot(task)
inspection=cc.CarryMeshPathAuditor(task,'left',palm_to_bottle=np.asarray(grip_measure(task)['relative_bottle_palm']))
try:
    bm=task.bottle.get_pose().to_transformation_matrix()
    verts=inspection.bottle_vertices@bm[:3,:3].T+bm[:3,3]
    table_obj=inspection.static['table']
    tv=inspection.static_vertices['table'] if hasattr(inspection,'static_vertices') else None
    gap,hit=inspection._distance(inspection.static['bottle'],table_obj)
    x={'min_bottle_world_z':float(verts[:,2].min()),'table_nominal_top_z':.740,'nominal_z_clearance_m':float(verts[:,2].min()-.740),'fcl_gap_m':float(gap),'fcl_overlap':bool(hit),'actual_pose':bm.tolist(),'grip':grip_measure(task)}
    (BASE/'closed_geometry_v16_readonly.json').write_text(json.dumps(x,indent=2))
    print('READONLY_CLOSED_GEOMETRY', {k:v for k,v in x.items() if k not in ['actual_pose','grip']},flush=True)
finally:
    inspection.model.set_qpos(inspection.old_model_q,full=True)
    _assert_snapshot(task,before_read)
