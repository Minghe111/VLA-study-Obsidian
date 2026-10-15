"""Independent zero-step CPU reconstruction for a recorded boundary event.

Constructs one private CPU scene; no renderer, controller, planner, scene.step,
GPU scene, model inference, hardware IO, or new physical episode is run.
"""
from __future__ import annotations
import ast
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import hashlib
import numpy as np
from dataset50_audit import require, sha
from dataset50_boundary_v2 import CONSTRUCTOR_SHA, META_KEYS, exact_number

SHAPE_OVERLAY_SHA='c0caabd5a576e636c2ca35c6941e3030afd9ddd355b6b36a465e73b5bd816ac8'
FMA_SHA='6e63def2bc02f1c2bd9e4e90fdd71ee70e58961dddafdc6e858a235ce454a9f9'


def fingerprint(component):
    h=hashlib.sha256()
    for shape in component.collision_shapes:
        h.update(type(shape).__name__.encode())
        for name,dtype in [('vertices','<f8'),('triangles','<i8'),('scale','<f8')]:
            require(hasattr(shape,name),'Unsupported native convex shape')
            x=np.asarray(getattr(shape,name),dtype=dtype);h.update(str(x.shape).encode());h.update(x.tobytes())
        h.update(np.asarray(shape.local_pose.to_transformation_matrix(),dtype='<f8').tobytes())
    return h.hexdigest()


def module_from(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def event_scene(trial,row,work):
    p=work/'benchmark_geometry_cache.py';require(sha(p)==CONSTRUCTOR_SHA,'Frozen CPU constructor source changed')
    cache=module_from(p,'dataset50_independent_cpu_constructor')
    tree=ast.parse(inspect.getsource(cache.cpu_snapshot_scene));calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call)]
    require(not any(isinstance(x.func,ast.Attribute) and x.func.attr=='step' for x in calls),'Constructor advances physics')
    require(sum(isinstance(x.func,ast.Attribute) and x.func.attr=='set_qpos' for x in calls)==1,'Constructor must initialize exactly once')
    qs=[x for x in ast.walk(tree) if isinstance(x,ast.With) and len(x.body)==1 and isinstance(x.body[0],ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='initial' for t in x.body[0].targets)]
    bs=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and ast.unparse(x.func)=='b.set_initial_pose']
    require(len(qs)==len(bs)==1,'Frozen initialization sites differ')
    class InitOnly(ast.NodeTransformer):
        def visit_With(self,node):
            return ast.parse('initial = np.asarray(_recorded_qpos, dtype=float).copy()').body[0] if node is qs[0] else self.generic_visit(node)
    tree=InitOnly().visit(tree)
    bs[0].args=[ast.parse('sapien.Pose(np.asarray(_recorded_bottle, dtype=np.float32))',mode='eval').body]
    ast.fix_missing_locations(tree);env=dict(cache.cpu_snapshot_scene.__globals__)
    env.update(_recorded_qpos=row['full_qpos_actual_rad'],_recorded_bottle=row['bottle']['pose'])
    exec(compile(tree,str(p)+':independent_recorded_event_initialization','exec'),env)
    task,receipt=env['cpu_snapshot_scene'](trial)
    expected={'cpu_only_scene':True,'render_system_count':0,'scene_steps':0,'snapshot_initial_qpos_writes_before_audits':1,
        'audit_sapien_qpos_writes':0,'snapshot_initial_qpos_exact':True,'hardware_io':False,'new_physics_episodes':0}
    require(all(receipt.get(k)==v for k,v in expected.items()),'CPU constructor invariants differ')
    require(np.array_equal(np.asarray(task.robot.left_entity.qpos,float),row['full_qpos_actual_rad']),'CPU event qpos not exact')
    receipt.update(event_step=row['step'],constructor_source_sha256=CONSTRUCTOR_SHA,changed_initialization_values=2)
    return task,receipt


def snapshot(task):
    from sapien.physx import PhysxRigidDynamicComponent
    art=task.robot.left_entity;body=task.bottle.actor.find_component_by_type(PhysxRigidDynamicComponent)
    return {'q':np.asarray(art.qpos).copy(),'v':np.asarray(art.qvel).copy(),'root':np.asarray(art.root_pose.to_transformation_matrix()).copy(),
        'bottle':np.asarray(task.bottle.actor.pose.to_transformation_matrix()).copy(),'linear':np.asarray(body.linear_velocity).copy(),
        'angular':np.asarray(body.angular_velocity).copy(),
        'actors':[(a.name,np.asarray(a.pose.to_transformation_matrix()).copy()) for a in task.scene.get_all_actors()]}


def assert_frozen(task,old):
    now=snapshot(task)
    for k,v in old.items():
        if k=='actors':require(len(v)==len(now[k]) and all(a[0]==b[0] and np.array_equal(a[1],b[1]) for a,b in zip(v,now[k])),'CPU actor state changed')
        else:require(np.array_equal(v,now[k]),'CPU physical state changed: '+k)


def verify_cpu_event(trial,row,link,metadata,certificate):
    trial=Path(trial).resolve();work=Path(__file__).resolve().parent
    robotwin=work.parents[2]/'RoboTwin';legacy=work.parent/'adjust_bottle_20261002'
    require(robotwin.is_dir(),'Frozen RoboTwin checkout unavailable for CPU event verification')
    overlay=work/'geometry_overlay.py';require(sha(overlay)==SHAPE_OVERLAY_SHA,'Native shape converter overlay changed')
    require(sha(legacy/'full_mesh_path_audit.py')==FMA_SHA,'Native geometry auditor source changed')
    oldcwd=Path.cwd();oldpath=list(sys.path);task=auditor=None;physical=None
    pins={p:sha(p) for p in [work/'benchmark_geometry_cache.py',overlay,legacy/'full_mesh_path_audit.py']}
    try:
        sys.path[:0]=[str(work),str(legacy),str(robotwin)];os.chdir(robotwin)
        overlay_mod=module_from(overlay,'dataset50_independent_shape_overlay');overlay_mod.install_native_shape_count_overlay(legacy)
        from full_mesh_path_audit import FullMeshPathAuditor, _pose_matrix, _link_vertices
        from sapien.physx import PhysxRigidBaseComponent
        task,receipt=event_scene(trial,row,work);physical=snapshot(task)
        auditor=FullMeshPathAuditor(task,check_table=True)
        require(all(auditor.metadata[k]==metadata[k] for k in META_KEYS),'Reconstructed event geometry metadata differs')
        v=certificate['event_specific_vertex_certificate']
        for key in ['urdf_sha256','model_data_sha256','collision_asset_sha256','geometry_configuration_source_sha256','table_geometry_source_sha256']:
            require(receipt[key]==v['scene_receipt'][key],'Independent CPU asset/config source pin differs: '+key)
        c=task.bottle.actor.find_component_by_type(PhysxRigidBaseComponent)
        ff,bf=fingerprint(auditor.links[link]),fingerprint(c)
        require(ff==v['finger_collision_content_sha256'] and bf==v['bottle_collision_content_sha256'],'Independently rebuilt collision content fingerprint differs')
        vertices=auditor.vertices[link];nativeM=auditor.initial_scene_poses[link];auxM=_pose_matrix(auditor.objects[link].pose)
        nativeW=vertices@nativeM[:3,:3].T+nativeM[:3,3];auxW=vertices@auxM[:3,:3].T+auxM[:3,3]
        source=auditor.objects[link]
        finger=auditor.fcl.FCLObject('independent_native_'+link,auditor.Pose(nativeM),
            [auditor.fcl.CollisionObject(s.get_collision_geometry()) for s in source.shapes],source.shape_poses)
        convertedM=_pose_matrix(finger.pose);convertedW=vertices@convertedM[:3,:3].T+convertedM[:3,3]
        bv,_=_link_vertices(c);nativeBM=np.asarray(task.bottle.actor.pose.to_transformation_matrix(),float)
        bottle=auditor.static['bottle'];fclBM=_pose_matrix(bottle.pose);recordBM=np.asarray(row['bottle']['pose'],float)
        recordBW=bv@recordBM[:3,:3].T+recordBM[:3,3];nativeBW=bv@nativeBM[:3,:3].T+nativeBM[:3,3];fclBW=bv@fclBM[:3,:3].T+fclBM[:3,3]
        measured={'native_to_auxiliary_finger_vertex_error_m':float(np.linalg.norm(nativeW-auxW,axis=1).max()),
            'native_finger_matrix_to_fcl_vertex_error_m':float(np.linalg.norm(nativeW-convertedW,axis=1).max()),
            'recorded_to_native_bottle_vertex_error_m':float(np.linalg.norm(recordBW-nativeBW,axis=1).max()),
            'native_bottle_matrix_to_fcl_vertex_error_m':float(np.linalg.norm(nativeBW-fclBW,axis=1).max())}
        for k,value in measured.items():exact_number(value,v[k],'CPU rebuilt '+k,atol=1e-12)
        conversion=sum(value for k,value in measured.items() if k!='native_to_auxiliary_finger_vertex_error_m');budget=sum(measured.values())
        exact_number(conversion,v['native_signed_gap_conversion_budget_m'],'CPU pure conversion budget',atol=1e-12)
        exact_number(budget,certificate['numerical_vertex_budget_m'],'CPU event vertex budget',atol=1e-12)
        request=auditor.fcl.CollisionRequest;distance=auditor.fcl.DistanceRequest
        def one_shape(obj,i):
            return auditor.fcl.FCLObject('independent_'+obj.name+str(i),obj.pose,[auditor.fcl.CollisionObject(obj.shapes[i].get_collision_geometry())],[obj.shape_poses[i]])
        def query(first,second,tolerance):
            hit=auditor.fcl.collide(first,second,request(num_max_contacts=512,enable_contact=True,gjk_tolerance=tolerance))
            cs=hit.get_contacts();require(len(cs)==hit.num_contacts() and len(cs)<512 and (not hit.is_collision() or cs),'CPU depth probe missing/truncated')
            ds=[float(c.penetration_depth) for c in cs]
            require(all(np.isfinite(x) and x>=0 for x in ds),'CPU nonfinite/negative depth')
            value=float(auditor.fcl.distance(first,second,distance(enable_nearest_points=True,enable_signed_distance=True,distance_tolerance=tolerance)).min_distance)
            require(np.isfinite(value),'CPU nonfinite signed distance')
            return bool(hit.is_collision()),value,ds
        for p in v['native_fk_signed_gap_probes']:
            hit,gap,depth=query(finger,bottle,p['gjk_tolerance'])
            require(hit is p['native_fcl_collision'],'Independent native collision differs')
            exact_number(gap,p['signed_distance_m'],'Independent aggregate native signed gap',atol=1e-12)
            require(gap>=-conversion and all(x<=conversion for x in depth),'CPU aggregate native gap/depth exceeds pure conversion budget')
        for p in v['native_all_shape_pair_probes']:
            first=one_shape(finger,p['finger_shape_index']);second=one_shape(bottle,p['bottle_shape_index'])
            hit,gap,depth=query(first,second,p['gjk_tolerance'])
            require(hit is p['collision'] and len(depth)==len(p['penetration_depths_m']),'Native shape collision/depth count differs')
            exact_number(gap,p['signed_distance_m'],'Native pair signed gap',atol=1e-12)
            for x,y in zip(depth,p['penetration_depths_m']):exact_number(x,y,'Native pair penetration',atol=1e-12)
            require(gap>=-conversion and all(x<=conversion for x in depth),'CPU native pair exceeds pure conversion budget')
        depth_bottle=auditor.fcl.FCLObject('independent_recorded_bottle',auditor.Pose(recordBM),
            [auditor.fcl.CollisionObject(s.get_collision_geometry()) for s in bottle.shapes],bottle.shape_poses)
        maximum=0.
        for p in certificate['shape_pair_depths']:
            first=one_shape(source,p['finger_shape_index']);second=one_shape(depth_bottle,p['bottle_shape_index'])
            for declared in p['probes']:
                hit,gap,depth=query(first,second,declared['gjk_tolerance'])
                require(hit is declared['collision'] and len(depth)==declared['contact_count'],'Aux shape collision/depth count differs')
                for x,y in zip(depth,declared['penetration_depths_m']):exact_number(x,y,'Aux pair penetration',atol=1e-12)
                maximum=max([maximum]+depth)
        exact_number(maximum,certificate['measured_penetration_depth_m'],'Independently rebuilt maximum auxiliary depth',atol=1e-12)
        require(maximum<=budget,'Rebuilt auxiliary depth outside per-event vertex budget')
        return {'passed':True,'cpu_only':True,'scene_steps':0,'render_system_count':0,'gpu_runs':0,'hardware_io':False,'new_physics_episodes':0,
            'scene_receipt':receipt,'finger_collision_content_sha256':ff,'bottle_collision_content_sha256':bf,
            'independent_event_fk_audit':auditor.fk_audit,'independent_vertex_errors_m':measured,'independent_pure_conversion_budget_m':conversion,
            'independent_event_vertex_budget_m':budget,'independent_aux_depth_m':maximum,'shape_pairs':len(certificate['shape_pair_depths']),
            'all_three_precision_probes_recomputed':True,'verifier_source_sha256':sha(Path(__file__))}
    finally:
        if auditor is not None:auditor.model.set_qpos(auditor.old_model_q,full=True)
        if task is not None and physical is not None:assert_frozen(task,physical)
        require(all(sha(p)==value for p,value in pins.items()),'CPU reconstruction source changed during verification')
        os.chdir(oldcwd);sys.path[:]=oldpath