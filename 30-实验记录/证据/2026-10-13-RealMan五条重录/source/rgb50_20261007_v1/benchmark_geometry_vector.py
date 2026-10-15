"""Independent CPU comparison of original FMA versus vector-only broadphase.

Collision-path fallback is a synthetic auxiliary FK benchmark, never a physical
rollout. It is used when a rejected production trial saved no failed path array.
"""
from pathlib import Path
from types import MethodType
import argparse,copy,json,os,sys,time
import numpy as np
from benchmark_geometry_cache import cpu_snapshot_scene,sha,write,ROOT,LEGACY,WORK


def collision_path(auditor,side='right'):
    indices=[auditor.sim_joints.index(n) for n in getattr(auditor.task.robot,side+'_arm_joints_name')]
    base=auditor.initial_sim_q.copy();limits=np.concatenate([np.asarray(j.limits).reshape(-1,2) for j in auditor.art.active_joints])
    prefix='l_' if side=='left' else 'r_'
    tests=0
    try:
        for amount in [.1,.2,.3,.4,.5,.6,.8,1.,1.2]:
            for joint in indices:
                for sign in [-1.,1.]:
                    q=base.copy();q[joint]+=sign*amount
                    if not limits[joint,0] <= q[joint] <= limits[joint,1]:continue
                    auditor.model.set_qpos(q,full=True);tests+=1
                    for name,obj in auditor.objects.items():
                        if not name.startswith(prefix):continue
                        if auditor._distance(obj,auditor.static['table'])[1]:
                            return np.vstack([base[indices],q[indices]]),{
                                'synthetic_auxiliary_FK_collision_path_only':True,'physical_success_claim':False,
                                'side':side,'deterministic_ordered_endpoint_searches':tests,
                                'changed_joint':auditor.sim_joints[joint],'delta_rad':sign*amount,
                                'known_true_table_collision_link':name,'physics_steps':0,'gpu_runs':0}
        raise RuntimeError('No deterministic auxiliary table collision endpoint found')
    finally:auditor.model.set_qpos(auditor.old_model_q,full=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--collision-path',action='store_true');ap.add_argument('--dense-path',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)];os.chdir(ROOT)
    from geometry_overlay import install_native_shape_count_overlay
    install_native_shape_count_overlay(LEGACY)
    import full_mesh_path_audit as fma
    task,receipt=cpu_snapshot_scene(args.trial);write(args.out/'scene_recreation.json',receipt)
    auditor=fma.FullMeshPathAuditor(task)
    saved=json.loads((args.trial/'mesh_path_0.json').read_text())
    for key in ('full_dof','collision_links','collision_shapes','geometry_types','bottle_shapes','native_exemption_pairs',
                'effective_exemption_pairs','checked_self_pairs','urdf_kinematic_export_sha256'):
        if auditor.metadata[key]!=saved['metadata'][key]:raise RuntimeError('Native geometry mismatch: '+key)
    if args.collision_path:
        side='right';path,path_receipt=collision_path(auditor,side)
        np.savez(args.out/'synthetic_auxiliary_right_table_collision_path.npz',position=path)
    elif args.dense_path:
        side='left';p=args.trial/'planned_path_0.npz'
        with np.load(p,allow_pickle=False) as z:path=z['position'].copy()
        path_receipt={'source':str(p),'source_sha256':sha(p),'accepted_source_dense_path':True}
    else:
        side='left';p=args.trial/'free_approach_0.json';e=json.loads(p.read_text())
        path=np.asarray(e['planning_metadata']['retiming']['audit_vertices'],float)
        path_receipt={'source':str(p),'source_sha256':sha(p),'accepted_source_exact_audit_vertices':True}
    write(args.out/'path_receipt.json',path_receipt)
    kwargs=dict(vertex_step_m=.001,min_bottle_clearance_m=.0005,min_table_clearance_m=0.,min_self_clearance_m=0.,include_samples=True)
    counts={'original':0,'vector':0};original_distance=auditor._distance
    def counted_original(self,a,b):counts['original']+=1;return original_distance(a,b)
    auditor._distance=MethodType(counted_original,auditor);print('ORIGINAL_START',side,len(path),flush=True)
    start=time.monotonic();original=auditor.audit(path,side,**kwargs);original_wall=time.monotonic()-start
    write(args.out/'original_fma.json',original);del auditor._distance
    print('ORIGINAL_DONE',original_wall,counts['original'],'PASS',original['passed'],flush=True)
    from geometry_vector_overlay import install_vectorized_aabb_overlay
    overlay=install_vectorized_aabb_overlay(LEGACY)
    def counted_vector(self,a,b):counts['vector']+=1;return original_distance(a,b)
    auditor._distance=MethodType(counted_vector,auditor);print('VECTOR_START',flush=True)
    start=time.monotonic();vector=auditor.audit(path,side,**kwargs);vector_wall=time.monotonic()-start
    write(args.out/'vector_fma.json',vector);del auditor._distance
    a=copy.deepcopy(original);b=copy.deepcopy(vector)
    for item in (a,b):item.pop('elapsed_s',None);item.pop('vector_broadphase_audit',None)
    bit_identical=(a==b);same_calls=counts['original']==counts['vector']
    realcollision=not args.collision_path or any(v['category']=='table' for v in original['collisions_first_64'])
    result={'passed':bool(bit_identical and same_calls and realcollision and receipt['snapshot_initial_qpos_exact']),
            'bit_identical_entire_report_excluding_wall_time_and_new_optimization_receipt':bit_identical,
            'identical_all_sample_minima_and_stamps':original['samples']==vector['samples'],
            'native_fcl_call_counts_identical':same_calls,'fcl_call_counts':counts,
            'true_table_collision_observed_if_requested':realcollision,
            'original_passed':original['passed'],'vector_passed':vector['passed'],
            'path_receipt':path_receipt,'scene_recreation':receipt,
            'original_wall_s':original_wall,'vector_wall_s':vector_wall,'wall_speedup':original_wall/vector_wall,
            'vector_broadphase_audit':vector['vector_broadphase_audit'],'overlay_receipt':overlay,
            'new_physics_runs':0,'gpu_runs':0,'model_queries':0,'hardware_io':False,
            'production_activation_authorized':False}
    write(args.out/'vector_benchmark_report.json',result)
    print('VECTOR_DONE',vector_wall,'SPEEDUP',result['wall_speedup'],'BIT_IDENTICAL',bit_identical,'SAME_FCL',same_calls,flush=True)
    if not result['passed']:raise RuntimeError('Exact vector-only benchmark failed')


if __name__=='__main__':main()
