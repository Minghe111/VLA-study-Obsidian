"""CPU-only zero-step cache benchmark using an accepted physical episode snapshot.

This creates a separate PhysX CPU scene from exactly the saved URDF/meshes,
loads the accepted initial qpos once during construction, and never advances
physics or writes its qpos during either audit. No render system, visuals,
planner, controller, model or GPU is initialized. Production scenes untouched.
"""
from pathlib import Path
from types import SimpleNamespace, MethodType
import argparse
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')
LEGACY = ROOT.parent/'deploy/realman_gen4/adjust_bottle_20261002'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+"\n")


def cpu_snapshot_scene(trial):
    import sapien
    import yaml
    from integrations.realman_rm65b_gen4.physics_setup import configure_collisions
    config = yaml.safe_load((trial/'run_config.yml').read_text())
    report = json.loads((trial/'expert_report.json').read_text())
    receipt = json.loads((trial/'scene_sample.json').read_text())
    ec = config['left_embodiment_config']
    urdf = Path(ec['urdf_path'])
    if not urdf.is_absolute(): urdf = Path(config['left_robot_file'])/urdf
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    # Source accepted scene root was entity7; six empty placeholders preserve
    # the exporter's ephemeral robot name as well as every kinematic field.
    for _ in range(6): scene.add_entity(sapien.Entity())
    loader = scene.create_urdf_loader(); loader.fix_root_link = True
    # Strip visual elements before parsing so no rendering objects exist.
    tree = ET.fromstring(urdf.read_text())
    for link in tree.findall('link'):
        for visual in link.findall('visual'): link.remove(visual)
    loader.urdf_dir = str(urdf.parent); loader.package_dir = None
    srdf = urdf.with_suffix('.srdf')
    loader.ignore_pairs = loader.parse_srdf(srdf.read_text()) if srdf.exists() else []
    builders, actors, cameras = loader._parse_urdf(ET.tostring(tree,encoding='unicode'))
    if len(builders)!=1 or actors or cameras:raise RuntimeError('CPU URDF expected one articulation and no cameras')
    if any(b.visual_records for b in builders[0].link_builders): raise RuntimeError('Unexpected visual records')
    art = builders[0].build()
    rootpose=ec['robot_pose'][0]; art.set_root_pose(sapien.Pose(rootpose[:3],rootpose[3:]))
    with np.load(trial/'physics_telemetry.npz',allow_pickle=False) as z: initial = z['full36_qpos_before_rad'][0].copy()
    if [j.name for j in art.active_joints] != report['full_qpos_joint_names']:
        raise RuntimeError('Snapshot joint ordering differs from native URDF')
    # One-time initialization of an independent CPU-only preview, before audits.
    art.set_qpos(initial)
    # Match exported non-geometric drive/friction fields from the source snapshot.
    manifest=json.loads((Path(config['left_robot_file'])/'source_manifest.json').read_text())
    for side in ('left','right'):
        arm_ec=config[side+'_embodiment_config']
        for name in arm_ec['arm_joints_name'][0 if side=='left' else 1]:
            art.find_joint_by_name(name).set_drive_property(arm_ec['joint_stiffness'],arm_ec['joint_damping'],manifest['arm_limits'][name]['effort'],mode='force')
    for row in report['hardware_hand_drive']['before_motion']['records']:
        values=row['actual'];art.find_joint_by_name(row['joint']).set_drive_property(values['stiffness'],values['damping'],values['force_limit'],mode=values['drive_mode'])
    for name,value in report['hand_joint_friction_actual'].items():art.find_joint_by_name(name).set_friction(float(value))
    physics=configure_collisions(art,Path(config['left_robot_file']))
    model=int(receipt['sample_spec']['model_id'])
    datafile=ROOT/f'assets/objects/001_bottle/model_data{model}.json'
    meshfile=ROOT/f'assets/objects/001_bottle/collision/base{model}.glb'
    data=json.loads(datafile.read_text()); b=scene.create_actor_builder()
    b.add_multiple_convex_collisions_from_file(filename=str(meshfile),scale=data['scale'])
    b.set_initial_pose(sapien.Pose(np.asarray(receipt['initial_pose'],dtype=np.float32)))
    bottle=b.build(name='001_bottle')
    # Native create_table() five collision boxes, without its render visuals.
    height=float(report['table_top_z_m']);thickness=.05; length=1.2;width=.7
    tb=scene.create_actor_builder();tb.set_physx_body_type('static')
    tb.add_box_collision(pose=sapien.Pose([0,0,-thickness/2]),half_size=[length/2,width/2,thickness/2])
    for i in [-1,1]:
        for j in [-1,1]:
            tb.add_box_collision(pose=sapien.Pose([i*(length/2-.1/2),j*(width/2-.1/2),-height/2-.002]),half_size=[thickness/2,thickness/2,height/2-.002])
    tb.set_initial_pose(sapien.Pose([0,0,height]));table=tb.build(name='table')
    robot=SimpleNamespace(left_entity=art,right_entity=art,physics_audit=physics,
                          left_arm_joints_name=ec['arm_joints_name'][0],right_arm_joints_name=ec['arm_joints_name'][1])
    # All scene state is now frozen; audits only write their independent MPLib model.
    task=SimpleNamespace(robot=robot,bottle=SimpleNamespace(actor=bottle),scene=scene)
    return task, {'cpu_only_scene':True,'render_system_count':0,'scene_steps':0,
                  'empty_nonphysical_scene_prefix_entities':6,
                  'snapshot_initial_qpos_writes_before_audits':1,'audit_sapien_qpos_writes':0,
                  'snapshot_initial_qpos_exact':bool(np.array_equal(np.asarray(art.qpos,dtype=float),initial)),
                  'urdf':str(urdf),'urdf_sha256':sha(urdf),
                  'model_data_sha256':sha(datafile),'collision_asset_sha256':sha(meshfile),
                  'geometry_configuration_source_sha256':sha(ROOT/'integrations/realman_rm65b_gen4/physics_setup.py'),
                  'table_geometry_source_sha256':sha(ROOT/'envs/utils/create_actor.py'),
                  'original_trial':str(trial),'hardware_io':False,'new_physics_episodes':0}


def compare_results(a,b,tolerance):
    exact_fields=['passed','side','path_rows','audited_samples','max_vertex_displacement_bound_m',
                  'max_joint_limit_excess_rad','joint_limit_audit','commanded_excess_rad','frozen_excess_rad',
                  'start_excess_rad','thresholds_m','collisions_first_64','unique_collision_pairs',
                  'collision_pair_sample_occurrences','scope']
    exact={key:a[key]==b[key] for key in exact_fields}
    metrics={}
    stamp_changes=[]
    for group,key in [('sampled_minimum_gap','distance_m'),('continuous_gap_lower_bound','lower_bound_m')]:
        for category in a[group]:
            first,second=a[group][category],b[group][category]
            delta=abs(first[key]-second[key])
            metrics[group+'/'+category]={'original_m':first[key],'cached_m':second[key],
                                         'abs_delta_m':delta,'within_tolerance':delta<=tolerance,
                                         'same_pair':first.get('link')==second.get('link') and first.get('other')==second.get('other'),
                                         'original_bound_m':first.get('interval_vertex_bound_m'),
                                         'cached_bound_m':second.get('interval_vertex_bound_m')}
            if {k:v for k,v in first.items() if k!=key}!={k:v for k,v in second.items() if k!=key}:
                stamp_changes.append({'metric':group+'/'+category,'original':first,'cached':second})
    sample_delta=0.;same_nulls=True;same_stamps=True
    for first,second in zip(a['samples'],b['samples']):
        same_stamps &= all(first[k]==second[k] for k in ('sample_index','path_index','interval_fraction'))
        for c,x in first['evaluated_minima_m'].items():
            y=second['evaluated_minima_m'][c]
            if x is None or y is None:same_nulls &= x is y
            else:sample_delta=max(sample_delta,abs(x-y))
    return {'exact_original_fields':exact,'all_exact_fields_identical':all(exact.values()),
            'metric_agreement':metrics,'metric_stamp_changes_from_FCL_roundoff':stamp_changes,
            'all_metric_values_within_tolerance':all(x['within_tolerance'] for x in metrics.values()),
            'sample_stamps_identical':same_stamps,'sample_evaluation_nulls_identical':same_nulls,
            'max_sample_minimum_abs_delta_m':sample_delta,'tolerance_m':tolerance,
            'bitwise_report_identical_excluding_time_and_metadata':not stamp_changes and sample_delta==0. and same_nulls,
            'equivalent_within_reported_numeric_tolerance':all(exact.values()) and all(x['within_tolerance'] for x in metrics.values()) and same_stamps and same_nulls and sample_delta<=tolerance}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--path-index',type=int,default=0);ap.add_argument('--dense-path',action='store_true')
    ap.add_argument('--verify-hits',action='store_true');ap.add_argument('--tolerance-m',type=float,default=1e-6)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)];os.chdir(ROOT)
    from geometry_overlay import install_native_shape_count_overlay
    shape=install_native_shape_count_overlay(LEGACY)
    import full_mesh_path_audit as fma
    task,scene_receipt=cpu_snapshot_scene(args.trial)
    print('CPU_SCENE_READY',flush=True);write(args.out/'scene_recreation.json',scene_receipt)
    if args.dense_path:
        p=args.trial/f'planned_path_{args.path_index}.npz'
        with np.load(p,allow_pickle=False) as z:path=z['position'].copy()
    else:
        p=args.trial/f'free_approach_{args.path_index}.json'
        evidence=json.loads(p.read_text())
        path=np.asarray(evidence['planning_metadata']['retiming']['audit_vertices'],dtype=float)
    original_saved=json.loads((args.trial/f'mesh_path_{args.path_index}.json').read_text())
    auditor=fma.FullMeshPathAuditor(task)
    # Ensure all production robot shapes/exemptions and kinematic chain remain.
    for key in ('full_dof','collision_links','collision_shapes','geometry_types','bottle_shapes',
                'native_exemption_pairs','effective_exemption_pairs','checked_self_pairs','urdf_kinematic_export_sha256'):
        if auditor.metadata[key] != original_saved['metadata'][key]:raise RuntimeError('Recreated native geometry differs: '+key)
    write(args.out/'native_geometry_same.json',{'checked_metadata_equal':True,'metadata':auditor.metadata})
    original_distance=auditor._distance;count={'calls':0}
    def counted(self,a,b):count['calls']+=1;return original_distance(a,b)
    auditor._distance=MethodType(counted,auditor)
    kwargs=dict(vertex_step_m=.001,min_bottle_clearance_m=.0005,min_table_clearance_m=0.,min_self_clearance_m=0.,include_samples=True)
    start=time.monotonic();print('ORIGINAL_START',len(path),flush=True)
    original=auditor.audit(path,'left',**kwargs)
    original_wall=time.monotonic()-start;write(args.out/'original_fma.json',original)
    print('ORIGINAL_DONE',original_wall,count,flush=True);del auditor._distance
    from geometry_cache_overlay import install_exact_distance_cache_overlay
    overlay=install_exact_distance_cache_overlay(LEGACY,verify_hits=args.verify_hits,verification_tolerance_m=args.tolerance_m)
    start=time.monotonic();print('CACHED_START',flush=True)
    cached=auditor.audit(path,'left',**kwargs)
    cached_wall=time.monotonic()-start;write(args.out/'cached_fma.json',cached)
    comparison=compare_results(original,cached,args.tolerance_m)
    result={'source_trial_accepted':json.loads((args.trial/'expert_report.json').read_text())['accepted'],
            'source_path':str(p),'source_path_sha256':sha(p),'dense_path_used':args.dense_path,
            'scene_recreation':scene_receipt,'comparison':comparison,'original_fcl_calls':count['calls'],
            'original_wall_s':original_wall,'cached_wall_s':cached_wall,
            'wall_speedup':original_wall/cached_wall,'cached_distance_audit':cached['distance_cache_audit'],
            'overlay_receipt':overlay,'production_activation_authorized':False,
            'hardware_io':False,'new_physics_runs':0,'gpu_runs':0,'model_queries':0}
    result['passed']=bool(scene_receipt['snapshot_initial_qpos_exact'] and comparison['equivalent_within_reported_numeric_tolerance'])
    write(args.out/'cache_benchmark_report.json',result)
    print('CACHED_DONE',cached_wall,'SPEEDUP',result['wall_speedup'],'PASSED',result['passed'],flush=True)
    if not result['passed']:raise RuntimeError('Independent cache equivalence did not pass')


if __name__=='__main__':main()



