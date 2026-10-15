"""Task-scoped old-collector bootstrap; default check never starts physics.

Actual new episodes require the reviewed quality addon and original worker
arguments. This does not consume old future command24 labels or overwrite them.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,importlib.util,types

ROOT=Path(__file__).resolve().parent.parent
REPO=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
ARCHIVE=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')
SELECTED=(17,18,20,29,34)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true',help='Starts actual physical collection only after explicit root coordination')
    ap.add_argument('--source-package',type=Path,default=ROOT/'source/rgb50_20261007_v1')
    ap.add_argument('--legacy',type=Path,default=ROOT/'source/adjust_bottle_20261002')
    ap.add_argument('--quality-addon',type=Path);ap.add_argument('--index',type=int,choices=SELECTED)
    ap.add_argument('worker_args',nargs=argparse.REMAINDER);a=ap.parse_args()
    subprocess.run([sys.executable,str(ROOT/'preflight/inspect_source_dependencies.py')],check=True)
    inventory=json.loads((ROOT/'preflight/collector_dependency_inventory.json').read_text())
    guard=json.loads((ARCHIVE/'hardware_guard.json').read_text());mapping={
      '/tmp/realman-eval60k-20261011/local_eval/hardware/robot_head_down_fixed.urdf':ARCHIVE/'shared_frozen_runtime/robot_head_down_fixed.urdf',
      '/tmp/realman-eval60k-20261011/local_eval/scripts/remote_base_task_snapshot.py':ARCHIVE/'shared_frozen_runtime/remote_base_task_snapshot.py'}
    errors=[];actual={}
    for name,expected in guard['expected_sha256'].items():
        p=mapping.get(name,Path(name))
        if not p.is_file():errors.append('missing protected input '+str(p));continue
        actual[str(p)]=sha(p)
        if actual[str(p)]!=expected:errors.append('protected input changed '+str(p))
    evidence={'scope':'Task-local bootstrap readiness, no fake replay collection','selected_indices':list(SELECTED),
              'source_package':str(a.source_package),'legacy':str(a.legacy),'missing_collector_modules':inventory['missing_mandatory_generation_modules'],
              'hardware_guard_expected_count':len(guard['expected_sha256']),'hardware_guard_errors':errors,'hardware_guard_actual_sha256':actual,
              'frozen_base_task':str(mapping['/tmp/realman-eval60k-20261011/local_eval/scripts/remote_base_task_snapshot.py']),
              'original_future_command24_labels_loaded':False,'original_dataset_overwritten':False,'gpu_or_simulation_started':False}
    (ROOT/'preflight/scratch_entry_readiness.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if not a.run:
        print(json.dumps({k:v for k,v in evidence.items() if k!='hardware_guard_actual_sha256'},indent=2));return 0
    # Fail before import/renderer if dependencies, an original scene or new
    # planner-quality rules are missing. No unreviewed old long-wrist replay.
    if errors or inventory['missing_mandatory_generation_modules']:
        raise RuntimeError('Protected source/dependency readiness has not passed')
    if a.index is None or a.quality_addon is None:raise ValueError('Actual rerecord requires one authorized original scene and reviewed quality addon')
    quality=a.quality_addon.resolve();quality.relative_to(ROOT)
    if not quality.is_file():raise FileNotFoundError(quality)
    if not a.source_package.is_dir() or not a.legacy.is_dir():raise RuntimeError('Mirrored original source directories missing')
    args=a.worker_args[1:] if a.worker_args and a.worker_args[0]=='--' else a.worker_args
    if not args:raise ValueError('Pass exact original generation worker parser arguments after--')
    sys.path[:0]=[str(a.source_package),str(a.legacy),str(REPO)]
    os.chdir(REPO)
    # Exact actual Base_Task snapshot from successful current expert/model
    # scene restoration. Canonical checkout stays untouched.
    import envs
    snapshot=mapping['/tmp/realman-eval60k-20261011/local_eval/scripts/remote_base_task_snapshot.py']
    base=types.ModuleType('envs._base_task');base.__package__='envs';base.__file__=str(snapshot)
    sys.modules['envs._base_task']=base;exec(compile(snapshot.read_text(),str(snapshot),'exec'),base.__dict__)
    spec=importlib.util.spec_from_file_location('rerecord5_reviewed_quality_addon',quality);addon=importlib.util.module_from_spec(spec);spec.loader.exec_module(addon)
    if not callable(getattr(addon,'install_before_original_worker',None)):raise RuntimeError('Quality addon must explicitly install task-local path/scene/plan-quality binding')
    receipt=addon.install_before_original_worker(index=a.index,repository=REPO,source_package=a.source_package,legacy_directory=a.legacy,
                                               output_root=ROOT/'new_physical_episodes',frozen_sapien_urdf=mapping['/tmp/realman-eval60k-20261011/local_eval/hardware/robot_head_down_fixed.urdf'])
    required={'original_scene_restore_bound':True,'same_hardware_and_cameras_and_physics':True,'SDK24_15x4ms_clock_retained':True,
              'future_original_commands_not_used':True,'planner_quality_acceptance_enabled':True,'original_dataset_unchanged':True}
    if not isinstance(receipt,dict) or any(receipt.get(k)!=v for k,v in required.items()):raise RuntimeError('Reviewed addon contract incomplete')
    (ROOT/'preflight'/f'quality_addon_{a.index:07d}_receipt.json').write_text(json.dumps({'addon':str(quality),'addon_sha256':sha(quality),'receipt':receipt},indent=2)+'\n')
    # The downloaded, unchanged original worker owns genuine scene stepping,
    # force-feedback control, RGB/time/state capture and native HDF export.
    import worker_settle_wake_v5
    original=sys.argv;sys.argv=[str(a.source_package/'worker_settle_wake_v5.py'),*args]
    try:worker_settle_wake_v5.main()
    finally:sys.argv=original
    return 0
if __name__=='__main__':raise SystemExit(main())
