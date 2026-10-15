"""Collect 20 unique official adjust_bottle seeds with the frozen accepted V29 standard.
New hardware-scoped orchestration only; official code/vendor assets/frozen helpers stay immutable.
Run supervisor in Conda robotwin; workers use independent GPU/process and per-attempt globals.
"""
from pathlib import Path
import argparse, ast, copy, datetime, hashlib, json, os, shutil, subprocess, sys, time, traceback

BASE = Path(__file__).resolve().parent
ROOT = Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')
PYTHON = Path('/bigdata2/liminghe/envs/robotwin/bin/python')
SELF = Path(__file__).resolve()


def sha(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing.' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def verify_sources(expected):
    if not expected:
        raise RuntimeError('Missing frozen source set')
    for file, digest in expected.items():
        if sha(file) != digest:
            raise RuntimeError('Frozen source changed: ' + file)


class UnsupportedOfficialSample(RuntimeError):
    pass


def check_official_sample(context):
    task = context['task']
    sampled = context['bottle_state'](task)
    record = {'seed': context['episode_seed'], 'official_sampling_unmodified': True,
              'sampled': sampled, 'supported_qpose_tag': 0, 'supported_model_id': 13}
    atomic_json(context['attempt_out'].parent / 'official_sample.json', record)
    if task.qpose_tag != 0 or task.model_id != 13:
        context['report'].update(result='unsupported_official_sample', accepted=False, sampled=sampled,
                                batch_overlay=context['overlay_manifest'])
        context['write']()
        raise UnsupportedOfficialSample('Official sample outside verified left/model13 subset')
    context['report']['batch_overlay'] = context['overlay_manifest']
    context['report']['official_sample'] = record


def candidate_overlay(base=BASE):
    """Modify six top-level orchestration expressions, never any original function body."""
    source = (base / 'try_coupled_side_entry_v29.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    original_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    top_loops = [n for n in tree.body if isinstance(n, ast.For)]
    if len(top_loops) != 1:
        raise RuntimeError('Expected one explicit V29 candidate loop')
    loop = top_loops[0]
    counts = {k: 0 for k in ('output', 'runtime', 'report_seed', 'setup_seed', 'accepted_seed', 'first_planner')}

    class Overlay(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node
        def visit_AsyncFunctionDef(self, node):
            return node
        def visit_Assign(self, node):
            targets = [ast.unparse(t) for t in node.targets]
            if targets == ['OUT']:
                counts['output'] += 1
                node.value = ast.copy_location(ast.Name(id='attempt_out', ctx=ast.Load()), node.value)
            elif targets == ['runtime_paths']:
                counts['runtime'] += 1
                if not isinstance(node.value, ast.List):
                    raise RuntimeError('Unexpected runtime source list')
                node.value.elts.append(ast.Name(id='BATCH_RUNTIME_PATH', ctx=ast.Load()))
            elif targets == ['report']:
                if not isinstance(node.value, ast.Dict):
                    raise RuntimeError('Unexpected candidate report literal')
                for i, key in enumerate(node.value.keys):
                    if isinstance(key, ast.Constant) and key.value == 'seed':
                        if not isinstance(node.value.values[i], ast.Constant) or node.value.values[i].value != 7:
                            raise RuntimeError('V29 report seed differs')
                        counts['report_seed'] += 1
                        node.value.values[i] = ast.Name(id='episode_seed', ctx=ast.Load())
            elif targets == ['accepted_seed']:
                if not isinstance(node.value, ast.Constant) or node.value.value != 7:
                    raise RuntimeError('V29 accepted seed differs')
                counts['accepted_seed'] += 1
                node.value = ast.Name(id='episode_seed', ctx=ast.Load())
            elif targets == ['task.robot.left_plan_path']:
                if ast.unparse(node.value) != 'MethodType(precise_path, task.robot)':
                    raise RuntimeError('V29 planner binding differs')
                counts['first_planner'] += 1
                node.value = ast.parse('MethodType(batch_precise_path,task.robot)', mode='eval').body
            return self.generic_visit(node)
        def visit_Call(self, node):
            if ast.unparse(node.func) == 'args.update':
                for kw in node.keywords:
                    if kw.arg == 'seed':
                        if not isinstance(kw.value, ast.Constant) or kw.value.value != 7:
                            raise RuntimeError('V29 setup seed differs')
                        counts['setup_seed'] += 1
                        kw.value = ast.Name(id='episode_seed', ctx=ast.Load())
            return self.generic_visit(node)

    Overlay().visit(loop)
    if any(value != 1 for value in counts.values()):
        raise RuntimeError('Unexpected candidate overlay sites: ' + str(counts))
    fresh_setup_sites = 0
    body = []
    for node in loop.body:
        body.append(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'task.setup_demo':
            fresh_setup_sites += 1
            body.append(ast.Expr(value=ast.Call(func=ast.Name(id='batch_check_sample', ctx=ast.Load()), args=[], keywords=[])))
    if fresh_setup_sites != 1:
        raise RuntimeError('Expected exactly one fresh candidate scene setup')
    loop.body = body
    ast.fix_missing_locations(tree)
    current_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if current_functions != original_functions:
        raise RuntimeError('Candidate callback/guard/timing function changed')
    compiled_digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    manifest = {'source': str(base / 'try_coupled_side_entry_v29.py'), 'source_sha256': sha(base / 'try_coupled_side_entry_v29.py'),
                'overlay_ast_sha256': compiled_digest, 'sites': counts, 'all_original_functions_ast_identical': True,
                'first_approach': 'fresh current-target Robot.left_plan_path followed by original full mesh audit; no seed7 cache',
                'official_sampler': 'unmodified; only verified qpose_tag0/model13 accepted',
                'candidate_and_replay_seed': 'same unique official seed per attempt', 'physical_guards_changed': False}
    return compile(tree, str(base / 'try_coupled_side_entry_v29.py'), 'exec'), manifest


def first_planner_source():
    return '''
def batch_precise_path(self,target_pose,constraint_pose=None,**kw):
    if dense_calls[0] != 0 or constraint_pose is not None:
        return precise_path(self,target_pose,constraint_pose=constraint_pose,**kw)
    res=Robot.left_plan_path(self,target_pose,constraint_pose=constraint_pose,**kw)
    if res['status']=='Success':
        audit=fma.audit_path(task,res,'left',min_bottle_clearance_m=.0005,min_table_clearance_m=0,min_self_clearance_m=0)
        (OUT/'mesh_path_0.json').write_text(json.dumps(audit,default=str,indent=2))
        np.savez(OUT/'planned_path_0.npz',position=res['position'],velocity=res['velocity'])
        report['batch_fresh_first_approach']={'passed':audit['passed'],'planner':'Robot.left_plan_path current scene/current target','target_pose':np.asarray(target_pose).tolist(),'seed':episode_seed,'cached_path_used':False,'samples':len(res['position'])}
        print('BATCH_FRESH_APPROACH',report['batch_fresh_first_approach'],flush=True)
        if not audit['passed']:return {'status':'Fail','full_mesh_audit_failed':True}
    return res
'''


def clear_runtime(task):
    for name in ('move', 'take_dense_action', 'choose_grasp_pose', 'get_place_pose', 'move_by_displacement', 'get_obs'):
        task.__dict__.pop(name, None)
    if hasattr(task, 'robot'):
        for name in ('set_gripper', 'set_arm_joints', 'left_plan_grippers', 'right_plan_grippers',
                     'left_plan_path', 'right_plan_path', 'left_plan_multi_path', 'right_plan_multi_path'):
            task.robot.__dict__.pop(name, None)


def make_context(seed, attempt, task, original_initialize_hands, overlay_manifest, spec):
    import importlib
    sys.argv = [str(BASE / 'run_adjust_bottle.py'), '--seed', str(seed), '--label', 'batch_v30', '--align-table']
    context = {'__name__': '__main__', '__file__': str(BASE / 'run_adjust_bottle.py'),
               'bootstrap_out': attempt / 'bootstrap', 'attempt_out': attempt / 'candidate',
               'episode_seed': seed, 'BATCH_RUNTIME_PATH': SELF, 'original_initialize_hands': original_initialize_hands,
               'accepted': False, 'overlay_manifest': dict(overlay_manifest, seed=seed), 'batch_spec': spec}
    prelude = (BASE / 'run_adjust_bottle.py').read_text(encoding='utf-8').split('\ntry:\n Task=')[0]
    original_out = "OUT=BASE/f'{opt.task}_{opt.label}_seed{opt.seed}'"
    if prelude.count(original_out) != 1 or prelude.count("BASE/'ready_home.json'") != 1:
        raise RuntimeError('Unexpected frozen bootstrap source')
    prelude = prelude.replace(original_out, 'OUT=bootstrap_out').replace("BASE/'ready_home.json'", "BASE/'ready_home_elevated.json'")
    exec(compile(prelude, str(BASE / 'run_adjust_bottle.py'), 'exec'), context)
    exec('import transforms3d as t3d\nfrom types import MethodType\nfrom envs._base_task import Base_Task\nfrom envs.robot.robot import Robot', context)
    if task is None:
        task = getattr(importlib.import_module('envs.adjust_bottle'), 'adjust_bottle')()
        context['task'] = task
        task.setup_demo(**context['args'])
    context['task'] = task
    exec(compile(first_planner_source(), str(SELF), 'exec'), context)
    context['batch_check_sample'] = lambda: check_official_sample(context)
    return context


def run_cpu(command):
    subprocess.run([str(PYTHON), '-B', *map(str, command)], check=True, cwd=ROOT)


def export_delivery_videos(replay):
    run_cpu([BASE / 'export_timed_preview_v12.py', replay])
    import cv2
    report_path = replay / 'replay_report.json'
    report_hash = sha(report_path)
    r = json.loads(report_path.read_text())
    out = {'passed': True, 'simulation_duration_s': r['physical_steps'] * r['dataset']['physical_step_s'],
           'source_report_sha256': report_hash, 'videos': {}}
    def decode(path):
        cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS); count = 0; shape = None
        if not cap.isOpened():raise RuntimeError('Cannot decode video ' + str(path))
        while True:
            ok, frame = cap.read()
            if not ok:break
            if shape is not None and shape != frame.shape:raise RuntimeError('Changed video resolution')
            shape = frame.shape; count += 1
        cap.release()
        if not count or fps <= 0:raise RuntimeError('Empty video')
        return {'decoded_frames': count, 'fps': fps, 'resolution': [shape[1],shape[0]], 'duration_s': count/fps}
    for name in ('external_realtime', 'head_realtime'):
        src = replay / 'video' / (name + '.mp4'); dst = replay / 'video' / (name + '_h264.mp4')
        before = sha(src)
        subprocess.run(['/usr/bin/ffmpeg', '-hide_banner','-loglevel','error','-n','-i',str(src),'-map','0:v:0',
                        '-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)],check=True)
        a, b = decode(src), decode(dst)
        if a != b or sha(src) != before:raise RuntimeError('Video transcode changed source or timing')
        out['videos'][name]={'source_sha256':before,'delivery_sha256':sha(dst),'delivery_path':str(dst),
                             'decode':b,'speed_changed':False,'source_unchanged':True}
    if sha(report_path) != report_hash:raise RuntimeError('Published report changed during video export')
    atomic_json(replay / 'delivery_video_validation.json', out)


def worker(batch, worker_id, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['ROBOTWIN_RT_DENOISER'] = 'none'
    spec = json.loads((batch / 'batch_spec.json').read_text())
    frozen = json.loads(Path(spec['source_freeze_file']).read_text())['source_hashes']
    verify_sources(frozen)
    sys.path.insert(0, str(BASE)); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
    from integrations.realman_rm65b_gen4.native_robot import RealManRobot
    original_initializer = RealManRobot._initialize_hands
    candidate_code, overlay_manifest = candidate_overlay()
    status_file = batch / 'workers' / ('worker_%d.json' % worker_id)
    task = None
    inbox = batch / 'jobs' / ('worker_%d' % worker_id)
    inbox.mkdir(parents=True, exist_ok=True)
    def status(phase, **extra):
        atomic_json(status_file, {'worker_id':worker_id,'gpu':gpu,'pid':os.getpid(),'phase':phase,
                                'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})
    status('idle')
    try:
        while not (batch / 'STOP').exists():
            pending = sorted(inbox.glob('*.job.json'))
            if not pending:time.sleep(.5);continue
            job = pending[0]; payload = json.loads(job.read_text()); seed = int(payload['seed']); job_id = payload['job_id']
            job.rename(job.with_suffix('.running'))
            attempt = batch / 'attempts' / job_id
            attempt.mkdir(parents=True, exist_ok=False)
            result = {'job_id':job_id,'seed':seed,'worker_id':worker_id,'gpu':gpu,'attempt_dir':str(attempt),'status':'running'}
            context = None
            try:
                verify_sources(frozen)
                status('candidate',seed=seed,job_id=job_id,attempt_dir=str(attempt))
                clear_runtime(task) if task is not None else None
                RealManRobot._initialize_hands = original_initializer
                context = make_context(seed,attempt,task,original_initializer,overlay_manifest,spec)
                task = context['task']
                exec(candidate_code, context)
                import faulthandler
                faulthandler.cancel_dump_traceback_later()
                cr = context['report']
                if cr.get('accepted') is not True or cr.get('result') != 'physics_success':
                    result.update(status='candidate_failed',candidate_result=cr.get('result'),candidate_dir=str(context['OUT']),
                                  error=cr.get('traceback','').splitlines()[-1:] or cr.get('full_native_planning_failures'))
                else:
                    verify_sources(frozen)
                    status('reset_replay',seed=seed,job_id=job_id,attempt_dir=str(attempt))
                    context['accepted_replay_version']='v29'; context['accepted_replay_label']='accepted_replay'
                    exec(compile((BASE / 'accepted_reset_replay_v29.py').read_text(),str(BASE / 'accepted_reset_replay_v29.py'),'exec'), context)
                    replay = attempt / 'candidate' / 'accepted_replay'
                    rr = json.loads((replay / 'replay_report.json').read_text())
                    if rr.get('result') != 'one_strictly_validated_episode_collected' or rr.get('accepted') is not True:
                        raise RuntimeError('Fresh accepted replay did not finish publication')
                    status('independent_validation',seed=seed,job_id=job_id,replay_dir=str(replay))
                    run_cpu([BASE/'validate_collected_dataset.py','--replay-dir',replay,'--repo',ROOT,'--output',replay/'dataset_validation.json'])
                    run_cpu([BASE/'verify_actual_support_departure.py','--episode-dir',replay,'--geometry',attempt/'candidate/carry_mesh_path_3.json','--output',replay/'actual_support_readback.json'])
                    run_cpu([BASE/'independent_short_episode_qa.py',replay,'--manifest',ROOT/'assets/embodiments/realman-rm65b-inspire-gen4/source_manifest.json',
                             '--first-lift-dense-call','4','--require-accepted','--output',replay/'independent_short_episode_physics_qa.json'])
                    status('video_export',seed=seed,job_id=job_id,replay_dir=str(replay))
                    export_delivery_videos(replay)
                    verify_sources(frozen)
                    result.update(status='accepted',replay_dir=str(replay),hdf5_sha256=rr['dataset']['sha256'],
                                  physical_steps=rr['physical_steps'],duration_s=rr['physical_steps']*rr['dataset']['physical_step_s'],
                                  pairs=rr['dataset']['pairs'],recorded_frames=rr['dataset']['recorded_frames'])
            except UnsupportedOfficialSample as exc:
                result.update(status='skipped_unsupported_official_sample',error=str(exc))
            except Exception:
                result.update(status='attempt_exception',traceback=traceback.format_exc())
                print(result['traceback'],flush=True)
            finally:
                try:
                    import faulthandler
                    faulthandler.cancel_dump_traceback_later()
                except ImportError:pass
                result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                atomic_json(batch/'results'/(job_id+'.json'),result)
                status('idle',last_job_id=job_id,last_status=result['status'])
                context = None
                import gc
                gc.collect()
    finally:
        if task is not None:
            try:task.close_env()
            except Exception:traceback.print_exc()
        RealManRobot._initialize_hands = original_initializer
        status('stopped')


def supervise(batch, spec):
    import fcntl
    from batch_dataset_publication_v30 import publish_accepted_episode, finalize_dataset
    batch.mkdir(parents=True,exist_ok=True)
    lock = (batch/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing = batch/'batch_spec.json'
    if existing.exists():
        if json.loads(existing.read_text()) != spec:raise RuntimeError('Refusing different spec on resume')
    else:atomic_json(existing,spec)
    if (batch/'complete.json').exists():
        print('Already complete',flush=True);return
    if (batch/'STOP').exists():raise RuntimeError('Existing STOP requires explicit resume review')
    for name in ('jobs','results','attempts','workers','logs','dataset'):(batch/name).mkdir(exist_ok=True)
    frozen=json.loads(Path(spec['source_freeze_file']).read_text())['source_hashes'];verify_sources(frozen)
    processes={}; logs={};active={};handled=set(); next_seed=int(spec.get('start_seed',7)); start=time.monotonic()
    accepted_records=[]
    if (batch/'dataset/manifest.json').exists():
        previous=json.loads((batch/'dataset/manifest.json').read_text());accepted_records=previous.get('episodes',[])
        handled.update(r['job_id'] for r in accepted_records)
    for f in (batch/'results').glob('*.json'):
        r=json.loads(f.read_text());next_seed=max(next_seed,int(r['seed'])+1)
    atomic_json(batch/'supervisor_pid.json',{'pid':os.getpid(),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    try:
        for i,gpu in enumerate(spec['gpus']):
            log=(batch/'logs'/('worker_%d.log'%i)).open('a',buffering=1);logs[i]=log
            p=subprocess.Popen([str(PYTHON),'-u',str(SELF),'worker','--batch',str(batch),'--worker-id',str(i),'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[i]=p
        while len(accepted_records) < int(spec['episode_num']):
            verify_sources(frozen)
            for file in sorted((batch/'results').glob('*.json')):
                r=json.loads(file.read_text());job_id=r['job_id']
                if job_id in handled:continue
                if r['status']=='accepted':
                    index=len(accepted_records)
                    record=publish_accepted_episode(r['replay_dir'],batch/'dataset',index,r['seed'],frozen,spec)
                    record=dict(record,job_id=job_id)
                    accepted_records.append(record)
                    print('BATCH_ACCEPTED',len(accepted_records),'/',spec['episode_num'],'seed',r['seed'],'duration',r['duration_s'],flush=True)
                handled.add(job_id)
                active.pop(int(r['worker_id']),None)
            if len(accepted_records) >= int(spec['episode_num']):break
            for i,p in processes.items():
                if p.poll() is not None:raise RuntimeError('Worker exited unexpectedly: %d code %s'%(i,p.returncode))
                if i in active or len(accepted_records)+len(active)>=int(spec['episode_num']):continue
                worker_status=batch/'workers'/('worker_%d.json'%i)
                if not worker_status.exists() or json.loads(worker_status.read_text()).get('phase')!='idle':continue
                seed=next_seed;next_seed+=1;job_id='seed_%07d'%seed
                if (batch/'attempts'/job_id).exists():raise RuntimeError('Attempt seed directory already exists without handling')
                payload={'job_id':job_id,'seed':seed,'worker_id':i}
                atomic_json(batch/'jobs'/('worker_%d'%i)/(job_id+'.job.json'),payload);active[i]=payload
            worker_states={str(i):json.loads((batch/'workers'/('worker_%d.json'%i)).read_text()) for i in processes if (batch/'workers'/('worker_%d.json'%i)).exists()}
            results=[json.loads(f.read_text()) for f in (batch/'results').glob('*.json')]
            atomic_json(batch/'progress.json',{'state':'running','accepted':len(accepted_records),'target':spec['episode_num'],'attempts_finished':len(results),
                        'skipped':sum(r['status']=='skipped_unsupported_official_sample' for r in results),
                        'failed':sum(r['status'] in ('candidate_failed','attempt_exception') for r in results),'workers':worker_states,'next_seed':next_seed,'elapsed_s':time.monotonic()-start})
            time.sleep(1)
        complete=finalize_dataset(batch/'dataset',frozen,spec)
        atomic_json(batch/'complete.json',complete)
        atomic_json(batch/'progress.json',{'state':'complete','accepted':len(accepted_records),'target':spec['episode_num'],'elapsed_s':time.monotonic()-start,'dataset_root':str(batch/'dataset')})
    except Exception:
        atomic_json(batch/'supervisor_error.json',{'traceback':traceback.format_exc(),'accepted':len(accepted_records),'active':active})
        raise
    finally:
        (batch/'STOP').touch(exist_ok=True)
        for p in processes.values():
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:print('Worker finishing current accepted attempt; STOP will be obeyed before next job',flush=True)
        for log in logs.values():log.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='mode',required=True)
    check=sub.add_parser('check-overlay')
    run=sub.add_parser('supervise');run.add_argument('--batch',type=Path,required=True);run.add_argument('--spec',type=Path,required=True)
    work=sub.add_parser('worker');work.add_argument('--batch',type=Path,required=True);work.add_argument('--worker-id',type=int,required=True);work.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args()
    if args.mode=='check-overlay':
        _,m=candidate_overlay();print(json.dumps(m,indent=2));return
    if args.mode=='worker':worker(args.batch.resolve(),args.worker_id,args.gpu)
    else:supervise(args.batch.resolve(),json.loads(args.spec.read_text()))


if __name__=='__main__':main()