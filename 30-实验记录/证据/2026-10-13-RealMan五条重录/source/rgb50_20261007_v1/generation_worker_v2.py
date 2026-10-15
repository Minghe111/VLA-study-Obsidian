"""Persistent additive collector. RealMan.reset rebinds planners to each fresh scene."""
from pathlib import Path
import argparse, gc, json, os, pickle, signal, sys, time, traceback
ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--no-rgb',action='store_true');ap.add_argument('--timeout',type=int,default=1800);ap.add_argument('--expert-module',default='expert_generation');ap.add_argument('--backend',choices=['curobo','full_native'],default='curobo')
opt=ap.parse_args();WORK=Path(__file__).resolve().parent
ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');LEGACY=WORK.parent/'adjust_bottle_20261002'
sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)];os.chdir(ROOT);os.environ.setdefault('ROBOTWIN_RT_DENOISER','none')
os.environ["RGB50_EXPERT_MODULE"]=opt.expert_module
import generation_runner_v2 as collector
import importlib
run_episode=importlib.import_module(opt.expert_module).run_episode
from integrations.realman_rm65b_gen4.native_robot import RealManRobot
jobs=json.loads(opt.jobs.read_text());opt.output.mkdir(parents=True,exist_ok=True)
robot=None;receipts=[]
def timeout(signum,frame): raise TimeoutError('Bounded expert trial exceeded timeout')
signal.signal(signal.SIGALRM,timeout)
for number,job in enumerate(jobs):
    spec=job.get('spec',job);spec['planning_backend']=opt.backend;name=job.get('name',f"slot_{int(spec.get('index',number)):07d}_attempt00")
    out=opt.output/name
    if out.exists():
        receipts.append({'name':name,'skipped_existing':True});continue
    out.mkdir();collector.json_write(out/'sample_spec.json',spec)
    start=time.time();task=None;original=None
    try:
        signal.alarm(opt.timeout)
        print('WORKER_START',name,json.dumps(spec),flush=True)
        task,scene,original=collector.prepare_task(spec,out,reuse_robot=robot,backend=opt.backend)
        robot=task.robot
        if abs(task.scene.get_timestep()-.004)>1e-9:raise RuntimeError('Physical clock differs')
        frames=[];cache=out/'.cache/episode0'
        if not opt.no_rgb:cache.mkdir(parents=True)
        def capture(obs,step):
            index=len(frames)
            with (cache/f'{index}.pkl').open('wb') as stream:pickle.dump(obs,stream)
            frames.append(step)
        report=run_episode(task,spec,out,legacy_dir=LEGACY,capture_callback=None if opt.no_rgb else capture)
        if not opt.no_rgb:collector.export_recording(task,out,report,frames)
        receipt={'name':name,'accepted':True,'elapsed_s':time.time()-start,'physics_steps':report['physical_steps'],'rgb_recorded':not opt.no_rgb}
        collector.json_write(out/'worker_receipt.json',receipt)
        print('WORKER_ACCEPTED',json.dumps(receipt),flush=True)
    except Exception as e:
        receipt={'name':name,'accepted':False,'elapsed_s':time.time()-start,'error':str(e),'traceback':traceback.format_exc()}
        collector.json_write(out/'runner_error.json',receipt);print('WORKER_REJECTED',json.dumps(receipt),flush=True)
        # A setup failure may leave planner binding partially initialized. End
        # this worker rather than reusing an uncertain articulation.
        if task is None:
            receipts.append(receipt);break
    finally:
        signal.alarm(0)
        if task is not None:
            try:task.close_env()
            except Exception:pass
        if original is not None:RealManRobot._initialize_hands=original
        gc.collect()
    receipts.append(receipt)
    collector.json_write(opt.output/'worker_status.json',{'jobs_total':len(jobs),'finished':len(receipts),'receipts':receipts})
collector.json_write(opt.output/'worker_status.json',{'jobs_total':len(jobs),'finished':len(receipts),'receipts':receipts,'worker_ended':True})