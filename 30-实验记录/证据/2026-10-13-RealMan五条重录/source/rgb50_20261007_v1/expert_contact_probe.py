"""CPU-only planning geometry probe; no scene steps after scene initialization."""
from pathlib import Path
import argparse, json, os, sys
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
opt=ap.parse_args()
ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')
LEGACY=ROOT.parent/'deploy/realman_gen4/adjust_bottle_20261002'
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)];os.chdir(ROOT)
from generation_runner import prepare_task,json_write
from expert_generation import ExpertSession,DEFAULT_SETTING
from types import SimpleNamespace
import full_mesh_path_audit as fma

spec=json.loads(opt.spec.read_text());opt.out.mkdir(parents=True,exist_ok=False)
task,receipt,original=prepare_task(spec,opt.out/'scene',need_planner=False)
side=spec['arm'];prefix='l_' if side=='left' else 'r_'
auditor=fma.FullMeshPathAuditor(task)
bottle=auditor.static['bottle']
start=auditor.initial_sim_q.copy();indices={n:i for i,n in enumerate(auditor.sim_joints)}
mapping=task.robot._hand_maps[side]
rows=[]

def clone(obj,name,world_matrix):
    shapes=[auditor.fcl.CollisionObject(s.get_collision_geometry()) for s in obj.shapes]
    return auditor.fcl.FCLObject(name,auditor.Pose(world_matrix),shapes,[auditor.Pose(p) for p in obj.shape_poses])

try:
    for thumb in [950,925,900,887,875,850,825,800,775,750,700,650,600]:
        sdk=list(DEFAULT_SETTING['close_sdk']);sdk[4]=thumb
        q=start.copy();positions=dict(zip(auditor.sim_joints,q))
        for name,value in mapping.from_sdk(sdk,positions).items():q[indices[name]]=value
        auditor.model.set_qpos(q,full=True)
        palm=fma._pose_matrix(auditor.objects[prefix+'inspire_base_link'].pose)
        local={name:np.linalg.inv(palm)@fma._pose_matrix(obj.pose) for name,obj in auditor.objects.items() if name.startswith(prefix+'inspire_')}
        for hand_x in [-.010,-.005,0.,.005,.010]:
            setting=dict(DEFAULT_SETTING,hand_x=hand_x)
            session=SimpleNamespace(side=side,setting=setting,report={})
            _,grasp=ExpertSession.grasp_pose(session,task.bottle,side)
            desired=task.robot._trans_from_gripper_to_endlink(grasp,arm_tag=side).to_transformation_matrix()
            groups={token:{'min_gap_m':1.,'collision':False,'closest_link':None} for token in ('thumb','index','middle','ring','little')}
            for name,matrix in local.items():
                token=next((t for t in groups if t in name),None)
                if token is None:continue
                candidate=clone(auditor.objects[name],name,desired@matrix)
                gap,hit=auditor._distance(candidate,bottle)
                if gap<groups[token]['min_gap_m']:groups[token]['min_gap_m']=gap;groups[token]['closest_link']=name
                groups[token]['collision']=groups[token]['collision'] or bool(hit)
            rows.append({'thumb_flex_sdk':thumb,'hand_x_m':hand_x,'groups':groups})
    json_write(opt.out/'contact_probe.json',{'spec':spec,'scope':'planning-only full vendor hand collision meshes at nominal task grasp pose; no actual contact-force claim',
                                           'hardware_io':False,'new_motion_physics':False,'rows':rows})
    for row in rows:
        if abs(row['hand_x_m'])<1e-9:
            print(row['thumb_flex_sdk'],{k:round(v['min_gap_m']*1000,3) for k,v in row['groups'].items()},flush=True)
finally:
    auditor.model.set_qpos(auditor.old_model_q,full=True)
    from integrations.realman_rm65b_gen4.native_robot import RealManRobot
    RealManRobot._initialize_hands=original
    task.close_env()
