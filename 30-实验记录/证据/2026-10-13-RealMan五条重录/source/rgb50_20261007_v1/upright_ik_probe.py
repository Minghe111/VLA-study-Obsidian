"""CPU-only upright reachability grid. Not a physical success certificate."""
from pathlib import Path
import json,sys,os,math,inspect
WORK=Path(__file__).resolve().parent;ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');LEGACY=WORK.parent/'adjust_bottle_20261002'
sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)];os.chdir(ROOT);os.environ.setdefault('ROBOTWIN_RT_DENOISER','none')
from geometry_overlay import install_native_shape_count_overlay
install_native_shape_count_overlay(LEGACY)
from generation_runner import prepare_task,json_write
from transforms3d.quaternions import mat2quat
import numpy as np
import full_mesh_path_audit as fma
from mplib import Pose,PlanningWorld
from integrations.realman_rm65b_gen4.native_robot import RealManRobot
OUT=WORK/'upright_ik_probe_v3';OUT.mkdir(exist_ok=False)
rows=[]
for side in ('left','right'):
    spec={'model_id':13,'arm':side,'pose_class':'upright','xy':[-.14 if side=='left' else .14,-.105],'yaw_deg':0.,'seed':101001 if side=='left' else 101002}
    task,receipt,original=prepare_task(spec,OUT/side,need_planner=False)
    try:
        robot=task.robot
        audit=fma.FullMeshPathAuditor(task)
        model=audit.model;palm_name=('l_' if side=='left' else 'r_')+'inspire_base_link'
        names=audit.sim_joints;indices=[names.index(n) for n in getattr(robot,side+'_arm_joints_name')]
        model.set_move_group(palm_name)
        world=PlanningWorld([model],list(audit.static.values()))
        acm=world.get_allowed_collision_matrix()
        for pair in audit.allowed:
            a,b=sorted(pair);acm.set_entry(a,b,True)
        pk=audit.pk;palm_index=list(pk.get_link_names()).index(palm_name)
        mask=np.ones(36,dtype=bool);mask[indices]=False
        limits=np.concatenate([np.asarray(j.limits).reshape(-1,2) for j in audit.art.active_joints])
        inverse_root=np.linalg.inv(robot.left_entity.root_pose.to_transformation_matrix());rng=np.random.default_rng(102003)
        print('CPU_IK',side,'group',indices,flush=True)
        full=np.asarray(robot.left_entity.get_qpos()).copy()
        M=task.bottle.get_pose().to_transformation_matrix();reflection=np.diag([-1.,1.,1.]) if side=='right' else np.eye(3)
        axis=reflection@M[:3,1];axis=axis if axis[2]>0 else -axis;axis/=np.linalg.norm(axis)
        center=reflection@task.bottle.get_contact_point(0,'matrix')[:3,3]
        for phi in (0,-30,30):
            angle=math.radians(phi);radial=np.array([math.sin(angle),math.cos(angle),0.]);radial-=axis*(axis@radial);radial/=np.linalg.norm(radial);lateral=np.cross(radial,axis)
            delta=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
            rotation=reflection@np.stack([radial,lateral,-axis],axis=1)@delta@(reflection if side=='right' else np.eye(3))@delta.T
            goal=reflection@(center-.105*radial-.085*lateral)
            pre=goal-.11*(reflection@lateral)
            for stage,p in (('pre',pre),('grasp',goal)):
                target=[*p,*mat2quat(rotation)];end=robot._trans_from_gripper_to_endlink(target,arm_tag=side)
                goal_base=inverse_root@end.to_transformation_matrix();solutions=[];diagnostics=[]
                for attempt in range(10):
                    seed=full.copy()
                    if attempt:seed[indices]=rng.uniform(limits[indices,0],limits[indices,1])
                    q,ok,error=pk.compute_IK_CLIK(palm_index,Pose(goal_base),seed,mask.tolist(),eps=1e-7,max_iter=2000,dt=.1,damp=1e-8)
                    q=np.asarray(q);valid=bool(ok) and np.all(q[indices]>=limits[indices,0]-1e-5) and np.all(q[indices]<=limits[indices,1]+1e-5)
                    collisions=[]
                    if valid:
                        model.set_qpos(q,full=True);collisions=[str(c) for c in world.check_collision()]
                        if not collisions:solutions.append(q)
                    diagnostics.append({'ok':bool(ok),'limits_pass':valid,'collisions':collisions})
                status='EndpointSuccess' if solutions else 'Fail'
                solutions=np.asarray(solutions);row={'side':side,'azimuth_deg':phi,'stage':stage,'target_pose':target,'IK_status':status,'solutions':solutions.tolist(),'attempts':diagnostics}
                rows.append(row);print('IK_ROW',json.dumps({k:row[k] for k in ('side','azimuth_deg','stage','IK_status')}),'n',len(solutions),flush=True)
                json_write(OUT/'results.json',{'simulation_motion_executed':False,'physical_success_validated':False,'rows':rows})
    finally:RealManRobot._initialize_hands=original;task.close_env()