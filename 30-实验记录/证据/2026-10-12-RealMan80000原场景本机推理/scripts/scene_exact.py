"""Original source initial scenes; unchanged current actuator hardware/physics."""
from __future__ import annotations
import ast,copy,hashlib,json,math,os,sys,subprocess
from pathlib import Path
from types import MethodType
import numpy as np
import yaml
ROBOTWIN=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
ROOT=Path(__file__).resolve().parent.parent
VENDOR=ROBOTWIN/'assets/embodiments/realman-rm65b-inspire-gen4'
RESET_LEFT=[-1.4338856935501099,-.17370359599590302,-1.2848105430603027,.7786584496498108,-.5827611684799194,.9314123392105103]
RESET_RIGHT=[1.440302848815918,.1718713343143463,1.2854284048080444,-.7795454263687134,.5846511721611023,-.9303343296051025]
OPEN=[1000,1000,1000,1000,1000,0]
DR={'random_background':False,'cluttered_table':False,'clean_background_rate':1.,'random_head_camera_dis':0.,'random_table_height':0.,'random_light':False,'crazy_random_light_rate':0.,'random_embodiment':False}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def assert_input_guard():
    guard=json.loads((ROOT/'hardware_guard.json').read_text())
    bad={p:[v,sha(p)] for p,v in guard['expected_sha256'].items() if sha(p)!=v}
    if bad:raise RuntimeError('Current source or hardware assets changed: '+str(bad))
    assert guard['fixed_reset']=={'left_arm_rad':RESET_LEFT,'right_arm_rad':RESET_RIGHT,'open_sdk6':OPEN}
    return guard
def read_physical_contract(task):
    robot=task.robot;drives=[]
    for name,j in robot._hand_joints.items():
        drives.append({'joint':name,'stiffness':float(j.stiffness),'damping':float(j.damping),'force_limit':float(j.force_limit),'drive_mode':str(j.drive_mode),'friction':float(j.friction),'position_target_rad':float(j.get_drive_target()[0]),'velocity_target_rad_s':float(j.get_drive_velocity_target()[0])})
    shapes=[]
    for link in robot.left_entity.get_links():
        for i,s in enumerate(link.get_collision_shapes()):
            m=s.get_physical_material()
            shapes.append({'link':link.name,'shape':i,'contact_offset_m':float(s.contact_offset),'rest_offset_m':float(s.rest_offset),'groups':list(s.get_collision_groups()),'material':[float(m.static_friction),float(m.dynamic_friction),float(m.restitution)]})
    return {'full_qpos_joint_names':list(robot._hand_joints),'drives':drives,'shapes':shapes,'robot_root_matrix':robot.left_entity.get_root_pose().to_transformation_matrix().tolist(),'tendons':copy.deepcopy(robot.hand_coupling_audit),'simulation_parameters_hardware_calibrated':False}
def structural_contract(contract):
    c=copy.deepcopy(contract)
    for row in c['drives']:row.pop('position_target_rad');row.pop('velocity_target_rad_s')
    # Runtime residuals are observations, not mutable mechanical settings.
    c['tendons']=c['tendons'].get('settings',c['tendons']) if isinstance(c['tendons'],dict) else c['tendons']
    return c

class ImmutableSetupRejected(Exception):
    """Only immutable side/model exclusion; no stability or XY verdict."""
    def __init__(self,metadata):
        self.metadata=metadata
        super().__init__('Immutable model/side candidate rejected before first bottle physics step: '+str(metadata))

def build_exact_expert_scene(seed,output,*,requested_model_id,scene_spec):
    if type(requested_model_id) is not int or requested_model_id not in (4,13,16):raise ValueError('Explicit ordinary quota model13/16 required')
    """Normal official setup/delay/stability with fixed episode-independent reset."""
    guard=assert_input_guard();output=Path(output);sys.path.insert(0,str(ROBOTWIN));os.chdir(ROBOTWIN)
    import importlib
    from integrations.realman_rm65b_gen4.native_robot import RealManRobot
    from integrations.realman_rm65b_gen4.hand_coupling import HandCoupling
    from integrations.realman_rm65b_gen4.three_camera_rgb import attach_rgb_recording
    class FreshRobot(RealManRobot):
        def _initialize_hands(self):
            super()._initialize_hands()
            q=np.asarray(self.left_entity.get_qpos()).copy()
            for side,home in (('left',RESET_LEFT),('right',RESET_RIGHT)):
                for n,v in zip(getattr(self,side+'_arm_joints_name'),home):q[self._joint_index[n]]=v
                initial=dict(zip(self._hand_joints,q))
                for n,v in self._hand_maps[side].from_sdk(OPEN,initial).items():q[self._joint_index[n]]=v
            self.left_entity.set_qpos(q);self.left_entity.set_qvel(np.zeros_like(q))
            for n,j in self._hand_joints.items():j.set_drive_target(float(q[self._joint_index[n]]));j.set_drive_velocity_target(0.)
            self.fresh_coupling=HandCoupling(self.left_entity,self._manifest,stiffness=100.,damping=.02,driver_stiffness=20.,driver_damping=.2,hand_material=(.5,.5,0.),joint_friction=.001)
            self.hand_coupling_audit=self.fresh_coupling.audit()
            for side in ('left','right'):self._hand_sdk_target[side]=np.asarray(OPEN,dtype=float);setattr(self,side+'_gripper_val',1.)
    task=importlib.import_module('envs.adjust_bottle').adjust_bottle()
    # Retain the official method AST; only select the already allocated renderer.
    base=importlib.import_module('envs._base_task')
    tree=ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Base_Task']
    nodes=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='setup_scene']
    if len(nodes)!=1:raise RuntimeError('Unique official setup_scene method required')
    node=nodes[0];calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='sapien' and n.func.attr=='SapienRenderer']
    if len(calls)!=1 or calls[0].args or calls[0].keywords:raise RuntimeError('Official renderer construction AST changed')
    calls[0].keywords=[ast.keyword(arg='device',value=ast.Constant(value='cuda:0'))]
    local=dict(vars(base));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(base.__file__),'exec'),local)
    task.setup_scene=MethodType(local['setup_scene'],task)
    # Model directory is independently fixed by hardware configuration, not an episode.
    robot_dir=Path(guard['robot_model_directory'])
    cfg=yaml.safe_load((VENDOR/'config.yml').read_text(encoding='utf-8'))
    cfg['urdf_path']=guard['fixed_urdf_path'];cfg['homestate']=[RESET_LEFT.copy(),RESET_RIGHT.copy()];cfg['hand_joint_stiffness']=20.;cfg['hand_joint_damping']=.2
    # Keep manufacturer's scalar close preset; model execution never uses it.
    if cfg['hand_synergy']['open']!=OPEN:raise ValueError('Manufacturer open preset differs')
    args=json.loads((ROBOTWIN/'scripts/_task_config_template.json').read_text(encoding='utf-8'))
    args.update(task_name='adjust_bottle',dual_arm_embodied=True,embodiment_dis=0.,seed=int(seed),now_ep_num=0,left_robot_file=str(robot_dir),right_robot_file=str(robot_dir),
        left_embodiment_config=copy.deepcopy(cfg),right_embodiment_config=copy.deepcopy(cfg),embodiment=['realman-rm65b-inspire-gen4'],
        save_data=False,collect_data=False,save_freq=None,save_path=str(output),render_freq=0,eval_mode=True,need_plan=False,eval_video_save_dir=None,eval_video_log=False,
        domain_randomization=copy.deepcopy(DR),direction_lights=[[[0,.5,-1],[.5,.5,.5]]],point_lights=[[[1,0,1.8],[1,1,1]],[[-1,0,1.8],[1,1,1]]])
    args['data_type'].update(rgb=True,depth=False,pointcloud=False,mesh_segmentation=False,actor_segmentation=False)
    def forbidden(*a,**k):raise RuntimeError('Expert play_once/planner/trajectory is forbidden in fresh model evaluation')
    def load_robot(self,**kw):
        self.robot=FreshRobot(self.scene,False,**kw);self.robot.communication_flag=False;self.robot.init_joints()
        self._fresh_bootstrap_steps=0;self._fresh_original_step=self.scene.step
        self._immutable_sampler_checked=False
        def startup_step():
            if not self._immutable_sampler_checked and hasattr(self,'bottle'):
                self._immutable_sampler_checked=True;side=int(self.qpose_tag);model=int(self.model_id);reasons=[]
                if side!=(0 if scene_spec['arm']=='left' else 1):reasons.append('different_requested_side')
                if model!=requested_model_id:reasons.append('normal_sampler_different_requested_balance_model')
                if reasons:raise ImmutableSetupRejected({'seed':int(seed),'requested_model_id':int(requested_model_id),'actual_model_id':model,'actual_qpose_tag':side,'reasons':reasons,'bootstrap_physical_steps':int(self._fresh_bootstrap_steps),'rejection_stage':'actor_created_before_first_actor_physics_step','official_stability_check_run':False,'pre_settle_XY_admission_used':False,'actor_physics_steps':0,'model_queried':False})
            self._fresh_bootstrap_steps+=1
            return self._fresh_original_step()
        self.scene.step=startup_step
        # Official startup requests the already-open scalar; no trajectory generation.
        def stationary_open(robot,current,target):
            if abs(current-1.)>1e-9 or abs(target-1.)>1e-9:raise RuntimeError('Only native constant-open startup permitted')
            return {'per_step':0.,'result':np.ones(250),'num_step':250}
        for side in ('left','right'):setattr(self.robot,side+'_plan_grippers',MethodType(stationary_open,self.robot))
        for n in ('left_plan_path','right_plan_path','left_plan_multi_path','right_plan_multi_path'):setattr(self.robot,n,forbidden)
    task.load_robot=MethodType(load_robot,task)
    profile=yaml.safe_load((ROBOTWIN/'integrations/realman_rm65b_gen4/camera_profiles/three_d435_rgb_default.yml').read_text(encoding='utf-8'))
    args=attach_rgb_recording(task,args,profile)
    # Creation-time sampled pose only. No object state writes during the policy episode.
    def load_actors(self):
        import sapien,trimesh
        from transforms3d.quaternions import mat2quat
        from transforms3d.axangles import axangle2mat
        from envs.utils.create_actor import create_actor
        self.model_id=int(scene_spec['model_id']);self.qpose_tag=0 if scene_spec['arm']=='left' else 1
        data=json.loads((ROBOTWIN/f'assets/objects/001_bottle/model_data{self.model_id}.json').read_text())
        mesh=trimesh.load(ROBOTWIN/f'assets/objects/001_bottle/collision/base{self.model_id}.glb',force='scene').dump(concatenate=True)
        vertices=np.asarray(mesh.vertices)*np.asarray(data['scale']);center=(vertices.min(0)+vertices.max(0))/2
        yaw=np.deg2rad(float(scene_spec['axis_yaw_deg']))
        if scene_spec['pose_class']=='upright':R=axangle2mat([0,0,1],yaw)@axangle2mat([1,0,0],np.pi/2)
        else:R=axangle2mat([0,0,1],yaw-np.pi/2)
        origin=np.zeros(3);origin[:2]=np.asarray(scene_spec['center_xy'])-(R@center)[:2]
        origin[2]=.74-float((vertices@R.T)[:,2].min())+.002
        saved=np.asarray(scene_spec['source_initial_pose']); initial_pose=sapien.Pose(saved[:3,3],mat2quat(saved[:3,:3]))
        self.bottle=create_actor(self,initial_pose,modelname='001_bottle',convex=True,model_id=self.model_id)
        from sapien.physx import PhysxRigidDynamicComponent
        self.bottle.actor.find_component_by_type(PhysxRigidDynamicComponent).put_to_sleep()
        self.delay(4);self.add_prohibit_area(self.bottle,padding=.15)
        self.left_target_pose=[-.25,-.12,.95,0,1,0,0];self.right_target_pose=[.25,-.12,.95,0,1,0,0]
    task.load_actors=MethodType(load_actors,task)
    task.play_once=forbidden
    try:task.setup_demo(**args)
    except Exception:
        if hasattr(task,'close_env'):task.close_env()
        raise
    task.scene.step=task._fresh_original_step
    for n in ('move','grasp_actor','place_actor','left_move_to_pose','right_move_to_pose'):setattr(task,n,forbidden)
    contract=read_physical_contract(task)
    if len(contract['drives'])!=36 or any(abs(s['contact_offset_m']-.01)>1e-8 or abs(s['rest_offset_m'])>1e-8 for s in contract['shapes']):raise RuntimeError('Actual native default10mm/rest0 shape configuration differs')
    driver=set(n for h in task.robot._manifest['hands'].values() for n in h['channels']);followers=set(n for h in task.robot._manifest['hands'].values() for n in h['mimic'])
    for d in contract['drives']:
        if d['joint'] in driver and (abs(d['stiffness']-20.)>1e-8 or abs(d['damping']-.2)>1e-6):raise RuntimeError('Actual primary PD differs')
        if d['joint'] in followers and (d['stiffness']!=0 or d['damping']!=0 or d['force_limit']!=0):raise RuntimeError('Actual follower passive drive differs')
    p=task.bottle.get_pose().to_transformation_matrix()
    # Expert diagnostic intentionally uses the original training seed.
    environment={'eval_mode':True,'domain_randomization':copy.deepcopy(DR),'wall_texture':task.wall_texture,'table_texture':task.table_texture,'random_background':False,'random_light':False}
    evidence={'seed':int(seed),'sample_spec':scene_spec,'actual_model_id':int(task.model_id),'actual_qpose_tag':int(task.qpose_tag),'actual_initial_actor_world_pose':p.tolist(),'fresh_seed_absent_from_all50_train':False,'environment':environment,'initial_full36_qpos_rad':np.asarray(task.robot.left_entity.get_qpos()).tolist(),'physical_contract':contract,'fixed_reset':guard['fixed_reset'],'fixed_reset_provenance':guard['fixed_reset_provenance'],'bootstrap_original_physics_delay_and_stability':True,'bootstrap_physical_steps':task._fresh_bootstrap_steps,'HDF_episode_state_read':False,'expert_actions_or_grasp_parameters_read':False,'planner_invocations':0,'hardware_io':False,'camera_profile':profile,'renderer_device':{'requested_device':'cuda:0','CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')},'fixed_head_configuration':guard['head_configuration'],'contact_numerical_profile':{'selected_all_robot_shapes_m':.01,'rest_m':0.,'hardware_calibrated':False},'hardware_guard_sha256':sha(ROOT/'hardware_guard.json')}
    return task,args,evidence
