"""Independent Base_Task labware scene; simulation only, SDK24/60-ms clock.

The original RoboTwin task/model and frozen imported assets are read-only.
No object pose is written after construction; manipulation is contact driven.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parent
ROBOTWIN = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
NATIVE = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin')
ASSETS = ROBOTWIN/'assets'
IMPORTED = ROOT.parent/'2026-10-22-labware-integration/prepared'
# The frozen successful native stack is selected before any envs imports.
if 'envs._base_task' not in sys.modules:
    sys.path.insert(0, str(NATIVE))
_import_cwd=os.getcwd()
os.chdir(ROBOTWIN)
from envs._base_task import Base_Task
from integrations.realman_rm65b_gen4.native_robot import RealManRobot
from integrations.realman_rm65b_gen4.rl_contract import RESET_LEFT, RESET_RIGHT, OPEN_SDK6, FULL50_PHYSICS
from integrations.realman_rm65b_gen4.hand_coupling import HandCoupling
from integrations.realman_rm65b_gen4.three_camera_rgb import MountedD435RGBCamera
from integrations.realman_rm65b_gen4.rl_factory import derive_urdf
from integrations.realman_rm65b_gen4.action_clock import ClockedTask24Adapter
os.chdir(_import_cwd)


class LabwareRobot(RealManRobot):
    def set_planner(self, scene=None):
        # The expert creates an independent CPU full-mesh planner when needed.
        self.left_planner = self.right_planner = None

    @staticmethod
    def _stationary_open(current,target):
        if not np.isclose(current,1.) or not np.isclose(target,1.):
            raise RuntimeError('Bootstrap only supports fixed open hand')
        return {'per_step':0.,'result':np.ones(250),'num_step':250}

    def left_plan_grippers(self,current,target):return self._stationary_open(current,target)
    def right_plan_grippers(self,current,target):return self._stationary_open(current,target)

    def _initialize_hands(self):
        super()._initialize_hands()
        q = np.asarray(self.left_entity.get_qpos()).copy()
        for side, values in [('left', RESET_LEFT), ('right', RESET_RIGHT)]:
            for name, value in zip(getattr(self, side+'_arm_joints_name'), values):
                q[self._joint_index[name]] = value
            for name, value in self._hand_maps[side].from_sdk(OPEN_SDK6, dict(zip(self._hand_joints, q))).items():
                q[self._joint_index[name]] = value
        self.left_entity.set_qpos(q)
        self.left_entity.set_qvel(np.zeros_like(q))
        for name, joint in self._hand_joints.items():
            joint.set_drive_target(float(q[self._joint_index[name]]))
            joint.set_drive_velocity_target(0.)
        self.fresh_coupling = HandCoupling(self.left_entity, self._manifest, **FULL50_PHYSICS)
        self.hand_coupling_audit = self.fresh_coupling.audit()
        for side in ('left','right'):
            self._hand_sdk_target[side] = np.array(OPEN_SDK6, dtype=float)
            setattr(self, side+'_gripper_val',1.)


class labware_pick_place(Base_Task):
    """Agent play_once operates on bottle; extension helper performs native grasp.

    Current minimum is an empty open bottle, not the screw-cap/tablet task.
    Success requires actual lift+hold, release, stable placement and hand retreat.
    """
    def setup_demo(self, **kwargs):
        import yaml
        self.runtime_path = Path(kwargs.pop('runtime_path', ROOT/'task_probes/default'))
        self.runtime_path.mkdir(parents=True, exist_ok=True)
        self.capture_rgb = kwargs.pop('capture_rgb', False)
        self.asset_kind = kwargs.pop('asset_kind','reagent_bottle_body')
        self.replay_commands=kwargs.pop('replay_commands',None)
        self.center_xy = kwargs.pop('center_xy', None)
        self.yaw = float(kwargs.pop('yaw',0.))
        self.arm = kwargs.pop('arm','right')
        self.grasp_z = float(kwargs.pop('grasp_z', .045))
        self.on_transition = kwargs.pop('on_transition', None)
        self.config = SimpleNamespace(max_episode_steps=1200,hold_seconds=.5,min_lift_m=.02,max_upright_angle_deg=15.)
        self.task = self
        self.physics_step_count = 0
        self.metrics = {}
        self.max_hold_seconds = 0.
        self.lift_achieved = False
        self.release_requested = False
        self.retreat_completed = False
        self.completed_stable_ticks = 0
        self.hold_ticks = 0
        self.physics_trace = []
        self.transition_log = []
        self._active = False
        self.workspace = self.runtime_path/'workspace'
        self.workspace.mkdir(exist_ok=True)
        if not (self.workspace/'assets').exists():
            (self.workspace/'assets').symlink_to(ASSETS, target_is_directory=True)
        os.chdir(self.workspace)
        vendor = ASSETS/'embodiments/realman-rm65b-inspire-gen4'
        derived = self.runtime_path/'robot_head_down.urdf'
        self.robot_source_receipt = derive_urdf(vendor/'robot_sapien.urdf',derived)
        cfg = yaml.safe_load((vendor/'config.yml').read_text())
        cfg.update(urdf_path=str(derived), homestate=[list(RESET_LEFT),list(RESET_RIGHT)])
        profile = yaml.safe_load((NATIVE/'integrations/realman_rm65b_gen4/camera_profiles/three_d435_rgb_default.yml').read_text())
        args = json.loads((NATIVE/'scripts/_task_config_template.json').read_text())
        args.update(task_name='labware_pick_place', seed=kwargs.pop('seed',0), timestep=.004,
            left_robot_file=str(vendor),right_robot_file=str(vendor),dual_arm_embodied=True,
            left_embodiment_config=copy.deepcopy(cfg), right_embodiment_config=copy.deepcopy(cfg),
            embodiment=['realman-rm65b-inspire-gen4'], save_data=False,collect_data=False,save_freq=None,
            save_path=str(self.runtime_path),render_freq=0,eval_mode=True,need_plan=False,
            eval_video_save_dir=None,eval_video_log=False,
            rgb_camera_profile=profile['rgb_profile'],rgb_camera_mounts=profile['mounts'])
        args['camera']={'head_camera_type':'Large_D435','wrist_camera_type':'Large_D435',
            'collect_head_camera':True,'collect_wrist_camera':True}
        args['data_type'].update(rgb=True,qpos=True,endpose=False,depth=False,pointcloud=False,
            mesh_segmentation=False,actor_segmentation=False,third_view=False,observer=False)
        args['domain_randomization'].update(random_background=False,cluttered_table=False,
            clean_background_rate=1.,random_head_camera_dis=0.,random_table_height=0.,random_light=False,
            crazy_random_light_rate=0.,random_embodiment=False)
        args.update(kwargs)
        self.seed=int(args['seed'])
        super()._init_task_env_(**args)
        self.step_lim=self.config.max_episode_steps
        self.set_instruction('Pick up the open reagent bottle, hold it upright, and place it back at its original spot.')
        self.info['info']={'{A}':str(ROOT/'metadata/description/objects_description'/self.asset_kind/'base0'),'{a}':self.arm}
        self.initial_pose=np.array(self.bottle.get_pose().to_transformation_matrix())
        self.target_xy=self.initial_pose[:2,3].copy()
        self._raw_step=self.scene.step
        self.clock=ClockedTask24Adapter(self,self,stop_on_task_success=False)
        self._active=True
        self._update_metrics()
        return self.info

    reset = setup_demo

    def setup_scene(self,**kwargs):
        import sapien
        from sapien.render import set_global_config
        set_global_config(max_num_materials=50000,max_num_textures=50000)
        self.engine=sapien.Engine()
        self.renderer=sapien.SapienRenderer(device='cuda:0')
        self.engine.set_renderer(self.renderer)
        sapien.render.set_camera_shader_dir('rt')
        sapien.render.set_ray_tracing_samples_per_pixel(16)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser('none')
        self.scene=self.engine.create_scene(sapien.SceneConfig())
        self.scene.set_timestep(.004)
        self.scene.add_ground(0.)
        self.scene.default_physical_material=self.scene.create_physical_material(.5,.5,0.)
        self.scene.set_ambient_light([.5]*3)
        self.direction_light_lst=[self.scene.add_directional_light([0,.5,-1],[.5]*3,shadow=True)]
        self.point_light_lst=[self.scene.add_point_light(p,[1]*3,shadow=True) for p in ([1,0,1.8],[-1,0,1.8])]

    def load_robot(self,**kwargs):
        self.robot=LabwareRobot(self.scene,False,**kwargs)
        self.robot.communication_flag=False
        self.robot.init_joints()
        for link in self.robot.left_entity.get_links():
            for shape in link.get_collision_shapes():
                shape.set_contact_offset(.01)
                shape.set_rest_offset(0.)

    def load_camera(self,**kwargs):
        self.cameras=MountedD435RGBCamera(bias=self.table_z_bias,random_head_camera_dis=0,**kwargs)
        self.cameras.load_camera(self.scene)
        self._step_simulation()
        self.scene.update_render()

    def load_actors(self):
        import sapien
        import trimesh
        from envs.utils.create_actor import create_actor
        registry=json.loads((ROOT/'metadata/registry.json').read_text())
        entry=registry['assets'][self.asset_kind]
        folder=Path(entry['absolute_modelname'])
        self.asset_data=json.loads((folder/'model_data0.json').read_text())
        self.qpose_tag=1 if self.arm=='right' else 0
        self.model_id=0
        if self.center_xy is None:
            sign=1 if self.arm=='right' else -1
            self.center_xy=[sign*np.random.uniform(.095,.105),np.random.uniform(-.115,-.105)]
        self.bottle=create_actor(self,sapien.Pose([*self.center_xy,.742],
            [np.cos(self.yaw/2),0,0,np.sin(self.yaw/2)]),modelname=str(folder),convex=True,model_id=0)
        mass=float(self.asset_data['annotation_provenance']['mass_kg'])
        self.bottle.set_mass(mass)
        mesh=trimesh.load(folder/'collision/base0.glb',force='scene').dump(concatenate=True)
        self.vertices=np.asarray(mesh.vertices)*np.asarray(self.asset_data['scale'])
        self.delay(1)
        self.add_prohibit_area(self.bottle,padding=.1)

    @property
    def step(self): return self.physics_step_count

    def advance(self):
        self._raw_step()
        self.physics_step_count+=1
        self._update_metrics()

    def _update_metrics(self):
        if not self._active: return
        matrix=np.asarray(self.bottle.get_pose().to_transformation_matrix())
        bottom=float((self.vertices@matrix[:3,:3].T+matrix[:3,3])[:,2].min())
        gap=bottom-.74
        angle=float(np.degrees(np.arccos(np.clip(matrix[2,2],-1,1))))
        thumb=fingers=table=0.
        normals={'thumb':np.zeros(3),'fingers':np.zeros(3)}
        prefix='r_' if self.arm=='right' else 'l_'
        for contact in self.scene.get_contacts():
            names=[b.entity.name for b in contact.bodies]
            if self.bottle.actor.name not in names: continue
            first=names[0]==self.bottle.actor.name
            other=names[1] if first else names[0]
            for point in contact.points:
                force=abs(float(np.asarray(point.impulse)@np.asarray(point.normal)))/.004
                if 'table' in other.lower(): table+=force
                if not other.startswith(prefix+'inspire_'):continue
                group='thumb' if 'thumb' in other else 'fingers'
                if group=='thumb':thumb+=force
                else:fingers+=force
                normals[group]+=force*np.asarray(point.normal)*(1 if first else -1)
        denom=np.linalg.norm(normals['thumb'])*np.linalg.norm(normals['fingers'])
        dot=float(normals['thumb']@normals['fingers']/denom) if denom>1e-12 else 1.
        opposed=thumb>.05 and fingers>.05 and dot<-.3
        hold=opposed and gap>=self.config.min_lift_m and table<1e-5 and angle<15
        self.hold_ticks=self.hold_ticks+1 if hold else 0
        self.max_hold_seconds=max(self.max_hold_seconds,self.hold_ticks*.004)
        self.lift_achieved=self.lift_achieved or self.max_hold_seconds>=self.config.hold_seconds
        component=next(c for c in self.bottle.actor.get_components() if hasattr(c,'linear_velocity'))
        speed=float(np.linalg.norm(component.linear_velocity))
        placed=bool(self.lift_achieved and self.release_requested and self.retreat_completed
            and abs(gap)<.003 and angle<10 and table>.01 and thumb+fingers<.02
            and np.linalg.norm(matrix[:2,3]-self.target_xy)<.025 and speed<.015)
        self.completed_stable_ticks=self.completed_stable_ticks+1 if placed else 0
        self.metrics=dict(success=self.completed_stable_ticks>=125,bottom_gap_m=gap,upright_angle_deg=angle,
            thumb_contact_N=thumb,finger_contact_N=fingers,table_contact_N=table,opposed_contact=opposed,
            opposing_normals_dot=dot,hold_seconds=self.hold_ticks*.004,max_hold_seconds=self.max_hold_seconds,
            lift_achieved=self.lift_achieved,speed_m_s=speed,stable_place_seconds=self.completed_stable_ticks*.004,
            failure_reason='below_table' if bottom<.64 else None)
        if self.physics_step_count%15==0:
            self.physics_trace.append(dict(tick=self.physics_step_count,pose=matrix.tolist(),**self.metrics))

    def capture(self):
        result={'state':np.asarray(self.robot.get_state24(True),dtype=np.float32),
            'instruction':self.instruction,'physics_tick':self.physics_step_count}
        if self.capture_rgb:
            obs=super().get_obs()
            result.update({name:np.asarray(obs['observation'][camera]['rgb']).copy() for name,camera in
                [('full_image','head_camera'),('left_wrist_image','left_camera'),('right_wrist_image','right_camera')]})
        self.last_observation=result
        return result

    def apply_sdk24(self,action,phase='unspecified'):
        before=self.capture()
        record=self.clock.take_action(action)
        after=self.last_observation
        row={'phase':phase,'state':before['state'].tolist(),'action':np.asarray(action).tolist(),
            'next_state':after['state'].tolist(),'tick_start':before['physics_tick'],
            'tick_end':after['physics_tick'],'metrics':copy.deepcopy(self.metrics),
            'release_requested':self.release_requested,'retreat_completed':self.retreat_completed}
        self.transition_log.append(row)
        if self.on_transition:self.on_transition(before,np.asarray(action).copy(),after,row)
        return record

    def check_success(self):return bool(self.metrics.get('success',False))

    def get_state24(self):
        return np.asarray(self.robot.get_state24(True),dtype=np.float32)

    def observe_rgb(self):
        previous=self.capture_rgb
        self.capture_rgb=True
        try:return self.capture()
        finally:self.capture_rgb=previous

    def execute_command24(self,target24,phase='replay'):
        return self.apply_sdk24(target24,phase)

    def grasp_lift_place(self,actor=None,arm_tag='right'):
        from robot_expert import run_contact_pick_place
        if actor is not None and actor is not self.bottle:raise ValueError('Only the registered bottle is supported')
        if str(arm_tag)!=self.arm:raise ValueError('Arm must match scene side')
        if self.replay_commands is not None:
            for record in self.replay_commands:
                if isinstance(record,dict):
                    self.release_requested=record.get('release_requested',False)
                    self.retreat_completed=record.get('retreat_completed',False)
                    self.apply_sdk24(record['action'],record.get('phase','replay'))
                else:self.apply_sdk24(record,'replay')
            return self.info
        return run_contact_pick_place(self)

    def play_once(self):
        self.grasp_lift_place(self.bottle,self.arm)
        return self.info

    def close(self):
        if getattr(self,'scene',None) is not None and getattr(self,'_raw_step',None) is not None:self.scene.step=self._raw_step
        self._active=False
        for name in ('clock','teacher','robot','cameras','bottle','table','wall','scene','engine','renderer'):
            if hasattr(self,name):setattr(self,name,None)
        self._raw_step=None
        import gc
        gc.collect()


def create_task(task_name='labware_pick_place',runtime_config=None,run_dir=None,**kwargs):
    """Official agent/collector factory: returns uninitialized task and kwargs.

    Registered generated class is selected when available; otherwise native
    reference play_once is used and provenance must be labelled accordingly.
    """
    options=dict(runtime_config or {})
    options.update(kwargs)
    if run_dir is not None:options.setdefault('runtime_path',str(Path(run_dir)/'simulation'))
    options.setdefault('capture_rgb',True)
    cls=labware_pick_place
    try:
        from envs_gen import gpt_labware_pick_place
        candidate=getattr(gpt_labware_pick_place,'gpt_'+task_name,None)
        if candidate is not None:cls=candidate
    except ImportError:pass
    task=cls()
    return task,options
