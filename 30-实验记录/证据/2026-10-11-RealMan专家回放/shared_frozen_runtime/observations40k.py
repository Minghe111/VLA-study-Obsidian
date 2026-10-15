"""Native Pi_05 deployment loop; only a 24D RealMan embodiment extension."""
import argparse,gzip,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scene40k import ROOT,ROBOTWIN,build_fresh_scene,sha
from client40k import LocalPolicyClient
from actual_rgb_recorder_v1 import ActualRGBRecorder
from fresh_contacts_v1 import contacts_and_grip,real_grip
from fresh_support_v1 import native_tabletop

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

class Observations:
    """Read actual state only. No success, collision, or motion gate."""
    def __init__(self,task,output):
        from full_mesh_path_audit import _link_vertices
        from sapien.physx import PhysxRigidBaseComponent
        self.task=task;self.step=0;self.dt=float(task.scene.get_timestep())
        self.vertices,_=_link_vertices(next(c for c in task.bottle.actor.get_components() if hasattr(c,'collision_shapes')))
        table=next(a for a in task.scene.get_all_actors() if a.name=='table')
        self.table=native_tabletop(table.find_component_by_type(PhysxRigidBaseComponent))
        self.top=self.table['table_top_z'];self.initial=np.asarray(task.bottle.get_pose().to_transformation_matrix())
        self.trace_path=output/'actual_physical_trace.jsonl.gz'
        self.trace=gzip.open(self.trace_path,'wt',encoding='utf-8')
        self.recorder=ActualRGBRecorder(task,output/'actual_rgb')
        self.max_gap=None;self.max_xy=0.;self.max_rotation=0.;self.air_run=0;self.max_air_run=0
        self.first_contact=None;self.first_grip=None;self.first_cm=None;self.grip_samples=0
        self.raw_step=task.scene.step
        self.capture()
    def capture(self):
        obs=self.task.get_obs();state=np.asarray(self.task.robot.get_state24(True))
        images={out:obs['observation'][name]['rgb'] for out,name in [('cam_head','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]}
        if not self.recorder.rows or self.recorder.rows[-1]['physics_step']!=self.step:
            self.recorder(self.step,self.step*self.dt,images,state)
        return {'images':images,'state':state,'sim_time_s':self.step*self.dt}
    def advance(self):
        self.raw_step();self.step+=1
        task=self.task;p=np.asarray(task.bottle.get_pose().to_transformation_matrix())
        v=self.vertices@p[:3,:3].T+p[:3,3];gap=float(v[:,2].min()-self.top)
        g=contacts_and_grip(task);held=real_grip(g)
        xy=float(np.linalg.norm(p[:2,3]-self.initial[:2,3]))
        theta=float(np.arccos(np.clip((np.trace(self.initial[:3,:3].T@p[:3,:3])-1)/2,-1,1)))
        self.max_gap=gap if self.max_gap is None else max(self.max_gap,gap);self.max_xy=max(self.max_xy,xy);self.max_rotation=max(self.max_rotation,theta)
        if g['thumb_normal_N']+g['fingers_normal_N']>0 and self.first_contact is None:self.first_contact=self.step
        if held:
            self.grip_samples+=1
            if self.first_grip is None:self.first_grip=self.step
        if gap>=.01 and self.first_cm is None:self.first_cm=self.step
        air=held and gap>=.01 and g['table_normal_N']<=1e-5
        self.air_run=self.air_run+1 if air else 0;self.max_air_run=max(self.max_air_run,self.air_run)
        row={'step':self.step,'sim_time_s':self.step*self.dt,'actual_state24':task.robot.get_state24(True),
             'actual_qpos36_rad':np.asarray(task.robot.left_entity.get_qpos()).tolist(),
             'actual_drive36_rad':[float(j.get_drive_target()[0]) for j in task.robot._hand_joints.values()],
             'bottle_world_pose':p.tolist(),'functional_point':np.asarray(task.bottle.get_functional_point(0)).tolist(),
             'native_bottom_gap_m':gap,'initial_xy_displacement_m':xy,'initial_rotation_rad':theta,
             'contact_observation':g,'opposed_contact_observed':held}
        self.trace.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        if self.step%15==0:self.capture()
    def finish(self):
        self.capture();self.trace.close()
        return self.recorder.write_manifest()
    def summary(self):
        return {'observation_only_did_not_control_actions_or_termination':True,
                'maximum_native_bottom_gap_m':self.max_gap,'maximum_initial_xy_displacement_m':self.max_xy,
                'maximum_initial_rotation_deg':self.max_rotation*180/np.pi,
                'first_positive_hand_contact_step':self.first_contact,'first_opposed_contact_step':self.first_grip,
                'first_bottom_gap_at_least_1cm_step':self.first_cm,'opposed_contact_samples':self.grip_samples,
                'longest_observed_air_grasp_s':self.max_air_run*self.dt,
                'user_air_grasp_observed_1cm_1s':self.max_air_run*self.dt>=1.,
                'native_table_geometry':self.table}

class PolicyClientAdapter:
    def __init__(self,client,prompt,output):self.client=client;self.prompt=prompt;self.output=output;self.obs=None
    def call(self,func_name,**kwargs):
        if func_name=='reset':self.obs=None;return
        if func_name=='update_obs':self.obs=kwargs['obs'];return
        if func_name=='get_action':
            result=self.client(dict(self.obs,prompt=self.prompt));raw=np.asarray(result['actions'])
            np.savez_compressed(self.output/f'query_{len(self.client.records):05d}.npz',raw_prediction24=raw)
            return raw
        raise ValueError(func_name)

