"""Only own actual RGB is rendered; no source cache/video reader."""
from pathlib import Path
import hashlib,json
import numpy as np

class ActualRGBRecorder:
    def __init__(self,task,output):
        self.task=task;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.rows=[];self.sequence=0
    def __call__(self,step,sim_time_s,images,measured):
        self.sequence+=1;name=f'observation_{self.sequence:06d}_step_{step:06d}.npz'
        poses={key:cam.entity.get_pose().to_transformation_matrix() for key,cam in self.task.cameras.camera_by_name.items()}
        for key,value in images.items():
            if np.asarray(value).dtype!=np.uint8 or np.asarray(value).shape!=(480,640,3):raise ValueError('Actual three RGB frames required')
        f=self.output/name
        np.savez_compressed(f,**images,state24=np.asarray(measured),full36_actual_rad=np.asarray(self.task.robot.left_entity.get_qpos()),
            physics_step=np.asarray(step),sim_time_s=np.asarray(sim_time_s),**{key+'_world_GL_pose':value for key,value in poses.items()})
        self.rows.append({'observation_index':self.sequence,'physics_step':step,'sim_time_s':float(sim_time_s),'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    def capture(self,step,gates):
        obs=self.task.get_obs();state=np.asarray(self.task.robot.get_state24(True))
        gates.observe_same_clock(step,float(self.task.scene.get_timestep()),obs,state)
        images={out:obs['observation'][name]['rgb'] for out,name in [('cam_head','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]}
        self(step,step*float(self.task.scene.get_timestep()),images,state)
        return {'images':images,'state':state,'sim_time_s':step*float(self.task.scene.get_timestep())}
    def write_manifest(self):
        p=self.output/'observations.json';p.write_text(json.dumps({'scope':'Own actual native RGB/state at recorded physical clock; not source trajectory playback','frames':self.rows},indent=2),encoding='utf-8')
        return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'observations':len(self.rows),'camera_keys':['cam_head','cam_left_wrist','cam_right_wrist']}
