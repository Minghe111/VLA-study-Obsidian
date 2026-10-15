"""Same three-camera/physical observer plus an independent display camera.

The fixed display camera is never present in observations returned to policy.
The callback runs on real new physics frames; stage updates carry no images.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from observations40k import Observations

class Observations80k(Observations):
    def __init__(self,task,output,capture_hook=None,*,external_camera=True):
        self.capture_hook=capture_hook;self.external_camera=None;self.live_last_step=-1
        self.first_native_success_step=None
        self.external_rows=[];self.external_output=Path(output)/'observer_rgb'
        if external_camera:
            import sapien
            from transforms3d.quaternions import mat2quat
            self.external_output.mkdir(parents=True,exist_ok=False)
            self.external_camera=task.scene.add_camera('display_only_fixed_observer',640,480,np.deg2rad(55.),.01,10.)
            eye=np.asarray([1.30,1.60,1.60]);target=np.asarray([0.,-.32,1.0]);forward=target-eye;forward/=np.linalg.norm(forward)
            left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left);up=np.cross(forward,left)
            pose=sapien.Pose(eye,mat2quat(np.stack([forward,left,up],axis=1)))
            self.external_camera.entity.set_pose(pose)
            self.external_contract={'scope':'Fixed observer camera only for viewing, not model inputs','world_entity_pose':pose.to_transformation_matrix().tolist(),
                                    'image_width':640,'image_height':480,'fovy_deg':55.,'camera_moves_during_episode':False}
        super().__init__(task,Path(output))
    def capture(self):
        obs=super().capture()
        if self.step!=self.live_last_step:
            display_images=dict(obs['images'])
            if self.external_camera is not None:
                self.external_camera.take_picture()
                rgb=(self.external_camera.get_picture('Color')[:,:,:3]*255).clip(0,255).astype(np.uint8)
                display_images['external']=rgb
                p=self.external_output/f'frame_{len(self.external_rows):06d}_step_{self.step:06d}.npz'
                np.savez_compressed(p,external=rgb,physics_step=np.asarray(self.step),sim_time_s=np.asarray(self.step*self.dt))
                self.external_rows.append({'physics_step':int(self.step),'sim_time_s':float(self.step*self.dt),'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            if self.capture_hook is not None:self.capture_hook(dict(obs,display_images=display_images),self,self.task)
            self.live_last_step=self.step
        return obs
    def finish(self):
        result=super().finish()
        if self.external_camera is not None:
            p=self.external_output/'observations.json';p.write_text(json.dumps({'camera':self.external_contract,'frames':self.external_rows},indent=2)+'\n')
            result['display_only_external_observer']={'manifest':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'frames':len(self.external_rows),'included_in_policy_input':False}
        return result
    def advance(self):
        super().advance()
        # Pure pose-based native success observation. This does not set flags,
        # change motion, or add a stop condition; the current adapter checks it.
        if self.first_native_success_step is None and self.task.check_success():
            self.first_native_success_step=int(self.step)
    def summary(self):
        result=super().summary()
        axis=np.asarray(self.task.bottle.get_pose().to_transformation_matrix())[:3,1]
        result.update(first_native_success_step=self.first_native_success_step,
                      first_native_success_sim_time_s=None if self.first_native_success_step is None else self.first_native_success_step*self.dt,
                      terminal_upright_error_deg=float(np.degrees(np.arccos(np.clip(axis[2]/np.linalg.norm(axis),-1,1)))))
        return result
