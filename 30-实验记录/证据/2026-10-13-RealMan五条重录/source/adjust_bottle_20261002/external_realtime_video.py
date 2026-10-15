"""Streaming external review video: one actual rendered frame per10physics steps.
This is an observer camera at the existing snapshot eye, not a vendor head view.
No scene stepping, arm/hand changes, resampling or generated frames are performed.
"""
from pathlib import Path
import math
import numpy as np


class ExternalRealtimeVideo:
    def __init__(self,task,path,*,pose_factory,quat_converter,cv2_module=None):
        if cv2_module is None:import cv2 as cv2_module
        self.cv2=cv2_module;self.task=task;self.path=Path(path);self.dt=float(task.scene.get_timestep())
        self.stride=10;self.fps=25.;self.width=640;self.height=480;self.steps=[];self.closed=False;self.result=None
        if not math.isclose(self.stride*self.dt,1/self.fps,abs_tol=1e-8):raise ValueError('10 physics steps must equal a25fps frame interval')
        if self.path.exists():raise ValueError('Refusing to overwrite observer video')
        self.writer=self.cv2.VideoWriter(str(self.path),self.cv2.VideoWriter_fourcc(*'mp4v'),self.fps,(self.width,self.height))
        if not self.writer.isOpened():self.writer.release();raise RuntimeError('Observer MP4 encoder did not open')
        try:
            self.camera=task.scene.add_camera('accepted_replay_external_review',self.width,self.height,math.radians(55),.005,10)
            self.eye=np.array([1.35,.28,1.65]);self.target=np.array([0.,-.36,.9]);forward=self.target-self.eye;forward/=np.linalg.norm(forward)
            left=np.cross([0,0,1.],forward);left/=np.linalg.norm(left);rotation=np.stack([forward,left,np.cross(forward,left)],axis=1)
            self.camera.entity.set_pose(pose_factory(self.eye,quat_converter(rotation)))
        except BaseException:self.writer.release();raise

    def capture(self,physics_step):
        if self.closed:raise RuntimeError('Observer video is closed')
        step=int(physics_step)
        if step%self.stride:return
        expected=0 if not self.steps else self.steps[-1]+self.stride
        if step!=expected:raise RuntimeError('Observer frames must follow t0,step10,20,... without gaps')
        self.task._update_render();self.camera.take_picture()
        rgb=np.asarray(self.camera.get_picture('Color'))[...,:3]
        if rgb.shape!=(self.height,self.width,3) or not np.isfinite(rgb).all():raise RuntimeError('Observer frame is invalid')
        bgr=np.ascontiguousarray((np.clip(rgb,0,1)*255).astype(np.uint8)[...,::-1]);self.writer.write(bgr);self.steps.append(step)

    def close(self,total_physics_steps,*,validate=True):
        if self.closed:return self.result
        self.closed=True;self.writer.release();n=int(total_physics_steps)
        result={'path':str(self.path),'purpose':'external observer review camera, not the vendor head camera',
                'eye_world_m':self.eye.tolist(),'target_world_m':self.target.tolist(),'vertical_fov_deg':55.,
                'width':self.width,'height':self.height,'encoded_fps':self.fps,'physics_stride':self.stride,
                'physics_dt_s':self.dt,'frame_physics_steps':self.steps,'frame_times_s':[step*self.dt for step in self.steps],
                'frames':len(self.steps),'physics_duration_s':n*self.dt,'encoded_duration_s':len(self.steps)/self.fps,
                'trailing_unsampled_physics_steps':n%self.stride,'captured_initial_t0':bool(self.steps and self.steps[0]==0)}
        self.result=result
        if validate:
            if self.steps!=list(range(0,n+1,self.stride)):raise RuntimeError('Observer video timing coverage is incomplete')
            decoder=self.cv2.VideoCapture(str(self.path))
            try:
                if not decoder.isOpened():raise RuntimeError('Observer MP4 readback failed')
                count=int(round(decoder.get(self.cv2.CAP_PROP_FRAME_COUNT)));fps=float(decoder.get(self.cv2.CAP_PROP_FPS))
                width=int(decoder.get(self.cv2.CAP_PROP_FRAME_WIDTH));height=int(decoder.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
                result['encoded_readback']={'frames':count,'fps':fps,'width':width,'height':height}
                if count!=len(self.steps) or not math.isclose(fps,self.fps,abs_tol=1e-4) or (width,height)!=(self.width,self.height):
                    raise RuntimeError('Observer encoded frame count/fps/dimensions do not match captured physics timing')
            finally:decoder.release()
        return result
