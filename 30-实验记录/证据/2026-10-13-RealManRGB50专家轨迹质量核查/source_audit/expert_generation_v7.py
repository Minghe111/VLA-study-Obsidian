"""v6 control with the official two-getter place_actor distance contract.

Preserve both the pre_dis=0 pre-place goal and default dis=.02 final goal.
Distance vectors follow Base_Task.get_place_pose exactly; the minimum upright
rotation replaces only its azimuth choice. No object/robot state is written.
"""
from pathlib import Path
import copy
import math
import numpy as np
from transforms3d.quaternions import quat2mat

from expert_generation_v6 import (ExpertSession as _BaseSession,
                                 install_initializer,command24,measured24,_sha)
from expert_generation_v5 import rigid_functional_point_target


def official_pre_dis_vector(live_ee_pose,bottle_matrix,rigid_rotation,
                            target_pose,pre_dis_axis='grasp'):
    """Same vector as Base_Task: target_grasp_R @ inv(live_R) @ grasp_bias."""
    if isinstance(pre_dis_axis,str) and pre_dis_axis=='grasp':
        vector=rigid_rotation@(np.asarray(bottle_matrix)[:3,3]-np.asarray(live_ee_pose)[:3])
    else:
        axis=np.array([0.,0.,1.]) if isinstance(pre_dis_axis,str) and pre_dis_axis=='fp' else np.asarray(pre_dis_axis,float)
        vector=quat2mat(np.asarray(target_pose)[-4:])@axis
    norm=np.linalg.norm(vector)
    if not np.isfinite(norm) or norm<1e-12:raise ValueError('Undefined official pre_dis direction')
    return vector/norm


class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        source=Path(__file__).resolve();self.sources[str(source)]=_sha(source)
        self.placement_batches=[];self.previous_place_selection=None
        self.report.update(expert_version='v7',source_hashes=self.sources,
            placement_interface='official place_actor pre_dis=0 and final dis=.02; distinct candidate batches, selected yaw retained')
        self.write()

    def place_pose(self,actor,arm_tag,target_pose,**kwargs):
        if actor is not self.task.bottle or str(arm_tag)!=self.side:
            raise RuntimeError('Unexpected bottle/arm in placement')
        if kwargs.get('functional_point_id')!=0:raise RuntimeError('adjust_bottle requires functional point 0')
        pre_dis=float(kwargs.get('pre_dis',.1))
        if not math.isfinite(pre_dis) or pre_dis<0.:raise ValueError('Invalid official placement distance')
        live=np.asarray(getattr(self.task.robot,'get_'+self.side+'_ee_pose')(),float)
        matrix=actor.get_pose().to_transformation_matrix();current=np.asarray(actor.get_functional_point(0))
        preferred=float(self.spec.get('placement_yaw_delta_deg',0.))
        yaws=[preferred]+[float(v) for v in self.spec.get('placement_yaw_candidates_deg',[0.,30.,-30.,60.,-60.,90.,-90.,180.]) if abs(float(v)-preferred)>1e-9]
        y_offsets=[float(v) for v in self.spec.get('placement_corridor_y_offsets_m',[0.])]
        if not y_offsets or not all(np.isfinite(v) for v in yaws+y_offsets):raise ValueError('Invalid placement candidate lists')
        batch_id=len(self.placement_batches);candidates=[]
        for offset in y_offsets:
            target_fp=np.asarray(target_pose[:3],float).copy();target_fp[1]+=offset
            for yaw in yaws:
                exact,rotation=rigid_functional_point_target(live,matrix,current,target_fp,yaw)
                direction=official_pre_dis_vector(live,matrix,rotation,target_pose,kwargs.get('pre_dis_axis','grasp'))
                pose=list(exact);pose[:3]=(np.asarray(exact[:3])-pre_dis*direction).tolist()
                candidates.append({'target_pose':pose,'target_functional_point0_world':(target_fp-pre_dis*direction).tolist(),
                    'official_target_functional_point0_world':target_fp.tolist(),'official_pre_dis_m':pre_dis,
                    'official_pre_dis_direction_world':direction.tolist(),'yaw_delta_deg':yaw,
                    'corridor_y_offset_m':offset,'placement_batch_id':batch_id,
                    'desired_bottle_axis_world':(rotation@matrix[:3,1]).tolist()})
        batch={'batch_id':batch_id,'pre_dis_m':pre_dis,'nominal_target_pose':candidates[0]['target_pose'],
               'candidates':candidates,'actual_live_ee_pose':live.tolist(),'actual_functional_point0':current.tolist()}
        self.placement_batches.append(batch)
        self.report.setdefault('placement_geometry_calls',[]).append(copy.deepcopy(batch))
        self.write()
        return batch['nominal_target_pose']

    def plan_path(self,robot,target_pose,constraint_pose=None,**kwargs):
        if self.dense_calls>=4:
            matching=[b for b in self.placement_batches if np.allclose(b['nominal_target_pose'],target_pose,rtol=0.,atol=1e-7)]
            self.placement_candidates=[]
            if matching:
                batch=matching[-1];candidates=copy.deepcopy(batch['candidates'])
                if self.previous_place_selection is not None:
                    previous=self.previous_place_selection
                    candidates.sort(key=lambda c:0 if abs(c['yaw_delta_deg']-previous.get('yaw_delta_deg',0.))<1e-9 and abs(c['corridor_y_offset_m']-previous.get('corridor_y_offset_m',0.))<1e-9 else 1)
                self.placement_candidates=candidates
                self.report.setdefault('placement_batch_matches',[]).append({'dense_call':self.dense_calls,'batch_id':batch['batch_id'],'pre_dis_m':batch['pre_dis_m']})
            else:
                self.report.setdefault('placement_batch_matches',[]).append({'dense_call':self.dense_calls,'batch_id':None,'unmatched_request_uses_exact_requested_goal':True})
            self.write()
        result=super().plan_path(robot,target_pose,constraint_pose=constraint_pose,**kwargs)
        if self.dense_calls>=4 and result.get('status')=='Success':
            self.previous_place_selection=copy.deepcopy(self.report['selected_carry_candidates'][str(self.dense_calls)])
        return result


def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)


def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,
                capture_callback=None,validate_actual_geometry=True):
    session=install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir)
    return session.run(record=record,capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
