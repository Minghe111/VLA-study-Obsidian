"""v7 with strictly paired official pre-place/final-place targets.

The .02 m final segment retains exactly the first segment's selected rigid
rotation and corridor. Failure of that short segment remains a failure; it
cannot select a different yaw or corridor. Canonical actual auditor v13 stays.
"""
from pathlib import Path
import copy
import numpy as np
from expert_generation_v7 import (ExpertSession as _BaseSession,
                                 install_initializer,command24,measured24,_sha)


class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        source=Path(__file__).resolve();self.sources[str(source)]=_sha(source)
        self.report.update(expert_version='v8',source_hashes=self.sources,
            placement_interface='official two getter distance contract; final segment uses only first segment selected yaw/corridor')
        self.write()

    def plan_path(self,robot,target_pose,constraint_pose=None,**kwargs):
        if self.dense_calls==5:
            matching=[b for b in self.placement_batches if np.allclose(b['nominal_target_pose'],target_pose,rtol=0.,atol=1e-7)]
            previous=self.previous_place_selection
            if not matching or previous is None:
                self.report['placement_pair_failure']='Final placement requires a matching second getter and successful first placement'
                self.write()
                return {'status':'Fail','reason':self.report['placement_pair_failure']}
            batch=matching[-1]
            paired=[c for c in batch['candidates'] if abs(c['yaw_delta_deg']-previous.get('yaw_delta_deg',0.))<1e-9
                    and abs(c['corridor_y_offset_m']-previous.get('corridor_y_offset_m',0.))<1e-9]
            if len(paired)!=1:
                self.report['placement_pair_failure']='Final placement did not have exactly one matching rigid rotation/corridor'
                self.write()
                return {'status':'Fail','reason':self.report['placement_pair_failure']}
            batch['candidates']=copy.deepcopy(paired)
            self.report['strict_placement_pair']={'first_selection':copy.deepcopy(previous),
                'final_candidate':copy.deepcopy(paired[0]),'final_candidate_count':1,
                'different_yaw_or_corridor_fallback_allowed':False}
            self.write()
        return super().plan_path(robot,target_pose,constraint_pose=constraint_pose,**kwargs)


def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)


def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,
                capture_callback=None,validate_actual_geometry=True):
    session=install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir)
    return session.run(record=record,capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
