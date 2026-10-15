"""Read-only algebra and two-getter contract checks; no simulator or motion."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from transforms3d.quaternions import mat2quat,quat2mat
from transforms3d.axangles import axangle2mat
import expert_generation_v7 as v7

rng=np.random.default_rng(712)
maximum_error=0.
for _ in range(100):
    live_R=axangle2mat(rng.normal(size=3),rng.uniform(-np.pi,np.pi))
    S=axangle2mat(rng.normal(size=3),rng.uniform(-np.pi,np.pi))
    live=np.r_[rng.normal(size=3),mat2quat(live_R)]
    bottle=np.eye(4);bottle[:3,3]=rng.normal(size=3)
    target=np.r_[rng.normal(size=3),mat2quat(axangle2mat(rng.normal(size=3),rng.uniform(-np.pi,np.pi)))]
    bias=np.linalg.inv(live_R)@(bottle[:3,3]-live[:3])
    official=S@live_R@bias;official/=np.linalg.norm(official)
    actual=v7.official_pre_dis_vector(live,bottle,S,target,'grasp')
    maximum_error=max(maximum_error,float(np.max(abs(official-actual))))
    assert np.allclose(actual,official,atol=1e-12,rtol=0.)
    for axis in ('fp',[1.,0.,0.]):
        reference=quat2mat(target[3:])@(np.array([0.,0.,1.]) if isinstance(axis,str) else axis)
        reference/=np.linalg.norm(reference)
        assert np.allclose(v7.official_pre_dis_vector(live,bottle,S,target,axis),reference,atol=1e-12,rtol=0.)

session=v7.ExpertSession.__new__(v7.ExpertSession)
live=np.array([.04,-.1,.95,1.,0.,0.,0.])
bottle_matrix=np.eye(4);bottle_matrix[:3,3]=[.13,-.1,.87]
bottle_matrix[:3,:3]=axangle2mat([1.,0.,0.],np.pi/2)
fp=np.array([.13,-.1,.94])
actor=SimpleNamespace(get_pose=lambda:SimpleNamespace(to_transformation_matrix=lambda:bottle_matrix),get_functional_point=lambda _:fp)
robot=SimpleNamespace(get_right_ee_pose=lambda:live)
session.task=SimpleNamespace(bottle=actor,robot=robot)
session.side='right';session.spec={};session.report={};session.write=lambda:None
session.placement_batches=[];session.previous_place_selection=None;session.dense_calls=4
target=[.25,-.12,.95,0.,1.,0.,0.]
pre=session.place_pose(actor,'right',target,functional_point_id=0,pre_dis=0.)
final=session.place_pose(actor,'right',target,functional_point_id=0,pre_dis=.02)
assert len(session.placement_batches)==2
assert not np.allclose(pre,final,rtol=0.,atol=1e-8)
for batch in session.placement_batches:
    for candidate in batch['candidates']:
        expected=np.asarray(candidate['official_target_functional_point0_world'])-batch['pre_dis_m']*np.asarray(candidate['official_pre_dis_direction_world'])
        assert np.allclose(expected,candidate['target_functional_point0_world'],atol=1e-12,rtol=0.)
original=v7._BaseSession.plan_path
observed=[]
def stub(self,robot,requested,**kwargs):
    observed.append((self.dense_calls,self.placement_candidates[0]))
    # First action's planner selects yaw +30; final action must try it first.
    choice=self.placement_candidates[1] if self.dense_calls==4 else self.placement_candidates[0]
    self.report.setdefault('selected_carry_candidates',{})[str(self.dense_calls)]=dict(choice,candidate=1 if self.dense_calls==4 else 0)
    return {'status':'Success'}
v7._BaseSession.plan_path=stub
try:
    assert session.plan_path(robot,pre)['status']=='Success'
    session.dense_calls=5
    assert session.plan_path(robot,final)['status']=='Success'
finally:v7._BaseSession.plan_path=original
assert observed[0][1]['placement_batch_id']==0 and observed[0][1]['official_pre_dis_m']==0.
assert observed[1][1]['placement_batch_id']==1 and observed[1][1]['official_pre_dis_m']==.02
assert observed[1][1]['yaw_delta_deg']==30.
report={'passed':True,'scope':'algebra and interface only; no IK, collision, force or simulation validation',
        'random_grasp_direction_cases':100,'maximum_direction_difference':maximum_error,
        'two_getters_distinct_batches':True,'selected_yaw_retained_for_final_offset':True,
        'observed_batch_sequence':[{'dense_call':i,'batch_id':c['placement_batch_id'],'pre_dis_m':c['official_pre_dis_m'],'first_yaw_deg':c['yaw_delta_deg']} for i,c in observed]}
print(json.dumps(report,indent=2))
