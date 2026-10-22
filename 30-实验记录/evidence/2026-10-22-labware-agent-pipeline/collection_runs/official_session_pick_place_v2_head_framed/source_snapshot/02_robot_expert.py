"""Contact-only native SDK24 expert extension for the independently staged task.

Reuses a frozen previously executed native planner/controller. Two explicit
local adaptations: general convex-part count and Z-up bottle axis convention.
No fixture, rigid attachment, gravity cancellation or object-state writes.
"""
from __future__ import annotations
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parent
FROZEN=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin')
LEGACY=FROZEN/'integrations/realman_rm65b_gen4'


def adapted_native():
    name='integrations.realman_rm65b_gen4.labware_expert_local'
    if name in sys.modules:return sys.modules[name]
    sys.path.insert(0,str(LEGACY))
    path=LEGACY/'acob_expert.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    original=module.fma
    fpath=Path(original.__file__)
    source=fpath.read_text()
    old='len(self.static["bottle"].shapes) != 7'
    if source.count(old)!=1:raise RuntimeError('Frozen collision shape assertion changed')
    source=source.replace(old,'len(self.static["bottle"].shapes) < 1')
    source=source.replace("Expected the official bottle's seven real convex collision shapes",
        'Expected imported vessel convex collision shapes')
    adapted=types.ModuleType(name+'.fma')
    adapted.__file__=str(fpath)
    exec(compile(source,str(fpath),'exec'),adapted.__dict__)
    module.fma=adapted
    retained=module.retained_grasp_matrices
    def zup_grasp(matrix,center,side,setting,pre_dis=.1):
        # The retained bottle used local +Y; our imported assets use local +Z.
        transform=np.eye(4);transform[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
        return retained(np.asarray(matrix)@transform,center,side,setting,pre_dis)
    module.retained_grasp_matrices=zup_grasp
    class MeasuredForceController(module.BoundaryContactController):
        # LP is an advisory diagnostic for this new vessel, not a feasibility
        # certificate. The physics rollout, not this heuristic, decides success.
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.direct_targets=np.array([0.,0.,1.2,1.2,2.4])
            self.force_targets=self.direct_targets.copy()
            self.direct_ready=0
        def target(self,*,carry=False):
            self.force_targets=self.direct_targets.copy()
            # Retain original SDK latch, rate cap, feedback and actual contacts;
            # do not trigger its legacy carry-LP early abort on this new geometry.
            result=super().target(carry=False)
            self.force_targets=self.direct_targets.copy()
            forces=self.last_evidence['forces']
            ready=bool(self.last_evidence['opposed'] and np.all(forces[2:5]>.05)
                and np.all(self.filtered[2:5]>=.8*self.direct_targets[2:5]))
            self.direct_ready=self.direct_ready+1 if ready else 0
            self.ready_boundaries=self.direct_ready
            self.last_ready_checks.update(
                acceptance_rule='actual three-main-finger opposing forces >=80% of direct targets for five 60ms boundaries; LP advisory only',
                direct_force_targets_N=self.direct_targets.tolist(),
                direct_observed_force_ready=ready,
                LP_feasibility_required=False,carry_requested=bool(carry))
            return result
    module.BoundaryContactController=MeasuredForceController
    module.labware_source_receipt={
        'native_expert_source':str(path),'native_expert_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'full_mesh_source':str(fpath),'full_mesh_sha256':hashlib.sha256(fpath.read_bytes()).hexdigest(),
        'local_changes':['Accept >=1 actual convex part rather than exactly7; geometry unchanged',
            'Transform Z-up vessel axis into retained Y-up grasp arithmetic',
            'Use bounded measured force servo (1.2/1.2/2.4N) with LP advisory, not a claimed LP-feasible gate'],
        'shared_source_modified':False,'hardware_io':False}
    return module


def run_contact_pick_place(task,max_targets=1100):
    native=adapted_native()
    # Fold the two unused long fingers through actual SDK drive targets before
    # approaching the 90mm vessel; their factory-open pose intersects the desk.
    prepare=native.command24(task.robot)
    hand=slice(18,24) if task.arm=='right' else slice(6,12)
    final=prepare.copy();final[hand]=[0.,0.,1000.,1000.,1000.,0.]
    for fraction in np.linspace(0.,1.,45)[1:]:
        task.apply_sdk24(prepare+(final-prepare)*fraction,'prepare_three_finger_grasp')
    setting={'open_sdk':[0.,0.,1000.,1000.,1000.,0.],
        'close_sdk':[0.,0.,608.,610.,610.,0.], 'hand_x':-.015}
    teacher=native.RealManSDK24Teacher(task,setting=setting,audit_approach=True,max_close_boundaries=150)
    teacher.planner.wall_budget_s=180.
    task.teacher=teacher
    (task.runtime_path/'expert_source_receipt.json').write_text(json.dumps(native.labware_source_receipt,indent=2)+'\n')
    while not task.lift_achieved:
        if task.take_action_cnt>=max_targets:raise RuntimeError('Finite expert target limit exceeded')
        proposal=teacher.next_action()
        task.apply_sdk24(proposal['action'],proposal['phase'])
        if task.take_action_cnt%25==0:print(json.dumps({'step':task.take_action_cnt,'phase':teacher.phase,**task.metrics}),flush=True)
    # Establish the placement from the measured palm-bottle relation, never by
    # resetting the object. Each <=10mm downward Cartesian knot is solved anew.
    palm=teacher._palm()
    required_descent=max(0.,float(task.metrics['bottom_gap_m'])-.001)
    for delta in np.linspace(0.,required_descent,max(2,int(np.ceil(required_descent/.008))+1))[1:]:
        goal=palm.copy();goal[2,3]-=delta
        queue=teacher.planner.solve(goal,approach=False,reserve_targets=150)
        for arm in queue:
            action=native.command24(task.robot)
            action[teacher.arm_slice]=arm
            action[teacher.hand_slice]=teacher.contact.target(carry=True)
            task.apply_sdk24(action,'lower_to_table')
    # Release to factory open posture over a finite velocity-bounded SDK ramp.
    task.release_requested=True
    begin=native.command24(task.robot)
    for f in np.linspace(0,1,40)[1:]:
        action=begin.copy()
        action[teacher.hand_slice]=begin[teacher.hand_slice]*(1-f)+np.asarray(setting['open_sdk'])*f
        task.apply_sdk24(action,'release')
    # Retreat away from the object along the reverse last approach direction.
    palm=teacher._palm()
    offset=teacher.palm_goals[0][:3,3]-teacher.palm_goals[1][:3,3]
    offset=offset/max(np.linalg.norm(offset),1e-10)*.07
    for fraction in np.linspace(.25,1,4):
        goal=palm.copy();goal[:3,3]+=offset*fraction
        for arm in teacher.planner.solve(goal,approach=False,reserve_targets=20):
            action=native.command24(task.robot);action[teacher.arm_slice]=arm
            action[teacher.hand_slice]=setting['open_sdk']
            task.apply_sdk24(action,'retreat')
    task.retreat_completed=True
    for _ in range(15):
        task.apply_sdk24(native.command24(task.robot),'settle')
        if task.check_success():break
    teacher.write_report()
    if not task.check_success():raise RuntimeError('Actual release and stable placement success condition not reached')
    return task.info
