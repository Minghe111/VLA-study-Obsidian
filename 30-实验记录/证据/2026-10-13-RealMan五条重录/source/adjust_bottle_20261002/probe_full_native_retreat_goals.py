"""Only root queues. Compare original official arm-frame retreat and world Z lift.
Masked full-native IK endpoints only: no RRT execution, scene steps or GPU.
"""
import importlib
import json
import hashlib
from pathlib import Path
import numpy as np
import full_native_carry_planner as fncp
import preflight_place_goals as ppg


def run_goal_probe(task,base,reference=None):
    importlib.reload(fncp);base=Path(base);before=ppg._snapshot(task)
    try:
        start=ppg._matrix(task.robot.get_left_ee_pose())
        original=start.copy();original[:3,3]-=.1*original[:3,0]
        world=start.copy();world[2,3]+=.1
        old=base/'coupled_side_entry_v10_B085_20261003_seed7/planned_carry_path_4.npz'
        supplied=np.load(old)['position'][-1] if old.exists() else None
        results={}
        for name,goal in [('official_arm_retreat',original),('official_world_Z_lift',world)]:
            try:
                result=fncp.plan_carry_to_task_pose(task,ppg._vector(goal),palm_to_bottle=reference,
                    endpoint_only=True,ik_attempts=24,candidate_arm_q=supplied if name=='official_arm_retreat' else None)
            except Exception as error:result={'status':'Exception','message':str(error),'exception_type':type(error).__name__}
            results[name]=result;meta=result.get('planning_metadata',{});counts={};pairs={}
            for candidate in meta.get('endpoint_candidates',[]):
                reason=candidate.get('reason','unknown');counts[reason]=counts.get(reason,0)+1
                for collision in candidate.get('collisions',[]):
                    pair=' / '.join(sorted([collision.get('link_name1','?'),collision.get('link_name2','?')]))
                    pairs[pair]=pairs.get(pair,0)+1
            print('FULL_NATIVE_GOAL',name,'status',result['status'],'goal',ppg._vector(goal),
                'free',meta.get('accepted_free_endpoints'),'reasons',json.dumps(counts),
                'collisionpairs',json.dumps(pairs),'error',result.get('message'),flush=True)
            ppg._assert_snapshot(task,before)
        report={'scope':'Read-only original retreat vs official world-lift endpoint probes. No path/action executes.',
            'helper_sha256':hashlib.sha256(Path(fncp.__file__).read_bytes()).hexdigest(),
            'actual_start_task_pose_world':ppg._vector(start),'results':results,'physical_state_unchanged':True}
        output=base/'full_native_retreat_goal_preflight_v3_20261003.json'
        with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
        print('FULL_NATIVE_GOAL_REPORT',str(output),flush=True);return report
    finally:ppg._assert_snapshot(task,before)


if 'task' in globals():
    reference=globals().get('carry_reference')
    reference=reference[0] if reference is not None else None
    full_native_goal_report=run_goal_probe(task,BASE,reference)
elif __name__=='__main__':print(json.dumps(fncp.run_cpu_tests(),indent=2))
