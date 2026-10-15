"""Read-only physical endpoint drive audit; no model or simulator execution."""
from pathlib import Path
import gzip,json,hashlib
import numpy as np

ROOT=Path('/tmp/realman-eval80k-20261012')
reports=[]
for folder in sorted((ROOT/'episodes').glob('ep*_model80000')):
    if not (folder/'report.json').is_file():continue
    initial=json.loads((folder/'initial_state.json').read_text())
    report=json.loads((folder/'report.json').read_text())
    names=initial['actual_physical_contract']['full_qpos_joint_names']
    slots=[names.index(f'l_joint{i}') for i in range(1,7)]+[names.index(f'r_joint{i}') for i in range(1,7)]
    physical_model_slots=np.r_[0:6,12:18]
    targets={row['physics_step_end']:row for row in report['executed_actions']}
    rows=[];max_drive=0.;max_qpos=0.;count=0
    trace=folder/'actual_physical_trace.jsonl.gz'
    with gzip.open(trace,'rt') as stream:
        for lineno,line in enumerate(stream,1):
            count=lineno
            if lineno not in targets:continue
            state=json.loads(line);assert state['step']==lineno
            target=np.asarray(targets[lineno]['raw_model_action24'])[physical_model_slots]
            drive=np.asarray(state['actual_drive36_rad'])[slots]
            actual=np.asarray(state['actual_qpos36_rad'])[slots]
            drive_error=float(abs(drive-target).max());qpos_error=float(abs(actual-target).max())
            assert drive_error==0
            max_drive=max(max_drive,drive_error);max_qpos=max(max_qpos,qpos_error)
            rows.append({'action_index':targets[lineno]['action_index'],'physical_step':lineno,'actual_arm_drive_exactly_equals_raw_model_target':True,'actual_endpoint_tracking_max_error_rad':qpos_error})
    assert count==report['actual_physics_steps'] and len(rows)==report['actual_action_count']
    reports.append({'episode':folder.name,'passed':True,'dense_physics_rows':count,'endpoint_actions':len(rows),
                    'all_arm_drive_endpoints_equal_model_targets_exactly':True,'max_arm_drive_endpoint_error_rad':max_drive,
                    'max_endpoint_arm_joint_tracking_error_rad':max_qpos,'max_endpoint_arm_joint_tracking_error_deg':float(np.rad2deg(max_qpos)),
                    'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'rows':rows})
assert len(reports)==6 and sum(v['endpoint_actions'] for v in reports)==2400
output={'passed':True,'scope':'independent dense physics trace arm drive endpoint readback; not a hand grasp/success claim',
        'completed_episodes':6,'dense_physics_rows':sum(v['dense_physics_rows'] for v in reports),
        'endpoint_actions':sum(v['endpoint_actions'] for v in reports),'all_actual_arm_drive_targets_equal_model_predictions':True,'episodes':reports}
(ROOT/'policy_runtime/final_dense_arm_drive_audit.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:output[k] for k in ('passed','completed_episodes','dense_physics_rows','endpoint_actions','all_actual_arm_drive_targets_equal_model_predictions')}))
