from pathlib import Path
import copy,json,sys
import dataset50_audit as original
before=original.pose_receipt
from dataset50_reader_v3 import native_pose_receipt,resolve_recorded_path
root=Path(__file__).resolve().parent
rel='batch_candidates/v9_gpu4_attempt02/slot0000043_candidate_attempt02_v9'
d=root/rel;scene=json.loads((d/'scene_sample.json').read_text());tests=[]
def run(name,fn,reject=False):
 try:fn();passed=not reject;error=None
 except (ValueError,KeyError,TypeError) as e:passed=reject;error=str(e)
 tests.append({'name':name,'passed':passed,'expected_rejection':reject,'error':error})
def mutate(change):
 s=copy.deepcopy(scene);change(s);return native_pose_receipt(s)
run('real_float32_report_exact',lambda:native_pose_receipt(scene))
run('heading_tamper_rejected',lambda:mutate(lambda s:s.update(actual_axis_yaw_deg=s['actual_axis_yaw_deg']+.001)),True)
run('physical_pose_class_not_relaxed',lambda:mutate(lambda s:s['sample_spec'].update(pose_class='diagonal')),True)
run('wrong_nearest_arm_rejected',lambda:mutate(lambda s:s['sample_spec'].update(arm='left')),True)
run('wrong_axis_rejected',lambda:mutate(lambda s:s.update(actual_axis_world=[0,1,0])),True)
run('nonrigid_matrix_rejected',lambda:mutate(lambda s:s['initial_pose'][0].__setitem__(0,2.0)),True)
g=json.loads((d/'expert_report.json').read_text())['actual36_geometry_audit'];raw=g['retained_original_v13_audit']['path']
run('producer_relative_path_correct',lambda:resolve_recorded_path(raw,d))
run('basename_guess_rejected',lambda:resolve_recorded_path('actual36_geometry_audit_v13_original.json',d),True)
run('cross_trial_absolute_path_rejected',lambda:resolve_recorded_path(root/'batch_candidates/sprite_lying_attempt00/slot0000015_attempt00/actual36_geometry_audit.json',d),True)
assert original.pose_receipt is before
run('circular_plus_minus180',lambda:mutate(lambda s:(s.update(initial_pose=[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]],actual_axis_world=[-1,0,0],actual_axis_yaw_deg=-180,upright_angle_deg=90))))
from publish_v3 import publish
out=root/'cpu_synthetic_v3_no_input_publication_MUST_NOT_EXIST';assert not out.exists();result=publish([],out);assert not result['complete'] and not out.exists()
tests.append({'name':'insufficient_dataset_never_published','passed':True})
r={'passed':all(t['passed'] for t in tests),'scope':'CPU pose/path/insufficient-publication fixtures only; no new episodes or GPU','test_count':len(tests),'tests':tests,'original_v1_global_unchanged':original.pose_receipt is before,'new_physics_runs':0,'gpu_runs':0}
(root/'cpu_synthetic_v3_readback_tests.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));sys.exit(0 if r['passed'] else 2)
