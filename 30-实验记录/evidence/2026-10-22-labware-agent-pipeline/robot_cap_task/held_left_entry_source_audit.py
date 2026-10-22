from pathlib import Path
import json,hashlib,importlib.util,numpy as np
P=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task')
source=(P/'held_left_planner_independent_audit.py').read_text();prefix=source.split("check('actual_contact_ID_and_sourcepart_only_allowlist',selection)")[0];ctx={};exec(compile(prefix,'<independent-held-fixtures>','exec'),ctx)
spec=importlib.util.spec_from_file_location('independent_left_contact_entry',P/'left_cap_contact_probe.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
make=ctx['make_auditor'];Q=ctx['Q'];L=ctx['L'];N=ctx['N'];RN=ctx['RN'];LI=ctx['LI'];RI=ctx['RI'];snap=ctx['snap'];goal=Q[L].copy();goal[0]+=.04
pairs=[]
for pair in [(RN,'bottle'),(LI,'cap'),(LI,'bottle'),(LI,RI),(LI,'table'),(LI,'pill_11')]:
 a=make(collision=pair);full=a.audit(goal[None,:],'left');a=make(collision=pair)
 try:r=mod.moving_path_audit(a,goal[None,:],N);scoped=r['scoped_left_sampled_path_passed']
 except mod.PathRejected as e:scoped=False;r=e.report
 pairs.append({'collision_pair':pair,'complete_held_audit_passed':full['passed'],'moving_left_scoped_passed':scoped,'expected_frozen_pair_outside_scopedcheck':pair[0]==RN})
empty_rejected=False
try:mod.moving_path_audit(make(),np.zeros((0,6)),N)
except ValueError:empty_rejected=True
assert empty_rejected
a=make();clear=mod.moving_path_audit(a,goal[None,:],N);assert clear['scoped_left_sampled_path_passed']
assert pairs[0]['moving_left_scoped_passed'] and not pairs[0]['complete_held_audit_passed'];assert all(not x['moving_left_scoped_passed'] for x in pairs[1:])
qd=np.array(snap['actual_drive_target24'][:6]);qm=Q[L]
o={'source_sha256':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['held_left_planner.py','left_cap_contact_probe.py','left_cap_contact_task.py']},'scope':'Read-only source and actual helper CPU negativecontrol; no scene/physics/actualpathvalidation','new_entry_control_scope':{'actual_q36_from_live_art':True,'every_new_action_from_actual_drive24':True,'rightarmdrive_fixed_with_originalrightfeedback':True,'metadata_rightactions_not_used':True,'partial_command_not_exported_as_complete_transition':True,'controller_created_after_actualleftprepare_andapproach':True},'moving_scope_negative_controls':pairs,'actualdynamichelper_exec_valid_clear_nonempty_path_passed':True,'actualdynamichelper_empty_path_rejected':True,'measured_vs_drive_start_discrepancy_example':{'actual_static_snapshot':str(P/'static_hold_probes/held115_static_reference_v1/actual_terminal_snapshot.json'),'measured_leftarm6':qm.tolist(),'actual_drive_leftarm6':qd.tolist(),'max_abs_difference_rad':float(abs(qd-qm).max()),'source_at_review':'Revisedactualentrycapturesactualdrive_leftonce, auditsnp.vstack([actualdrive_left,endpoint])withautomaticmeasuredstart, andbuildsquinticfromthesamecapture; executedlineiscovered','action':'Earlierfailurepreserved; sourceownerrevisedauditpathanddynamicindentation; auditoronlyrechecks'},'new_physical_trials':0,'physical_collision_masks_changed':False,'caller_source_modified_by_auditor':False,'interpretation':'Moving-left sampled scope is explicitly narrower than complete heldbaseline-only audit; it cannot inherit that audit acceptance or be described as checking allright collisions.'}
(P/'held_left_entry_source_audit.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');print(json.dumps(o,ensure_ascii=False,indent=2))
