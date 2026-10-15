import importlib,observed_full_trace_audit_v11 as actual36
importlib.reload(actual36)
candidate_actual_audit=actual36.audit_accepted_actual_trace(task,OUT/'physics_trace.json',report_path=OUT/'attempt_report.json',output_path=OUT/'actual36_geometry_audit.json')
print('CANDIDATE_ACTUAL36_RESULT',candidate_actual_audit['passed'],candidate_actual_audit.get('events'),flush=True)
snapshot(task,'accepted_final')
