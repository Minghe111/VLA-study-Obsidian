from preflight_place_goals import _snapshot,_assert_snapshot
before=_snapshot(task)
names_auditor=fma.FullMeshPathAuditor(task)
print('AUX_STATIC_OBJECT_NAMES',{k:(v.name,len(v.shapes)) for k,v in names_auditor.static.items()},flush=True)
_assert_snapshot(task,before)
import observed_trace_audit as ota
observed_v9=ota.analyze_observed_trace(task, BASE/'coupled_side_entry_v9_B085_20261003_seed7'/'physics_trace.json', output_path=BASE/'v9_observed_actual_fk_audit_20261003.json')
print('OBSERVED_V9_DONE',{k:v for k,v in observed_v9.items() if k not in ['samples','metadata','source_hashes']},flush=True)
