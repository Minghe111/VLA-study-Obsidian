"""Zero-step AST scope verification for v9 auditor selection."""
import ast,inspect,textwrap,json,tempfile
from pathlib import Path
import expert_generation_v9 as v9
from expert_generation import ExpertSession as Original
from expert_generation_v8 import ExpertSession as Previous

tree=ast.parse(textwrap.dedent(inspect.getsource(Original.run)))
original=ast.dump(tree,include_attributes=False)
for node in ast.walk(tree):
    if isinstance(node,ast.ImportFrom) and node.module=='observed_full_trace_audit_v13':
        node.module='expert_generation_v9'
        node.names=[ast.alias(name='audit_with_retained_report',asname='audit_accepted_actual_trace')]
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='audit_accepted_actual_trace':
        node.keywords.append(ast.keyword(arg='original_output_path',
            value=ast.parse("self.out / 'actual36_geometry_audit_v13_original.json'",mode='eval').body))
import hashlib
assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==v9._RUN_RECEIPT['private_run_ast_sha256']
for node in ast.walk(tree):
    if isinstance(node,ast.ImportFrom) and node.module=='expert_generation_v9':
        node.module='observed_full_trace_audit_v13';node.names=[ast.alias(name='audit_accepted_actual_trace')]
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='audit_accepted_actual_trace':
        node.keywords=[k for k in node.keywords if k.arg!='original_output_path']
assert ast.dump(tree,include_attributes=False)==original
assert v9.ExpertSession.plan_path is Previous.plan_path
assert v9.ExpertSession.dense is Previous.dense
assert v9.ExpertSession.step is Previous.step
assert v9.ExpertSession.hold is Previous.hold
assert v9.ExpertSession.place_pose is Previous.place_pose
import closure_boundary_audit_v14 as helper
real_audit=helper.audit_accepted_actual_trace_v14
def fake_audit(task,trace,*,report_path,original_output_path,output_path=None,**kwargs):
    assert output_path is None
    result={'passed':True,'samples':[],'report_sha256':v9._sha(report_path),
            'resolved_closure_boundary_events':[],
            'retained_original_v13_audit':{'path':str(original_output_path)}}
    Path(original_output_path).write_text(json.dumps(result))
    result['retained_original_v13_audit']['sha256']=v9._sha(original_output_path)
    return result
helper.audit_accepted_actual_trace_v14=fake_audit
try:
    with tempfile.TemporaryDirectory(prefix='robotwin_v9_provenance_check_') as directory:
        out=Path(directory);report=out/'expert_report.json'
        exact=b'{"accepted": true, "test": "provenance fixture only"}\n'
        report.write_bytes(exact)
        result=v9.audit_with_retained_report(None,out/'fixture_trace.json',report_path=report,
            original_output_path=out/'actual36_geometry_audit_v13_original.json',
            output_path=out/'actual36_geometry_audit.json')
        assert (out/'expert_report_pre_actual_audit.json').read_bytes()==exact
        assert result['retained_pre_actual_audit_report']['matches_original_report_sha256'] is True
        session=v9.ExpertSession.__new__(v9.ExpertSession);session.out=out
        session.report={'actual36_geometry_audit':result}
        session.write()
        assert session.report['actual36_geometry_audit_sha256']==v9._sha(out/'actual36_geometry_audit.json')
        assert session.report['boundary_used'] is False and session.report['boundary_resolved_event_count']==0
finally:helper.audit_accepted_actual_trace_v14=real_audit
print(json.dumps({'passed':True,'scope':'AST selection, import and method identity only; no physical episode',
                  'reverting_two_selected_AST_sites_restores_exact_original':True,
                  'plan_dense_step_hold_place_method_identity_unchanged':True,
                  'report_bytes_and_final_audit_sha_binding_fixture_passed':True,
                  'receipt':v9._RUN_RECEIPT},indent=2))
