"""v8 physical expert with additive, retained-v13 boundary review.

Only the independent final recorded-state auditor is selected differently.
The official task, hand control, LP, planning, motion and physical success
predicates stay byte-identical in the inherited methods. The original v13
audit is retained as a separate file for every fresh episode.
"""
from pathlib import Path
import ast
import hashlib
import inspect
import json
import textwrap
from expert_generation_v8 import (ExpertSession as _BaseSession,
                                 install_initializer,command24,measured24,_sha)
from expert_generation import ExpertSession as _OriginalSession

FROZEN_BOUNDARY_HELPER_SHA256='e40149ba15aff34de64a4e6d3cdcda700766934f077f47cab1888c1b36335020'
FROZEN_CPU_CONSTRUCTOR_SHA256='e5d13b16ef6aff9ad9624978e9c269a1bd4b86b4c526542bb03acaa9c139f490'


def audit_with_retained_report(task,trace_path,*,report_path,original_output_path,
                               output_path=None,**kwargs):
    """Bind complete original report bytes and both full audit results."""
    from closure_boundary_audit_v14 import audit_accepted_actual_trace_v14
    report_path=Path(report_path)
    retained=report_path.with_name('expert_report_pre_actual_audit.json')
    if retained.exists() or (output_path is not None and Path(output_path).exists()):
        raise ValueError('Refusing to overwrite a retained pre-audit report or final audit')
    payload=report_path.read_bytes()
    with retained.open('xb') as stream:stream.write(payload)
    result=audit_accepted_actual_trace_v14(task,trace_path,report_path=report_path,
        original_output_path=original_output_path,output_path=None,**kwargs)
    result['retained_pre_actual_audit_report']={'path':str(retained),'sha256':_sha(retained),
        'original_bytes_unmodified':True,'matches_original_report_sha256':_sha(retained)==result['report_sha256']}
    if result['retained_pre_actual_audit_report']['matches_original_report_sha256'] is not True:
        raise RuntimeError('Retained report bytes differ from original audit report identity')
    result['boundary_used']=bool(result['resolved_closure_boundary_events'])
    result['boundary_resolved_event_count']=len(result['resolved_closure_boundary_events'])
    if output_path is not None:
        with Path(output_path).open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def _retained_audit_run():
    """Private AST copy changes one final-auditor import and one keyword only."""
    original=_OriginalSession.run
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    imports=0;calls=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module=='observed_full_trace_audit_v13':
            if len(node.names)!=1 or node.names[0].name!='audit_accepted_actual_trace':
                raise RuntimeError('Unexpected canonical final-audit import')
            node.module='expert_generation_v9'
            node.names=[ast.alias(name='audit_with_retained_report',asname='audit_accepted_actual_trace')]
            imports+=1
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='audit_accepted_actual_trace':
            node.keywords.append(ast.keyword(arg='original_output_path',
                value=ast.parse("self.out / 'actual36_geometry_audit_v13_original.json'",mode='eval').body))
            calls+=1
    if imports!=1 or calls!=1:raise RuntimeError('Canonical final audit sites changed')
    ast.fix_missing_locations(tree)
    environment=dict(original.__globals__)
    exec(compile(tree,str(Path(__file__).resolve())+':retained_v13_audit_selection','exec'),environment)
    return environment['run'],{'final_audit_import_sites_changed':imports,
        'final_audit_call_keyword_sites_changed':calls,
        'private_run_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'physical_success_and_control_statements_unchanged':True,
        'canonical_run_file_unmodified':True,'retained_original_audit_filename':'actual36_geometry_audit_v13_original.json',
        'retained_pre_audit_report_filename':'expert_report_pre_actual_audit.json'}


_RUN,_RUN_RECEIPT=_retained_audit_run()


class ExpertSession(_BaseSession):
    run=_RUN

    def write(self):
        if 'actual36_geometry_audit' in self.report:
            audit=self.out/'actual36_geometry_audit.json'
            self.report.update(actual36_geometry_audit_sha256=_sha(audit),
                boundary_used=bool(self.report['actual36_geometry_audit']['boundary_used']),
                boundary_resolved_event_count=int(self.report['actual36_geometry_audit']['boundary_resolved_event_count']))
        return super().write()

    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        work=Path(__file__).resolve().parent
        expected={work/'closure_boundary_audit_v14.py':FROZEN_BOUNDARY_HELPER_SHA256,
                  work/'benchmark_geometry_cache.py':FROZEN_CPU_CONSTRUCTOR_SHA256}
        if any(_sha(p)!=digest for p,digest in expected.items()):
            raise RuntimeError('Reviewed boundary helper or zero-step CPU constructor changed')
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        self.sources.update({str(p):digest for p,digest in expected.items()})
        source=Path(__file__).resolve();self.sources[str(source)]=_sha(source)
        for relative in ('integrations/realman_rm65b_gen4/physics_setup.py','envs/utils/create_actor.py'):
            path=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')/relative
            self.sources[str(path)]=_sha(path)
        self.report.update(expert_version='v9',source_hashes=self.sources,
            independent_actual_audit_selection='reviewed closure_boundary_v14; retained original v13 result',
            final_audit_selection_receipt=_RUN_RECEIPT)
        self.write()


def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)


def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,
                capture_callback=None,validate_actual_geometry=True):
    session=install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir)
    return session.run(record=record,capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
