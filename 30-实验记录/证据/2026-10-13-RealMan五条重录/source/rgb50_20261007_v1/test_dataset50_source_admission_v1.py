"""Synthetic CPU metadata/path negative tests. No simulated success is created."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import dataset50_source_admission_v1 as app


def main():
    checks = []
    def rejected(name, function):
        try: function()
        except (ValueError, KeyError): checks.append({'name':name,'passed':True}); return
        raise AssertionError('Required negative was not rejected: '+name)
    expected = {s['index']:s for s in app.read_json(app.ROOT/'sample_plan.json')['specs']}
    def metadata(index, *, seed=None, trial=None, when='2026-10-07T00:00:00+00:00'):
        spec = expected[index]
        return {'trial_dir':trial or '/synthetic-metadata-only/slot'+str(index),
                'identity':{'spec_index':index,'seed':spec['seed'] if seed is None else seed,
                            'arm':spec['arm'],'model_id':spec['model_id'],'pose_class':spec['pose_class']},
                'qualified_utc':when,'synthetic_unit_metadata_only':True}
    with tempfile.TemporaryDirectory(prefix='dataset50_admission_cpu_unit_') as temporary:
        root = Path(temporary).resolve(); child = root/'child'; child.mkdir()
        rejected('path_escape', lambda:app.within(child, root/'other'))
        rejected('root_itself_not_trial', lambda:app.within(root, root))
        rejected('relative_path', lambda:app.within(root, Path('relative')))
        link = root/'alias'; link.symlink_to(child, target_is_directory=True)
        rejected('symlink_root_escape_or_alias', lambda:app.within(root, link/'trial'))
        source = root/'unit_source.py'; source.write_text('# Synthetic unit fixture only\n')
        actual = {str(source):app.sha(source)}
        rejected('wrong_required_source_sha', lambda:app.check_required_sources(actual,{str(source):'0'*64}))
        rejected('missing_required_source_pin', lambda:app.check_required_sources(actual,{str(root/'missing.py'):'0'*64}))
        source.write_text('# Changed synthetic unit fixture\n')
        rejected('current_source_changed', lambda:app.check_required_sources(actual,actual))
        rejected('changed_receipt_snapshot', lambda:app.same_snapshot({'receipt_sha':'a'},{'receipt_sha':'b'}))
        rejected('changed_raw_file_stats', lambda:app.same_snapshot({'stats':{'raw.pkl':[1,2]}},{'stats':{'raw.pkl':[1,3]}}))
        import dataset50_progress_v3 as gates
        (child/'worker_receipt.json').write_text(json.dumps({'accepted':False,'rgb_recorded':False,'scope':'synthetic negative metadata only'}))
        rejected('unfinished_worker', lambda:gates.ready_snapshot(child))
        (child/'worker_receipt.json').write_text(json.dumps({'accepted':True,'rgb_recorded':True,'scope':'synthetic negative metadata only'}))
        (child/'expert_report.json').write_text(json.dumps({'accepted':False,'sample_spec':expected[0],'synthetic_fixture_only':True}))
        rejected('failed_or_synthetic_expert', lambda:gates.ready_snapshot(child))
    rejected('unknown_formal_slot', lambda:app.formal_identity({'spec_index':100,'seed':1},expected))
    invalid=metadata(0);invalid['identity']['seed']=-1
    rejected('negative_actual_seed', lambda:app.formal_identity(invalid['identity'],expected))
    invalid=metadata(0);invalid['identity']['arm']='right' if invalid['identity']['arm']=='left' else 'left'
    rejected('formal_arm_relabel', lambda:app.formal_identity(invalid['identity'],expected))
    selected, conflicts, _, complete = app.select_unique([metadata(0),metadata(0,trial='/synthetic-metadata-only/later',when='2026-10-07T00:00:01+00:00')],expected)
    assert len(selected)==1 and len(conflicts)==1 and not complete
    checks.append({'name':'duplicate_slot_excluded','passed':True})
    selected, conflicts, _, complete = app.select_unique([metadata(0,seed=777),metadata(1,seed=777,when='2026-10-07T00:00:01+00:00')],expected)
    assert len(selected)==1 and len(conflicts)==1 and not complete
    checks.append({'name':'duplicate_seed_distinct_slots_excluded','passed':True})
    selected, conflicts, _, complete = app.select_unique([metadata(i) for i in range(49)],expected)
    assert len(selected)==49 and not conflicts and not complete
    checks.append({'name':'49_slots_never_complete','passed':True})
    print(json.dumps({'passed':all(x['passed'] for x in checks),'check_count':len(checks),'checks':checks,
                      'scope':'Synthetic CPU negative path/source/metadata decisions only; no successful simulator report, dataset accepted-list, physics, GPU or hardware IO created',
                      'new_physics_runs':0,'gpu_runs':0},indent=2))


if __name__=='__main__':main()
