import json
from pathlib import Path
import pytest
import archive_protocol_repair as archive


def fixture(tmp_path, status='completed'):
    root=tmp_path/'new'
    continuation=root/'reports/continuation_v1'
    run=continuation/'language_warmup_96_0600'
    run.mkdir(parents=True)
    stop='budget_exhausted_not_converged' if status=='completed' else None
    (continuation/'summary.json').write_text(json.dumps({'status':status,'stop_reason':stop}))
    for name in archive.RUN_FILES:
        target=run/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.suffix=='.yaml':
            target.write_text('data: {}\n')
        else:
            target.write_text(json.dumps({'status':'completed','training':{'global_step':600}} if name=='report.json' else {}))
    (run/'checkpoint').mkdir()
    (run/'checkpoint/full_weights.pt').write_bytes(b'never copy weights')
    (run/'camera.npz').write_bytes(b'never copy frames')
    (continuation/'large.log').write_text('never copy logs')
    aggregate=tmp_path/'aggregate.json'
    aggregate.write_text(json.dumps({'root':str(root),'orchestrator_status':status,'stop_reason':stop,
        'latest_completed_global_step':600,'confirmed_converged':False}))
    diagnostics=tmp_path/'diagnostics.json'
    diagnostics.write_text('{}')
    return dict(root=root,destination=tmp_path/'archive',aggregate=aggregate,diagnostics=diagnostics)


def test_whitelist_copies_only_small_receipts_and_manifest_hashes(tmp_path):
    inputs=fixture(tmp_path)
    result=archive.archive(**inputs,execute=True)
    destination=inputs['destination']
    assert result['status']=='archived'
    assert result['archive_kind']=='final_completed_run'
    assert result['confirmed_converged'] is False
    assert len(list(destination.rglob('*.pt')))==0
    assert len(list(destination.rglob('*.npz')))==0
    assert len(list(destination.rglob('*.log')))==0
    manifest=json.loads((destination/'manifest.json').read_text())
    for receipt in manifest['files']:
        assert archive.sha256((destination/receipt['relative_path']).read_bytes())==receipt['sha256']
    with pytest.raises(FileExistsError):
        archive.archive(**inputs,execute=True)


def test_running_requires_explicit_snapshot_and_dry_run_writes_nothing(tmp_path):
    inputs=fixture(tmp_path,'running')
    with pytest.raises(ValueError,match='Final archive refused'):
        archive.archive(**inputs)
    result=archive.archive(**inputs,allow_running=True)
    assert result['status']=='planned'
    assert result['archive_kind']=='explicit_running_or_failed_snapshot'
    assert not inputs['destination'].exists()


def test_stale_aggregate_or_missing_receipt_blocks_final(tmp_path):
    inputs=fixture(tmp_path)
    inputs['aggregate'].write_text(json.dumps({'root':str(inputs['root']),'orchestrator_status':'running',
        'latest_completed_global_step':600}))
    with pytest.raises(ValueError,match='aggregate is stale'):
        archive.archive(**inputs)
    assert not inputs['destination'].exists()


def test_invalid_json_and_symlink_are_rejected_before_any_write(tmp_path):
    inputs=fixture(tmp_path)
    target=inputs['root']/'reports/continuation_v1/language_warmup_96_0600/evaluation.json'
    target.write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        archive.archive(**inputs,execute=True)
    target.unlink()
    target.symlink_to(inputs['aggregate'])
    with pytest.raises(ValueError,match='without symlink'):
        archive.archive(**inputs,execute=True)
    assert not inputs['destination'].exists()
