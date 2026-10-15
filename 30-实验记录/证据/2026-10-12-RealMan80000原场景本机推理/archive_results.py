"""Copy immutable experimental evidence, not environments or full RGB caches."""
from pathlib import Path
import datetime, hashlib, json, shutil

ROOT = Path(__file__).resolve().parent
VAULT = Path('/home/admin123/liminghe/VLA-WAM-RL')
EVIDENCE = VAULT / '30-实验记录/证据/2026-10-12-RealMan80000原场景本机推理'
MEDIA = VAULT / '30-实验记录/附件/2026-10-12-RealMan80000原场景本机推理'
INDICES = [0,38,30,31,17,7]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    summary=json.loads((ROOT/'six_model_summary.json').read_text())
    assert len(summary['episodes'])==6 and summary['selected_indices']==INDICES
    assert summary['actual_model_queries']==150
    EVIDENCE.mkdir(parents=True,exist_ok=True); MEDIA.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def copy(p,base=EVIDENCE,rel=None):
        assert p.is_file() and not p.is_symlink(),str(p)
        dest=base/(rel if rel is not None else p.relative_to(ROOT))
        dest.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(p)
        if dest.exists():
            assert sha(dest)==digest, f'Refusing to overwrite differing evidence {dest}'
        else:shutil.copy2(p,dest)
        assert sha(dest)==digest
        manifest.append({'source':str(p),'destination':str(dest),'bytes':p.stat().st_size,'sha256':digest})

    for name in ['six_model_summary.json','six_model_posthoc_diagnostics.json','source_initial_records.json','source_initial_restoration_audit.json','source_hdf_transfer.json','ssh_cleanup.json','frozen_runtime_copy_receipt.json','scaffold_provenance.json','source_manifest.json','hardware_guard.json','archive_results.py','fetch_hdf.py']:
        copy(ROOT/name)
    for folder in ['scripts','source_helpers','official_openpi','viewer','hardware']:
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in {'.pyc','.tar'}:continue
            copy(p)
    for p in sorted((ROOT/'policy_runtime').glob('*')):
        if p.is_file() and p.suffix in {'.py','.sh','.json','.txt','.log'}:copy(p)
    for p in sorted((ROOT/'policy_runtime/archive_bundle').rglob('*')):
        if p.is_file():copy(p)
    for p in sorted((ROOT/'policy_runtime/service_v1').glob('*.json')):copy(p)
    for q in range(150):
        directory=ROOT/'policy_runtime/service_v1/requests'/f'{q:06d}'
        assert directory.is_dir(),str(directory)
        for p in sorted(directory.rglob('*')):
            if p.is_file():copy(p)

    excluded_rgb=[]
    for index in INDICES:
        episode=ROOT/'episodes'/f'ep{index:02d}_model80000'
        for p in sorted(episode.rglob('*')):
            if not p.is_file():continue
            if p.suffix=='.mp4':copy(p,MEDIA,p.name);continue
            if p.suffix=='.png':copy(p,MEDIA,p.name);continue
            if p.suffix=='.npz' and ('actual_rgb' in p.parts or 'observer_rgb' in p.parts):
                excluded_rgb.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
                continue
            copy(p)
        log=episode.with_suffix('.log')
        if log.exists():copy(log)
    for folder in ['rendered_videos','videos']:
        directory=ROOT/folder
        if not directory.exists():continue
        for p in sorted(directory.rglob('*')):
            if p.is_file():copy(p,MEDIA,p.relative_to(directory))
    for name in ['post_hardware_guard.json','terminal_diagnostic.json','render_validation.json','video_export.json','render_videos.py','render.log','checkpoint_preservation.json']:
        if (ROOT/name).is_file():copy(ROOT/name)
    receipt={
        'created':datetime.datetime.now().astimezone().isoformat(),
        'scope':'Six formal original-scene checkpoint80000 runs only; excludes later manual repeats',
        'verified_files':len(manifest),'verified_bytes':sum(r['bytes'] for r in manifest),
        'all_copied_sha256_match':True,'files':manifest,
        'temporary_rgb_not_copied':excluded_rgb,
        'temporary_rgb_not_copied_bytes':sum(r['bytes'] for r in excluded_rgb),
        'not_copied':['source/data six HDF5 (473473406 bytes, separate source SHA receipt)','policy_runtime venv/pip cache/JAX cache','large checkpoint weights already at persistent deploy path','mutable live frame spool and later manual_* runs'],
        'runtime_path':str(ROOT),'temporary_runtime_is_not_cross_cleanup_durable':True,
    }
    (EVIDENCE/'archive_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['files','temporary_rgb_not_copied']},ensure_ascii=False))

if __name__=='__main__':main()
