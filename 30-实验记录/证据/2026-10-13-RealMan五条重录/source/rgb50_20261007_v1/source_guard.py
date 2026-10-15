"""Hash protected official sources, hardware files and old20 without editing them."""
from pathlib import Path
import argparse, hashlib, json, subprocess
ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');WORK=Path(__file__).resolve().parent
CONFIG=ROOT.parent/'deploy/realman_gen4/batch_runs_v32/realman_adjust_bottle_batch20_20261003/attempts/seed_0000007/bootstrap/run_config.yml'
OLD20=ROOT.parent/'deploy/realman_gen4/rgb20_final_20261004/dataset'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def gather():
    import yaml
    cfg=yaml.safe_load(CONFIG.read_text());files={CONFIG,OLD20/'manifest.json',OLD20/'episodes/episode_0000000/profile_used.yml'}
    for folder in (ROOT/'envs',ROOT/'integrations/realman_rm65b_gen4'):
        files.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.yml','.yaml','.json','.urdf','.srdf'))
    for side in ('left','right'):
        folder=Path(cfg[side+'_robot_file'])
        if not folder.is_absolute():folder=ROOT/folder
        files.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        config=cfg[side+'_embodiment_config']
        for key,value in config.items():
            if isinstance(value,str) and value.endswith(('.urdf','.srdf','.yml','.yaml')):
                for base in (ROOT,folder):
                    p=base/value
                    if p.is_file():files.add(p)
    for model in (13,16,4):
        for relative in (f'model_data{model}.json',f'visual/base{model}.glb',f'collision/base{model}.glb'):files.add(ROOT/'assets/objects/001_bottle'/relative)
    files.update(OLD20.glob('data/*.hdf5'))
    return {'files':{str(p):sha(p) for p in sorted(files)},'official_git_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'official_git_status':subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).splitlines(),'old20_manifest_sha256':sha(OLD20/'manifest.json')}
ap=argparse.ArgumentParser();ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--verify',action='store_true');opt=ap.parse_args()
now=gather()
if opt.verify:
    before=json.loads(opt.snapshot.read_text());changes=[p for p in set(now['files'])|set(before['files']) if now['files'].get(p)!=before['files'].get(p)]
    result={'unchanged':not changes and before['official_git_head']==now['official_git_head'],'changed_files':changes,'protected_files':len(now['files']),'git_status_unchanged':before['official_git_status']==now['official_git_status'],'old20_manifest_sha256':now['old20_manifest_sha256']}
    opt.snapshot.with_name(opt.snapshot.stem+'_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0 if result['unchanged'] else 2)
else:
    if opt.snapshot.exists():raise RuntimeError('No baseline overwrite')
    opt.snapshot.write_text(json.dumps(now,indent=2));print(json.dumps({'protected_files':len(now['files']),'old20_manifest_sha256':now['old20_manifest_sha256']}))