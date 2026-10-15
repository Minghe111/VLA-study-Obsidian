from pathlib import Path
import json,hashlib,subprocess,time
ROOT=Path(__file__).resolve().parent
BASE=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1')
REMOTE='/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1/source_training_files/'
idx=json.loads((BASE/'data50_v1/native_index.json').read_text());rows=[]
for i in [0,38,30,31,17,7]:
    e=next(e for e in idx['episodes'] if e['index']==i);rel=e['source_hdf_relative_path'];target=ROOT/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        p=subprocess.run(['scp','-6','-o','BatchMode=yes','-o','HostKeyAlias=172.17.27.161','-o','ControlPath='+str(ROOT/'ssh.sock'),'liminghe@[2001:da8:9000:a837:ce29:cc6:56d1:1c3e]:'+REMOTE+rel,str(target)],capture_output=True,timeout=90)
        assert p.returncode==0,p.stderr.decode()
    digest=hashlib.sha256(target.read_bytes()).hexdigest();assert digest==e['hdf5_sha256'] and target.stat().st_size==e['hdf5_bytes']
    rows.append({'index':i,'path':str(target),'bytes':target.stat().st_size,'sha256':digest,'source_sha_matched':True})
    print('HDF_READY',i,flush=True)
(ROOT/'source_hdf_transfer.json').write_text(json.dumps({'selected_indices':[0,38,30,31,17,7],'files':rows,'all_source_sha_matched':True,'no_source_modifications':True},indent=2)+'\n')
