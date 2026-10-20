from pathlib import Path
import os,socket,json,time,hashlib,zipfile,subprocess
ROOT=Path('/bigdata2/liminghe/rlinf_history3000_20261019_v1')
assert ROOT.resolve()==ROOT and socket.gethostname()=='user-4U-GPU-Server' and os.getuid()==1014
DEST=ROOT/'source_tree/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2/reports/continuation_v1/language_warmup_96_2200/checkpoint/full_weights.pt'
SRC=ROOT/'weight_repair_v1/full_weights.pt';BAD=ROOT/'migration_source/full_weights.truncated_first_transfer.pt'
EXPECTED='5e16c2c36ba8c99351d4b0f9af66b6600d94250e1a07bea4f227b8bd5e2e4531'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
for p in (SRC,DEST): assert p.resolve()==p and p.is_file()
assert SRC.stat().st_size==6968087357 and sha(SRC)==EXPECTED
assert DEST.stat().st_size==1852211200 and sha(DEST)=='4b319bc421a47b03089e68a98b2688f7d4568c23e176ca511787c503a9988947'
assert not BAD.exists()
receivers=[line for line in subprocess.check_output(['ps','-u','liminghe','-o','pid=,comm=,args='],text=True).splitlines() if len(line.split(None,2))==3 and line.split(None,2)[1]=='rsync'];assert not receivers,receivers
with zipfile.ZipFile(SRC) as archive:
 names=archive.namelist();assert len(names)>144
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
state=torch.load(SRC,map_location='cpu',weights_only=False)
assert isinstance(state,dict) and len(state)>144
count=len(state);del state
oldstat=DEST.stat();os.replace(DEST,BAD);os.replace(SRC,DEST)
assert DEST.stat().st_size==6968087357 and sha(DEST)==EXPECTED
report={'status':'passed_weight_repaired_byte_exact_and_CPU_deserialized','timestamp_unix':time.time(),'expected_sha256':EXPECTED,'bytes':DEST.stat().st_size,'zip_members':len(names),'top_level_state_entries':count,'destination':str(DEST),'quarantined_bad_file':str(BAD),'bad_size':oldstat.st_size,'bad_mtime':oldstat.st_mtime,'bad_ctime':oldstat.st_ctime,'no_live_rsync_receiver':True,'base_weight_conversion':False,'model_parameter_updates':0,'CUDA_initialized':torch.cuda.is_initialized(),'source_local_weight_unchanged':True}
assert not report['CUDA_initialized']
(ROOT/'reports/weight_repair_v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
