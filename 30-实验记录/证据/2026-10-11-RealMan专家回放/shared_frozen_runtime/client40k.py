"""Localhost-only policy client, separate from all simulator/source readers."""
from __future__ import annotations
import base64,hashlib,io,json,time,subprocess,shlex
from urllib.request import Request,urlopen
from urllib.parse import urlparse
from pathlib import Path
import numpy as np
from PIL import Image

CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')

class LocalPolicyClient:
    def __init__(self,base_url='http://127.0.0.1:18080',*,expected_model_identity_sha256,identity_path,timeout_s=300.):
        parsed=urlparse(base_url)
        if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.port!=18080 or parsed.path not in ('','/'):
            raise ValueError('Only the explicitly bound localhost model endpoint is supported')
        if len(expected_model_identity_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_model_identity_sha256):raise ValueError('Explicit current model identity SHA required')
        self.expected_identity=expected_model_identity_sha256
        self.identity_path=Path(identity_path)
        raw=self.identity_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.expected_identity:raise RuntimeError('Current published model identity file differs')
        identity=json.loads(raw)
        if identity.get('checkpoint_step')!=60000 or identity.get('full50_train') is not True or Path(identity['checkpoint']).name!='60000':raise RuntimeError('Expected full50 checkpoint60000 differs')
        self.url=base_url.rstrip('/');self.timeout=float(timeout_s);self.records=[]
    def remote_request(self,path,payload=None):
        code="import sys,urllib.request; data=sys.stdin.buffer.read(); r=urllib.request.Request('http://127.0.0.1:18080"+path+"',data=data if data else None,headers={'Content-Type':'application/json'},method='POST' if data else 'GET'); response=urllib.request.urlopen(r,timeout=600); sys.stdout.buffer.write(response.read())"
        command=['ssh','-S','/tmp/realman-eval60k-20261011/ssh.sock','-o','BatchMode=yes','liminghe@2001:da8:9000:a837:ce29:cc6:56d1:1c3e','python3 -c '+shlex.quote(code)]
        p=subprocess.run(command,input=payload or b'',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=self.timeout+60)
        if p.returncode:raise RuntimeError('SSH inference transport failed: '+p.stderr.decode()[-3000:])
        return p.stdout
    def health(self):
        result=json.loads(self.remote_request('/health'))
        if result.get('ready') is not True or result.get('model_identity_sha256')!=self.expected_identity:raise RuntimeError('Ready model identity differs')
        return result
    def __call__(self,observation):
        images=observation['images']
        if set(images)!=set(CAMERAS):raise ValueError('Exactly three official adapter camera keys required')
        encoded={};image_hashes={}
        for name in CAMERAS:
            arr=np.asarray(images[name])
            if arr.dtype!=np.uint8 or arr.shape!=(480,640,3):raise ValueError('Current D435 RGB uint8 480x640x3 required')
            buffer=io.BytesIO();Image.fromarray(arr).save(buffer,format='PNG');data=buffer.getvalue()
            encoded[name]=base64.b64encode(data).decode('ascii');image_hashes[name]=hashlib.sha256(data).hexdigest()
        state=np.asarray(observation['state'],dtype=np.float64)
        if state.shape!=(24,) or not np.isfinite(state).all():raise ValueError('Finite measured24 required')
        if not isinstance(observation['prompt'],str) or not observation['prompt'].strip():raise ValueError('Nonempty trial instruction required')
        payload={'images':encoded,'state24':state.tolist(),'prompt':observation['prompt']}
        raw=json.dumps(payload,allow_nan=False,separators=(',',':')).encode('utf-8')
        started=time.monotonic()
        out=self.remote_request('/infer',raw)
        if len(out)>8*1024*1024:raise ValueError('Unexpectedly large model response')
        result=json.loads(out.decode('utf-8'))
        if result.get('model_identity_sha256')!=self.expected_identity or result.get('raw_commands_unclipped') is not True or result.get('expert_actions_used') is not False:raise RuntimeError('Response model identity/raw-command provenance differs')
        actions=np.asarray(result['actions'])
        if actions.shape!=(16,24) or actions.dtype.kind not in 'fiu' or not np.isfinite(actions).all():
            raise ValueError('Model inverse-transformed actions must be finite16x24 absolute physical commands')
        self.records.append({'sim_time_s':float(observation['sim_time_s']),'request_sha256':hashlib.sha256(raw).hexdigest(),
            'response_sha256':hashlib.sha256(out).hexdigest(),'input_png_sha256':image_hashes,
            'input_state24':state.tolist(),'raw_actions':actions.tolist(),'wall_latency_s':time.monotonic()-started,
            'query_id':result.get('query_id'),'model_identity_sha256':result.get('model_identity_sha256'),
            'server_raw_actions_npy_sha256':result.get('raw_actions_sha256'),'server_timing':result.get('timing')})
        if hashlib.sha256(self.identity_path.read_bytes()).hexdigest()!=self.expected_identity:raise RuntimeError('Identity changed during inference')
        return result
