"""Current local checkpoint only; real PNG observations and physical24 outputs."""
import base64,hashlib,io,json,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlparse
import numpy as np
from PIL import Image

class LocalPolicyClient:
    CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')
    def __init__(self,base_url,*,expected_model_identity_sha256,identity_path,timeout_s=600.):
        u=urlparse(base_url)
        if u.scheme!='http' or u.hostname not in ('127.0.0.1','localhost') or u.path not in ('','/'):
            raise ValueError('Local loopback inference only')
        self.url=base_url.rstrip('/');self.expected_identity=expected_model_identity_sha256
        self.identity_path=Path(identity_path);self.timeout=timeout_s;self.records=[]
        raw=self.identity_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.expected_identity:raise RuntimeError('Identity SHA differs')
        ident=json.loads(raw)
        if ident.get('checkpoint_step')!=80000 or Path(ident['checkpoint']).name!='80000':raise RuntimeError('Expected80000 checkpoint identity')
    def request(self,path,payload=None):
        req=Request(self.url+path,data=payload,headers={'Content-Type':'application/json'},method='GET' if payload is None else 'POST')
        with urlopen(req,timeout=self.timeout) as response:return response.read()
    def health(self):
        result=json.loads(self.request('/healthz'))
        if result.get('ready') is not True or result.get('model_identity_sha256')!=self.expected_identity:raise RuntimeError('Actual model ready identity differs')
        return result
    def __call__(self,observation):
        imgs=observation['images']
        if set(imgs)!=set(self.CAMERAS):raise ValueError('Policy accepts exactly3camera observations')
        encoded={};hashes={}
        for key in self.CAMERAS:
            arr=np.asarray(imgs[key])
            if arr.dtype!=np.uint8 or arr.shape!=(480,640,3):raise ValueError('Real640x480RGB needed')
            buf=io.BytesIO();Image.fromarray(arr).save(buf,format='PNG');raw=buf.getvalue()
            encoded[key]=base64.b64encode(raw).decode('ascii');hashes[key]=hashlib.sha256(raw).hexdigest()
        state=np.asarray(observation['state'],dtype=float)
        if state.shape!=(24,) or not np.isfinite(state).all():raise ValueError('Measured24 required')
        prompt=observation['prompt']
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Static prompt required')
        payload=json.dumps({'images':encoded,'state24':state.tolist(),'prompt':prompt},allow_nan=False,separators=(',',':')).encode()
        start=time.monotonic();out=self.request('/infer',payload);result=json.loads(out)
        if result.get('model_identity_sha256')!=self.expected_identity or result.get('raw_commands_unclipped') is not True or result.get('expert_actions_used') is not False:
            raise RuntimeError('Response provenance differs')
        actions=np.asarray(result['actions'],dtype=float)
        if actions.shape!=(16,24) or not np.isfinite(actions).all():raise ValueError('FullH16physical24 required')
        self.records.append({'sim_time_s':float(observation['sim_time_s']),'request_sha256':hashlib.sha256(payload).hexdigest(),
            'response_sha256':hashlib.sha256(out).hexdigest(),'input_png_sha256':hashes,'input_state24':state.tolist(),
            'raw_actions':actions.tolist(),'wall_latency_s':time.monotonic()-start,'query_id':result.get('query_id'),
            'model_identity_sha256':result['model_identity_sha256'],'server_timing':result.get('timing')})
        if hashlib.sha256(self.identity_path.read_bytes()).hexdigest()!=self.expected_identity:raise RuntimeError('Identity changed during evaluation')
        return result

class PolicyClientAdapter:
    def __init__(self,client,prompt,output,stage_hook=None):
        self.client=client;self.prompt=prompt;self.output=Path(output);self.obs=None;self.stage_hook=stage_hook or (lambda *a:None)
    def call(self,func_name,**kwargs):
        if func_name=='reset':self.obs=None;return
        if func_name=='update_obs':self.obs=kwargs['obs'];return
        if func_name=='get_action':
            self.stage_hook('waiting_for_model');result=self.client(dict(self.obs,prompt=self.prompt))
            raw=np.asarray(result['actions']);np.savez_compressed(self.output/f'query_{len(self.client.records):05d}.npz',raw_prediction24=raw)
            self.stage_hook('executing_actions');return raw
        raise ValueError(func_name)
