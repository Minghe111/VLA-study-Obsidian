"""Loopback-only additive model service, separating model/simulator Conda.
RGB input only. Raw model commands are returned without expert substitutions.
"""
import argparse
import base64
import datetime
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from runtime40k import Runtime,sha,write_new
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run=args.output;run.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from PIL import Image
    runtime=Runtime(run)
    import data_adapter_rgb50_v1 as adapter
    requests=run/'requests';requests.mkdir(exist_ok=False)
    counter=0
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a): pass
        def response(self,status,value):
            payload=json.dumps(value,allow_nan=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path!='/health': return self.response(404,{'error':'unknown endpoint'})
            self.response(200,{'ready':True,'model_identity_sha256':runtime.identity_sha256,
                'checkpoint':runtime.identity['checkpoint'],'query_count':counter,'raw_commands':True})
        def do_POST(self):
            nonlocal counter
            if self.path=='/shutdown':
                self.response(200,{'stopping':True});threading.Thread(target=self.server.shutdown,daemon=True).start();return
            if self.path!='/infer': return self.response(404,{'error':'unknown endpoint'})
            target=None;tick=time.monotonic()
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=12*1024*1024: raise ValueError('Invalid request size')
                request=json.loads(self.rfile.read(size))
                if set(request)-{'images','state24','prompt','context'}: raise ValueError('Unexpected input keys')
                if set(request['images'])!=set(adapter.CAMERAS): raise ValueError('Native three-camera names required')
                state=request['state24']
                if not(isinstance(state,list) and len(state)==24 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in state)):
                    raise ValueError('Expected numeric nonboolean state24')
                state=np.asarray(state,dtype=np.float32)
                if not np.isfinite(state).all(): raise ValueError('Nonfinite measured state')
                prompt=request['prompt']
                if not isinstance(prompt,str) or not 0<len(prompt)<=4096: raise ValueError('Expected nonempty prompt')
                images={};hashes={}
                for key,payload in request['images'].items():
                    if not isinstance(payload,str): raise ValueError('Expected base64 image bytes')
                    encoded=base64.b64decode(payload,validate=True)
                    with Image.open(io.BytesIO(encoded)) as im:
                        if im.format not in ('PNG','JPEG') or im.mode!='RGB' or im.size!=(640,480):
                            raise ValueError('Expected unchanged native RGB640x480 PNG/JPEG')
                        pixels=np.asarray(im,dtype=np.uint8).copy()
                    images[key]=pixels;hashes[key]={'encoded_sha256':hashlib.sha256(encoded).hexdigest(),'rgb_pixels_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()}
                query=counter;counter+=1
                target=requests/('%06d'%query);target.mkdir(exist_ok=False)
                write_new(target/'input.json',{'query_id':query,'model_identity_sha256':runtime.identity_sha256,
                    'state24':state.tolist(),'prompt':prompt,'image_hashes':hashes,'context_log_only':request.get('context'),
                    'source_request_sha256':hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
                # Closed-loop RGB is retained losslessly; offline RGB is reproducible from immutable HDF and row provenance.
                context=request.get('context') or {}
                if not isinstance(context,dict) or context.get('kind')!='offline_validation':
                    np.savez_compressed(target/'input_rgb_state.npz',state24=state,**images)
                actions,timing=runtime.infer({'images':images,'state':state,'prompt':prompt})
                np.save(target/'raw_actions16x24.npy',actions,allow_pickle=False)
                result={'query_id':query,'actions':actions.tolist(),'timing':timing,
                    'request_wall_s':time.monotonic()-tick,'model_identity_sha256':runtime.identity_sha256,
                    'raw_actions_sha256':sha(target/'raw_actions16x24.npy'),'raw_commands_unclipped':True,'expert_actions_used':False}
                write_new(target/'result.json',result)
                self.response(200,result)
            except Exception as exc:
                result={'error':type(exc).__name__+': '+str(exc),'model_identity_sha256':runtime.identity_sha256}
                if target is not None: write_new(target/'error.json',result)
                self.response(422,result)
    server=HTTPServer(('127.0.0.1',18080),Handler)
    write_new(run/'ready.json',{'ready':True,'pid':os.getpid(),'bind':'127.0.0.1:18080','model_identity_sha256':runtime.identity_sha256})
    print('MODEL_SERVICE_READY',flush=True)
    def finish(sig,frame): threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,finish);signal.signal(signal.SIGINT,finish)
    try: server.serve_forever(poll_interval=.5)
    finally:
        server.server_close();write_new(run/'terminal.json',{'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'query_count':counter,'normal_service_shutdown':True,'task_success_claimed':False})

if __name__=='__main__': main()
