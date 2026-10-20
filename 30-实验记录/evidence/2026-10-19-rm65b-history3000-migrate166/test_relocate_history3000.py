import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import torch
import numpy as np

spec=importlib.util.spec_from_file_location('relocator',Path(__file__).with_name('relocate_history3000.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RelocationTests(unittest.TestCase):
    def test_longest_prefix_only_and_digest(self):
        obj={'a':['/old/root/a','prefix /old/root/a','/old/rootish/a','digest'],'b':'/old/root/narrow/a'}
        out=m.replace_tree(obj,{'/old/root':'/new','/old/root/narrow':'/specific'},{'digest':'newdigest'})
        self.assertEqual(out,{'a':['/new/a','prefix /old/root/a','/old/rootish/a','newdigest'],'b':'/specific/a'})
        self.assertEqual(obj['a'][0],'/old/root/a')

    def test_deep_optimizer_exact_guard(self):
        state={'optimizer':{'state':{0:{'step':torch.tensor(2000.),'exp_avg':torch.arange(5.)}}},
               'rng_state':{'torch':torch.arange(30,dtype=torch.uint8)},'sampler_state':{'data_fingerprint':'old','order':[3,1,2]}}
        changed=copy.deepcopy(state);changed['sampler_state']['data_fingerprint']='new'
        self.assertEqual(m.same_tree(state,changed),['root.sampler_state.data_fingerprint'])
        changed['optimizer']['state'][0]['exp_avg'][0]+=1
        self.assertIn('root.optimizer.state.0.exp_avg',m.same_tree(state,changed))

    def test_sha_closure_and_original_bytes_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a.json';b=root/'b.json';c=root/'c.json'
            a.write_text('{"file": "/old/root/npz"}\n');ah=m.sha(a)
            b.write_text(json.dumps({'a':{'path':'/old/root/a.json','sha256':ah}}));bh=m.sha(b)
            c.write_text(json.dumps({'b':{'path':'/old/root/b.json','sha256':bh}}));ch=m.sha(c)
            original={p.name:p.read_bytes() for p in (a,b,c)}
            manifest=root/'spec.json';m.write_json(manifest,{'path_map':{'/old/root':str(root)},
                'files':[{'source':'/old/root/'+p.name,'target':str(p),'sha256':m.sha(p)} for p in (a,b,c)]})
            result=m.structured(SimpleNamespace(manifest=manifest,output=root/'proof'))
            self.assertEqual(json.loads(b.read_text())['a']['sha256'],m.sha(a))
            self.assertEqual(json.loads(c.read_text())['b']['sha256'],m.sha(b))
            for row in result['files']:self.assertEqual(Path(row['original_backup']).read_bytes(),original[Path(row['target']).name])

    def test_reject_wrong_copy_before_editing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a.json';a.write_text('{}')
            manifest=root/'spec.json';m.write_json(manifest,{'path_map':{'/old':str(root)},'files':[{'source':'/old/a.json','target':str(a),'sha256':'0'*64}]})
            with self.assertRaisesRegex(ValueError,'Copied source SHA'):m.structured(SimpleNamespace(manifest=manifest,output=root/'proof'))
            self.assertEqual(a.read_text(),'{}');self.assertFalse((root/'proof').exists())

    def test_fingerprint_resolves_only_byte_reads(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'obs.npz';np.savez(p,a=np.arange(3))
            streams={'high':[{'observation':{'npz_path':'/old/obs.npz'}}],'actions':[],'anchor_high':[],'anchor_actions':[]}
            before=copy.deepcopy(streams)
            digest=m.fingerprint_with_resolver(streams,{'/old':d})
            self.assertEqual(streams,before);self.assertEqual(len(digest),64)
            np.savez(p,a=np.arange(4))
            self.assertNotEqual(digest,m.fingerprint_with_resolver(streams,{'/old':d}))

    def test_reject_retention_manifest_rewrite_before_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'manifest.json'
            a.write_text(json.dumps({'format':'rm65b_k1_retention_anchor_v1','source_episode':'/old/episode.json'}))
            before=a.read_bytes();manifest=root/'spec.json'
            m.write_json(manifest,{'path_map':{'/old':str(root)},'files':[{'source':'/old/manifest.json','target':str(a),'sha256':m.sha(a)}]})
            with self.assertRaisesRegex(ValueError,'sealed by reference snapshot SHA'):
                m.structured(SimpleNamespace(manifest=manifest,output=root/'proof'))
            self.assertEqual(a.read_bytes(),before);self.assertFalse((root/'proof').exists())

if __name__=='__main__':unittest.main()
