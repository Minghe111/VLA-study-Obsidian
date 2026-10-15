"""CPU-only publication boundary tests; synthetic metadata and .bin blobs only.
Never fabricates an accepted physics trace, HDF5, or simulator episode.
"""
from pathlib import Path
import copy,json,shutil,tempfile,unittest
from unittest.mock import patch
import batch_dataset_publication_v30 as p

class PublicationBoundaries(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix='publisher_cpu_',dir=Path(__file__).parent)).resolve()
  self.assertTrue(self.root.is_relative_to(Path(__file__).parent.resolve()))
 def tearDown(self):
  self.assertTrue(self.root.is_relative_to(Path(__file__).parent.resolve()));shutil.rmtree(self.root)
 def spec(self):return dict(p._REQUIRED_SPEC,delivered_video_paths=dict(p._VIDEO_DEFAULT))
 def test_schema_rejects_wrong_count_bool_or_video(self):
  self.assertEqual(p._validate_spec(self.spec())['episode_num'],20)
  for key,value in [('episode_num',19),('qpose_tag',False),('model_id',12),('profile_version','v28'),('delivered_video_paths',{})]:
   s=self.spec();s[key]=value
   with self.assertRaises(ValueError):p._validate_spec(s)
 def test_index_seed_exact_int_boundaries(self):
  self.assertEqual(p._integer(19,'index',0,19),19)
  for value in [True,-1,20,0.5,'0']:
   with self.assertRaises(ValueError):p._integer(value,'index',0,19)
 def test_incomplete_replay_rejected_before_files(self):
  for r in [{},{'result':'physics_success','accepted':True},{'result':'one_strictly_validated_episode_collected','accepted':False},{'result':'one_strictly_validated_episode_collected','accepted':True,'canonical_sources_unchanged':False}]:
   with self.assertRaises(ValueError):p._check_flags(r)
  self.assertEqual(list(self.root.iterdir()),[])
 def test_unique_seeds_contiguous_official_indices(self):
  records=[{'index':i,'seed':7+i*3} for i in range(20)];p._validate_records(records)
  for bad in [records+[records[0]], [{'index':0,'seed':7},{'index':1,'seed':7}], [{'index':1,'seed':7}], [{'index':True,'seed':7}]]:
   with self.assertRaises(ValueError):p._validate_records(bad)
 def test_bytecopy_unchanged_and_conflict_preserved(self):
  src=self.root/'source.bin';src.write_bytes(bytes(range(256))*20);dst=self.root/'copies/copy.bin';wanted=p._sha(src)
  p._copy_verified(src,dst,wanted);p._copy_verified(src,dst,wanted);self.assertEqual(src.read_bytes(),dst.read_bytes())
  dst.write_bytes(b'conflict')
  with self.assertRaises(ValueError):p._copy_verified(src,dst,wanted)
  self.assertEqual(dst.read_bytes(),b'conflict')
 def test_stale_source_sha_rejected(self):
  src=self.root/'source.bin';src.write_bytes(b'initial');wanted=p._sha(src);src.write_bytes(b'changed')
  with self.assertRaises(ValueError):p._copy_verified(src,self.root/'copy.bin',wanted)
  self.assertFalse((self.root/'copy.bin').exists())
  with self.assertRaises(ValueError):p._check_hashes({str(src):wanted})
 def test_shared_source_must_be_frozen(self):
  src=self.root/'shared.bin';src.write_bytes(b'shared');m={str(src):p._sha(src)}
  self.assertEqual(p._check_hashes(m,m),m)
  with self.assertRaises(ValueError):p._check_hashes(m,{})
 def test_atomic_manifest_failure_preserves_previous(self):
  f=self.root/'mock_manifest.json';p._atomic_json(f,{'episodes':[]})
  with patch.object(p.os,'replace',side_effect=OSError('mock interrupted commit')):
   with self.assertRaises(OSError):p._atomic_json(f,{'episodes':[{'index':0,'seed':7}]})
  self.assertEqual(p._json(f),{'episodes':[]});self.assertEqual(len(list(self.root.glob('.*.tmp'))),1)
 def test_pending_resume_same_identity_only(self):
  f=self.root/'pending/mock_identity.json';identity={'index':0,'seed':7,'source_report_sha256':'1'*64}
  p._reserve_identity(f,identity);p._reserve_identity(f,identity)
  for delta in [{'seed':8},{'source_report_sha256':'2'*64}]:
   with self.assertRaises(ValueError):p._reserve_identity(f,dict(identity,**delta))
   self.assertEqual(p._json(f),identity)
 def test_finalize_partial_never_publishes_complete(self):
  root=self.root/p.BATCH_NAME/'dataset';root.mkdir(parents=True);expected={str(Path(p.__file__).resolve()):p._sha(p.__file__)};spec=self.spec()
  manifest={'batch_spec_sha256':p._digest(spec),'expected_sources':expected,'episodes':[{'index':i,'seed':7+i} for i in range(19)]}
  p._atomic_json(root/'manifest.json',manifest)
  with self.assertRaises(ValueError):p.finalize_dataset(root,expected,spec)
  self.assertFalse((root/'complete.json').exists())
 def test_api_returns_record_and_manifest_episodes_ast(self):
  import ast
  source=Path(p.__file__).read_text();ast.parse(source)
  self.assertIn("manifest['episodes'].append(record)",source)
  self.assertIn("return copy.deepcopy(record)",source)
  self.assertIn("'job_id':f'seed_{seed:07d}'",source)
  self.assertTrue(callable(p.finalize_dataset))

if __name__=='__main__':unittest.main(verbosity=2)
