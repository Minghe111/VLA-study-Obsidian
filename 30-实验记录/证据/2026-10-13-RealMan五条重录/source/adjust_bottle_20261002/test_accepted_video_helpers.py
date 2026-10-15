"""Pure CPU tests for accepted-video temporal provenance; no simulation or GPU."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from export_timed_preview_v12 import export_head_realtime
from accepted_video_qa import _iter_json_array, _clock, _key_records, _select_keyframes, _compact_trace


class TemporalProvenanceTests(unittest.TestCase):
    def test_streaming_json_chunk_boundaries(self):
        records=[{"step":i,"label":"边界🙂","nested":{"a":list(range(15))}} for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"trace.json"
            path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
            for n in (1,7,31,1024):
                self.assertEqual(list(_iter_json_array(path,chunk_size=n)),records)
            path.write_text('[{"step":1},',encoding="utf-8")
            with self.assertRaises(ValueError):list(_iter_json_array(path,chunk_size=3))

    def _replay(self,root,steps=(0,15,15,30,45,60)):
        root=Path(root);(root/"video").mkdir();cache=root/".cache/episode0";cache.mkdir(parents=True)
        dt=.004000000189989805
        report={"result":"one_strictly_validated_episode_collected","accepted":True,"physical_steps":60,
                "dataset":{"physical_step_s":dt,"recorded_frames":len(steps)}}
        (root/"replay_report.json").write_text(json.dumps(report),encoding="utf-8")
        originals={}
        for index,step in enumerate(steps):
            p=cache/f"{index}.pkl"
            p.write_bytes(pickle.dumps({"recording_clock":{"physics_step":step,"sim_time_s":step*dt},
                  "observation":{"head_camera":{"rgb":np.full((240,320,3),index*35,dtype=np.uint8)}}}))
            originals[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        return originals

    def test_real_codec_duplicate_last_absolute_time_and_immutable_source(self):
        with tempfile.TemporaryDirectory() as d:
            originals=self._replay(d);r=export_head_realtime(d)
            self.assertEqual(r["decoded_frames"],8)
            self.assertEqual(r["source_cache_index"],[0,0,2,2,3,3,4,4])
            self.assertEqual(r["source_physics_steps"],[0,0,15,15,30,30,45,45])
            self.assertEqual(r["duplicate_source_clock_count"],1)
            self.assertLessEqual(r["max_source_age_s"],.06+1e-8)
            self.assertEqual(r["final_endpoint_source_hold_s"],0)
            for p,sha in originals.items():self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(),sha)
            t,source,steps,age=_clock(r,"head",30.)
            self.assertTrue(np.array_equal(t,np.arange(8)/30))
            self.assertTrue(np.allclose(source,steps*r["physical_step_s"],atol=1e-12))
            with self.assertRaises(FileExistsError):export_head_realtime(d)

    def test_stale_clock_gap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self._replay(d,(0,30,60))
            with self.assertRaisesRegex(ValueError,"No current raw sample"):export_head_realtime(d)

    def test_future_source_and_head_age_rejected(self):
        meta={"frame_times_s":[0,1/30],"source_observation_times_s":[0,.04],"source_physics_steps":[0,10]}
        with self.assertRaisesRegex(ValueError,"future"):_clock(meta,"head",30.)
        meta={"frame_times_s":[0,1/30,2/30],"source_observation_times_s":[0,0,0],"source_physics_steps":[0,0,0]}
        with self.assertRaisesRegex(ValueError,".06"):_clock(meta,"head",30.)

    def test_lift_turn_labels_use_saved_action_provenance(self):
        def record(step,name,dense=None):
            return {"step":step,"phase":{"name":name,"dense_call":dense},
                    "grip":{"contacts":[{"link":"l_"+k,"normal_force_N":.3} for k in ("middle","index","thumb")]}}
        trace=[record(1,"approach"),record(2,"close"),record(3,"close_hold"),
               record(4,"after_grasp",4),record(5,"after_grasp",5),record(6,"after_grasp",6),record(7,"after_grasp",7)]
        events=dict(_key_records(trace,{"replay_wrench_actions":[{"action_dense_call":5},{"action_dense_call":6}]}))
        self.assertEqual(events["lift_end"]["phase"]["dense_call"],5)
        self.assertEqual(events["turn_end"]["phase"]["dense_call"],6)

    def test_key_contact_frame_is_after_actual_threshold_not_before(self):
        events=[("first_main3_contact",{"step":3510}),("closed_hold_end",{"step":6521})]
        self.assertEqual(_select_keyframes(events,np.array([3501,3516,6507,6522])),[1,2])

    def test_inactive_physx_shells_are_not_active_self_contacts(self):
        r={"step":1,"sim_time_s":.004,"phase":{"name":"approach"},
           "bottle":{"pose":np.eye(4).tolist(),"functional_point0":[0,0,0]},"grip":{},
           "actual_self_contacts":[{"active":False,"normal_impulse_Ns":0.},{"active":True,"normal_impulse_Ns":.01}]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"trace.json";p.write_text(json.dumps([r]),encoding="utf-8")
            compact=_compact_trace(p)[0]
            self.assertEqual(len(compact["actual_self_contacts"]),1)
            self.assertEqual(compact["inactive_self_contact_shells"],1)


if __name__=="__main__":unittest.main(verbosity=2)
