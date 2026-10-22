#!/usr/bin/env python3
"""Evidence-based acceptance for the explicitly scoped empty-bottle workflow."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--agent-receipt',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'pipeline_acceptance.json')
    args=p.parse_args()
    read=lambda path:json.loads(Path(path).read_text())
    metadata=read(ROOT/'metadata_validation.json')
    assert metadata['passed'] and len(metadata['tests'])==6
    assert all(v['original_enrich_actors_semantics_read'] for v in metadata['tests'])
    agent=read(args.agent_receipt)
    collection=read(args.collection/'collection_receipt.json')
    launch=read(args.collection/'launch.json')
    independent=read(args.collection/'independent_dataset_audit.json')
    assert independent['passed'] and len(independent['episodes'])==collection['validated_episodes']
    assert agent['original_test_gen_code_run_called'] and agent['status']=='physical_test_complete'
    assert agent['official_threshold_0_5_met'] and agent['source_unchanged'] and agent['candidate_unchanged']
    assert agent['check_num']==10,'Official default 10-trial test was not completed'
    assert collection['status']=='passed' and collection['official_source_unchanged']
    assert collection['validated_episodes']==collection['requested_episodes']>=1
    assert launch['expert']['tested_candidate_sha256']==agent['candidate_sha256']
    assert launch['expert']['kind']=='current_session_candidate_officially_tested'
    assert launch['unchanged_official_entry']=='scripts.collect_data.run'
    for v in collection['validations']:
        path=Path(v['hdf5']);assert v['passed'] and v['state_shape'][1]==v['action_shape'][1]==24
        assert v['actual_drive_readback_matches_action_exact'] and v['measured_state_chain_exact']
        assert v['control_dt_seconds']==.06 and len(v['cameras'])==3
        assert all(cam['decoded_frames']==v['frames'] for cam in v['cameras'].values())
        assert v['language_count']==12 and v['physical_final']['success']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==v['sha256'],'HDF5 changed after actual readback'
    head_margins=[]
    for episode in independent['episodes']:
        assert episode['plan_physical_success'] and episode['replay_physical_success']
        assert episode['cross_camera_keyframe_identity_check']['all_views_distinct']
        head=episode['head_framing_geometry_check']
        assert head['applied'] and head['all_target_geometry_inside_with_at_least_40px_margin']
        assert head['stored_observations_checked']==episode['frames_per_camera']
        head_margins.append(head['minimum_margin_pixels'])
    old=read(ROOT/'metadata/registry.json')['frozen_input_fingerprints']
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in old.items())
    result={'status':'passed_for_empty_bottle_pick_hold_place_collection',
            'date':'2026-10-22','scope':'RealMan RM65-B simulation; empty open reagent bottle; small declared near-arm reset region',
            'agent_generation':'current Codex session authorized by user, original Prompt and original physical tester; no API request',
            'official_test_trials':agent['check_num'],'official_test_success_rate':agent['success_rate'],
            'actual_collected_episodes':collection['validated_episodes'],
            'collected_frames':[v['frames'] for v in collection['validations']],
            'data_contract':'three simulated RGB cameras, actual measured state24, actual drive readback action24, 60ms',
            'frozen_asset_fingerprints_unchanged':True,
            'independent_dataset_audit_passed':True,'minimum_head_target_margin_pixels':min(head_margins),
            'evidence':{'metadata':str(ROOT/'metadata_validation.json'),'agent':str(args.agent_receipt.resolve()),
                        'collection':str((args.collection/'collection_receipt.json').resolve()),
                        'independent_dataset_audit':str((args.collection/'independent_dataset_audit.json').resolve())},
            'remaining_operations':{'robot_unscrewing':'not validated','robot_pouring_tablets':'not validated',
                                    'other_five_asset_grasp_collection':'not validated','physical_hardware':'not tested'},
            'all_labware_operations_ready':False}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
