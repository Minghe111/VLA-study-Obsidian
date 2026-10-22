"""Check that delivered evidence still refers to these exact local assets.

This validates existing records; it does not rerun the physical experiment.
"""
from pathlib import Path
import hashlib
import json

from PIL import Image
from annotation_contract import validate_annotations

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hash(path, expected):
    assert sha(path) == expected, f'File changed: {path}'


def main():
    interface = read('asset_interface.json')
    annotations = read('annotation_specs.json')
    audit = read('acceptance_annotation_audit.json')
    physics = read('physics_validation_summary.json')
    scene = read('scene_render_receipt.json')
    replay = read('physics_render_receipt.json')
    expected_ids = {'beaker','erlenmeyer','graduated_cylinder','petri_dish',
                    'reagent_bottle_body','reagent_bottle_cap'}
    assert {item['id'] for item in interface['assets']} == expected_ids
    for name,digest in audit['frozen_files'].items():
        check_hash(HERE/name,digest)
    assert audit['status'] == 'passed_with_declared_approximations'
    actual_rows = {row['id']:row for row in annotations['prepared_asset_validation']}
    counts = {}
    for item in interface['assets']:
        ident = item['id']
        data = read(f'prepared/{ident}/model_data0.json')
        assert validate_annotations(data)['passed']
        row = actual_rows[ident]
        check_hash(row['annotation_file'],row['annotation_sha256'])
        check_hash(item['visual_file'],item['visual_sha256'])
        check_hash(item['visual_glb'],row['official_loader']['visual_glb_sha256'])
        check_hash(item['collision_glb'],row['official_loader']['collision_glb_sha256'])
        assert row['official_loader']['passed']
        for part in item['collision_parts']:
            check_hash(part['filename'],part['sha256'])
        assert row['official_loader']['collision_shapes'] == len(item['collision_parts'])
        if ident != 'reagent_bottle_cap':
            assert row['aperture_drop']['passed']
        counts[ident] = row['official_loader']['collision_shapes']
    cap = read('prepared/reagent_bottle_cap/model_data0.json')
    assert cap['annotation_provenance']['mass_kg'] == read('physics_spec.json')['cap']['mass_kg'] == .012
    for source in read('source_manifest.json')['sources']:
        check_hash(source['source_file'],source['source_sha256'])
        if 'converted_source_obj' in source:
            check_hash(source['converted_source_obj'],source['converted_obj_sha256'])

    assert physics['all_passed'] and physics['negative_control']['passed']
    assert len(physics['final_runs']) == 3
    verified_runs = []
    for entry in physics['final_runs']:
        report_path = Path(entry['report'])
        if not report_path.is_absolute():
            report_path = HERE/report_path.name
        report = json.loads(report_path.read_text())
        check_hash(report_path,entry['report_sha256'])
        for name,digest in report['source_sha256'].items():
            check_hash(HERE/name,digest)
        assert report['passed'] and report['closed_inverted_min_inside'] == 12
        assert report['closed_inverted_checked_physics_steps'] == 2500
        assert report['final_receiver_count'] == 12 and report['remaining_bottle_count'] == 0
        assert report['tablets_pose_writes_after_initialization'] == 0
        assert report['contact_pair_step_counts']['tablet_floor'] == 0
        trace_path = HERE/f"physics_final_seed{report['seed']}_trace.json"
        check_hash(trace_path,entry['trace_sha256'])
        trace = json.loads(trace_path.read_text())
        assert len(trace) == 730 and trace[-1]['step'] == 14600
        assert all(len(frame['tablets']) == 12 for frame in trace)
        verified_runs.append({'seed':report['seed'],'steps':report['physics_steps'],
                              'receiver_count':report['final_receiver_count']})

    check_hash(replay['input_trace'],replay['input_trace_sha256'])
    check_hash(replay['input_physics_report'],replay['input_physics_report_sha256'])
    check_hash(HERE/'render_physics_record.py',replay['source_script_sha256'])
    check_hash(replay['video']['path'],replay['video']['sha256'])
    assert replay['new_physics_steps'] == 0 and replay['video']['readback_passed']
    assert replay['video']['frames'] == 219 and replay['video']['fps'] == 15
    check_hash(HERE/'asset_interface.json',scene['asset_interface_sha256_at_load'])
    check_hash(HERE/'scene_layout.json',scene['scene_layout_sha256'])
    assert len(scene['official_loader_actors']) == 6
    assert scene['physics_steps'] == 0 and not scene['hardware_io']
    image_files = sorted((HERE/'scene_visualization').glob('*.png'))
    image_files += sorted((HERE/'physics_visualization').glob('*.png'))
    assert len(image_files) >= 14
    for path in image_files:
        with Image.open(path) as image:
            image.load()

    # Only source/artifact hashes: no claims about real robot task performance.
    deliverables = [path for path in HERE.iterdir() if path.is_file() and
                    path.suffix in {'.py','.json','.md'} and
                    path.name not in {'acceptance_report.json','task_status.json'}]
    result = {'date':'2026-10-22','passed':True,'prepared_assets':6,
              'official_loader_convex_shape_counts':counts,
              'open_container_aperture_drop_tests_passed':5,
              'fixture_physics_runs_verified':verified_runs,
              'blocked_cap_negative_control':True,
              'cap_mass_contract_kg':.012,'decoded_png_files':len(image_files),
              'scene_views':len(scene['views']),
              'replay_video':replay['video'],
              'artifact_sha256':{path.name:sha(path) for path in sorted(deliverables)},
              'scope':'Imported and derived assets, annotations, official loader, and fixture-driven physical chain. No autonomous robot task, real thread contact, liquid simulation, or VLA/RL training certified.'}
    (HERE/'acceptance_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
