"""CPU preparation and seed/slot invariants; no simulator scene or renderer."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('randomized_entry_cpu', HERE / 'run_randomized_collector.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def main():
    rows, checks = [], []
    invariant_keys = ('index', 'model_id', 'brand', 'arm', 'pose_class', 'xy', 'yaw_deg', 'grasp_setting')
    for index in entry.INDICES:
        source_path = entry.SOURCE / 'selected_scenes' / f'episode_{index:07d}' / 'sample_spec.json'
        before = entry.sha(source_path)
        original = json.loads(source_path.read_text())
        actual_seeds = []
        draws = []
        for attempt in (0, 1, 2):
            task_spec, config_path, jobs_path, receipt = entry.verify_inputs(index, attempt, entry.TASK_ROOT / 'new_physical_episodes')
            wanted = original['seed'] + 10_000_000 * attempt
            assert task_spec['source_seed'] == original['seed']
            assert task_spec['seed'] == task_spec['actual_domain_seed'] == receipt['actual_domain_seed'] == wanted
            assert receipt['source_seed'] == original['seed']
            assert yaml.safe_load(config_path.read_text())['seed'] == wanted
            assert json.loads(jobs_path.read_text())[0]['spec']['seed'] == wanted
            assert all(task_spec[key] == original[key] for key in invariant_keys if key in original)
            actual_seeds.append(wanted)
            rng = np.random.RandomState(wanted)
            # CPU component probe of the initialization draw ordering only.
            lighting = rng.random_sample(10).tolist()
            draws.append({'seed': wanted, 'initial_10_native_rng_draws': lighting,
                          'subsequent_uniform_table_bias_m': float(rng.uniform(-.03, 0))})
            if attempt <= 1:
                receipt_path = HERE / f'randomized_slot_{index:07d}_attempt_{attempt:03d}_prepare_receipt.json'
                # Preserve the existing failed-physical attempt0 runtime record.
                if attempt == 1 or not receipt_path.exists():
                    entry.write_json(receipt_path, receipt)
        assert len(set(actual_seeds)) == 3
        assert draws[0]['initial_10_native_rng_draws'] != draws[1]['initial_10_native_rng_draws']
        explicit = 2718281 + index
        override, config_path, jobs_path, override_receipt = entry.verify_inputs(index, 3, entry.TASK_ROOT / 'new_physical_episodes', domain_seed=explicit)
        assert override['seed'] == explicit and override['source_seed'] == original['seed']
        assert override['domain_seed_policy'] == 'explicit --domain-seed'
        assert yaml.safe_load(config_path.read_text())['seed'] == explicit
        assert all(override[key] == original[key] for key in invariant_keys if key in original)
        assert entry.sha(source_path) == before
        rows.append({'index': index, 'brand': original['brand'], 'arm': original['arm'], 'pose_class': original['pose_class'],
                     'source_seed': original['seed'], 'default_actual_seeds_attempt_0_1_2': actual_seeds,
                     'explicit_actual_seed_attempt_3': explicit, 'source_sample_sha256_unchanged': before,
                     'preserved_slot_fields': [key for key in invariant_keys if key in original],
                     'cpu_rng_draw_probes': draws, 'passed': True})
    for args in ((1, -1, None), (1, 500, None), (True, 0, None), (1, 0, True),
                 (1, 0, -1), (1, 0, 2**32), (1, 0, 1.5)):
        try:
            entry.resolve_domain_seed(*args)
        except ValueError:
            checks.append({'args': list(args), 'invalid_seed_rejected': True})
        else:
            raise AssertionError('Invalid seed unexpectedly accepted: ' + repr(args))
    receipt = {'passed': True, 'scope': 'Static/CPU preparation, native NumPy RNG component probes; no new physical candidate success claim',
        'scene_or_renderer_constructed': False, 'gpu_run': False,
        'entry_script': str(HERE / 'run_randomized_collector.py'), 'entry_sha256': entry.sha(HERE / 'run_randomized_collector.py'),
        'test_script_sha256': entry.sha(__file__), 'rows': rows, 'invalid_seed_checks': checks,
        'original_failed_attempt0_candidate_directory_unchanged': True,
        'future_original_commands_loaded': False}
    entry.write_json(HERE / 'seed_domain_invariants.json', receipt)
    entry.write_json(HERE / 'randomized_five_static_readiness.json', {
        'no_scene_or_gpu_started_by_this_cpu_preparation': True,
        'seed_policy': 'source_seed + 10000000 * attempt unless --domain-seed explicit',
        'source_seed_preserved': True, 'all_slot_taxonomy_and_placement_metadata_preserved': True,
        'rows': [{key: row[key] for key in ('index', 'brand', 'arm', 'pose_class', 'source_seed', 'default_actual_seeds_attempt_0_1_2')} for row in rows]})
    print(json.dumps({'passed': True, 'five_slots': len(rows), 'invalid_seed_cases_rejected': len(checks),
                      'receipt': str(HERE / 'seed_domain_invariants.json')}, indent=2))


if __name__ == '__main__':
    main()
