"""CPU-only release-contract checks with synthetic frozen stage receipts."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import json
import unittest

import hybrid_release_contract as release


UNIT = 'realman-rerecord-20261014-v6-hybrid-collect-123.service'
ZERO_STAT_PROOF = {'memory.stat.swap_before_bytes': 0,
                   'memory.stat.total_swap_before_bytes': 0,
                   'memory.stat.swap_after_bytes': 0,
                   'memory.stat.total_swap_after_bytes': 0,
                   'memory.usage_in_bytes.raw': 1048576,
                   'memory.memsw.usage_in_bytes.raw': 1048576,
                   'memory.memsw_minus_memory.diagnostic_bytes': 0}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + '\n')


class ReleaseContractTest(TestCase):
    def fixture(self, root, *, negative_max=1, negative_memsw=0,
                positive_memsw=0):
        preflight = root / 'preflight'
        preflight.mkdir()
        for name in ('run_guarded_stage.py', 'hybrid_cgroup_adapter.py',
                     'hybrid_release_contract.py', 'guarded_stage_cpu_probe.py'):
            (preflight / name).write_text('# fixed ' + name + '\n')
        probe_dir = preflight / 'resource_runs/cpu_probe_1'
        results = []
        for negative in (False, True):
            unit = ('realman-rerecord-20261014-v6-hybrid-probe-negative-2.service'
                    if negative else 'realman-rerecord-20261014-v6-hybrid-probe-positive-1.service')
            phase = probe_dir / ('negative' if negative else 'positive')
            readback = {'limits_verified': True, 'mode': 'v1', 'actual': {
                'memory.soft_limit_in_bytes': str(24 * 1024**2),
                'memory.limit_in_bytes': str(32 * 1024**2),
                'memory.memsw.limit_in_bytes': str(32 * 1024**2),
                'memory.swappiness': '0', 'cpu.cfs_quota_us': '100000',
                'cpu.cfs_period_us': '100000', 'pids.max': '32'},
                'controller_paths': {
                'memory': f'/sys/fs/cgroup/memory/system.slice/{unit}',
                'cpu': f'/sys/fs/cgroup/cpu,cpuacct/system.slice/{unit}',
                'pids': f'/sys/fs/cgroup/pids/system.slice/{unit}'}}
            events = {'max': negative_max if negative else 0,
                      'oom': negative_max if negative else 0,
                      'oom_kill': 1 if negative else 0, 'oom_group_kill': 0,
                      'memsw_max': negative_memsw if negative else positive_memsw}
            resource = {'mode': 'v1', 'memory.events': events,
                        'memory.oom_control': {'under_oom': 0,
                                               'oom_kill': 1 if negative else 0},
                        'memory.swap.current': 0, **ZERO_STAT_PROOF}
            terminal = {'passed': True, 'source_unchanged': True,
                        'child_returncode': -9 if negative else None,
                        'resource': resource}
            dump(phase / 'cgroup_readback.json', readback)
            dump(phase / 'probe_terminal.json', terminal)
            results.append({'negative': negative, 'unit': unit, 'passed': True,
                            'launcher_returncode': 0, 'monitor_stop_reason': None,
                            'swap_zero_observed': True,
                            'cgroup_readback': readback, 'probe_terminal': terminal})
        probe = {'passed': True, 'results': results,
                 'guard_sha256': release.sha(preflight / 'run_guarded_stage.py'),
                 'adapter_sha256': release.sha(preflight / 'hybrid_cgroup_adapter.py'),
                 'probe_sha256': release.sha(preflight / 'guarded_stage_cpu_probe.py')}
        probe_path = probe_dir / 'receipt.json'
        dump(probe_path, probe)
        stage_dir = preflight / 'resource_runs/collect_1'
        resource = {'mode': 'v1', 'memory.events': {'max': 0, 'oom': 0,
                    'oom_kill': 0, 'oom_group_kill': 0, 'memsw_max': 0},
                    'memory.oom_control': {'under_oom': 0}, 'memory.swap.current': 0,
                    **ZERO_STAT_PROOF}
        worker = {'stage_returncode': 0, 'source_unchanged': True, 'resource': resource}
        dump(stage_dir / 'worker_terminal.json', worker)
        sample = {'elapsed_seconds': 1., 'cgroup': {**resource, 'swap_zero_observed': True,
                            'member_vm_swap_bytes': {'123': 0}},
                  'unit_state': {'ControlGroup': '/system.slice/' + UNIT}}
        samples = stage_dir / 'resources.jsonl'
        samples.write_text(json.dumps(sample) + '\n')
        readback = {'limits_verified': True, 'mode': 'v1', 'actual': {
                    'memory.soft_limit_in_bytes': str(10 * release.GIB),
                    'memory.limit_in_bytes': str(12 * release.GIB),
                    'memory.memsw.limit_in_bytes': str(12 * release.GIB),
                    'memory.swappiness': '0', 'cpu.cfs_quota_us': '800000',
                    'cpu.cfs_period_us': '100000', 'pids.max': '128'},
                    'controller_paths': {
                    'memory': f'/sys/fs/cgroup/memory/system.slice/{UNIT}',
                    'cpu': f'/sys/fs/cgroup/cpu,cpuacct/system.slice/{UNIT}',
                    'pids': f'/sys/fs/cgroup/pids/system.slice/{UNIT}'}}
        dump(stage_dir / 'cgroup_readback.json', readback)
        receipt = {'stage': 'collect', 'unit': UNIT,
                   'source_binding': {
                       'guard_sha256': probe['guard_sha256'],
                       'cgroup_adapter_sha256': probe['adapter_sha256'],
                       'release_contract_sha256': release.sha(preflight / 'hybrid_release_contract.py')},
                   'preflight_probe': {'path': str(probe_path),
                       'sha256': release.sha(probe_path), 'passed': True,
                       'guard_sha256': probe['guard_sha256'],
                       'adapter_sha256': probe['adapter_sha256'],
                       'probe_sha256': probe['probe_sha256']},
                   'worker_terminal': worker, 'swap_zero_observed': True,
                   'resource_samples_sha256': release.sha(samples),
                   'swap_observation_count': 1, 'maximum_sample_gap_seconds': .5,
                   'tail_sample_gap_seconds': .5,
                   'gate_release_elapsed_seconds': .5,
                   'unit_exit_elapsed_seconds': 1.5, 'wall_seconds': 1.8,
                   'monitor_stop_reason': None, 'launcher_returncode': 0,
                   'passed': True}
        receipt_path = stage_dir / 'receipt.json'
        dump(receipt_path, receipt)
        return preflight, receipt_path, receipt, stage_dir

    def test_good_stage_and_probe(self):
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, _ = self.fixture(Path(tmp))
            self.assertTrue(release.validate_stage(receipt, path, 'collect', preflight)['passed'])

    def test_v1_memsw_only_negative_is_valid_but_no_counter_is_not(self):
        # Real 161 probe: the 32 MiB memsw cap killed the child while
        # memory.failcnt stayed at zero.
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, _ = self.fixture(
                Path(tmp), negative_max=0, negative_memsw=42)
            self.assertTrue(release.validate_stage(receipt, path, 'collect', preflight)['passed'])
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, _ = self.fixture(
                Path(tmp), negative_max=0, negative_memsw=0)
            with self.assertRaisesRegex(ValueError, 'cap OOM kill'):
                release.validate_stage(receipt, path, 'collect', preflight)

    def test_positive_memsw_counter_must_remain_zero(self):
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, _ = self.fixture(Path(tmp), positive_memsw=1)
            with self.assertRaisesRegex(ValueError, 'Positive probe touched'):
                release.validate_stage(receipt, path, 'collect', preflight)

    def test_probe_swap_kernel_and_source_tamper_rejected(self):
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, stage_dir = self.fixture(Path(tmp))
            bad = dict(receipt)
            bad['swap_zero_observed'] = False
            with self.assertRaises(ValueError):
                release.validate_stage(bad, path, 'collect', preflight)
            (preflight / 'run_guarded_stage.py').write_text('# modified\n')
            with self.assertRaises(ValueError):
                release.validate_stage(receipt, path, 'collect', preflight)
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, stage_dir = self.fixture(Path(tmp))
            readback = json.loads((stage_dir / 'cgroup_readback.json').read_text())
            readback['actual']['memory.limit_in_bytes'] = '0'
            dump(stage_dir / 'cgroup_readback.json', readback)
            with self.assertRaises(Exception):
                release.validate_stage(receipt, path, 'collect', preflight)
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, _ = self.fixture(Path(tmp))
            probe_path = Path(receipt['preflight_probe']['path'])
            probe_path.write_text('{}\n')
            with self.assertRaises(ValueError):
                release.validate_stage(receipt, path, 'collect', preflight)

    def test_delayed_gate_and_cleanup_are_outside_live_stage(self):
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, stage_dir = self.fixture(Path(tmp))
            sample = json.loads((stage_dir / 'resources.jsonl').read_text())
            sample['elapsed_seconds'] = 9.5
            samples = stage_dir / 'resources.jsonl'
            samples.write_text(json.dumps(sample) + '\n')
            receipt.update({'gate_release_elapsed_seconds': 9.,
                            'unit_exit_elapsed_seconds': 10.,
                            'wall_seconds': 30.,
                            'resource_samples_sha256': release.sha(samples)})
            self.assertTrue(release.validate_stage(receipt, path, 'collect', preflight)['passed'])

    def test_live_gap_and_timing_tamper_are_rejected(self):
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, stage_dir = self.fixture(Path(tmp))
            samples = stage_dir / 'resources.jsonl'
            sample = json.loads(samples.read_text())
            sample['elapsed_seconds'] = 7.
            samples.write_text(json.dumps(sample) + '\n')
            receipt.update({'gate_release_elapsed_seconds': .5,
                            'unit_exit_elapsed_seconds': 7.5,
                            'wall_seconds': 8.,
                            'maximum_sample_gap_seconds': 6.5,
                            'resource_samples_sha256': release.sha(samples)})
            with self.assertRaisesRegex(ValueError, 'bounded interval'):
                release.validate_stage(receipt, path, 'collect', preflight)
            receipt['maximum_sample_gap_seconds'] = .5
            with self.assertRaisesRegex(ValueError, 'timing evidence'):
                release.validate_stage(receipt, path, 'collect', preflight)
        with TemporaryDirectory() as tmp:
            preflight, path, receipt, _ = self.fixture(Path(tmp))
            for changed in ({'gate_release_elapsed_seconds': 1.1},
                            {'unit_exit_elapsed_seconds': .9},
                            {'unit_exit_elapsed_seconds': None},
                            {'gate_release_elapsed_seconds': None},
                            {'wall_seconds': float('nan')},
                            {'tail_sample_gap_seconds': 0.}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    release.validate_stage({**receipt, **changed}, path, 'collect', preflight)

    def test_middle_tail_and_ordering_gaps_are_rejected(self):
        for times, unit_exit, maximum, tail in (
                ([1., 8.], 8.5, 7., .5),  # missing intermediate samples
                ([1.], 8., 7., 7.),      # no observation near unit exit
                ([2., 1.], 2.5, 1.5, 1.5)):  # reversed sample clock
            with self.subTest(times=times), TemporaryDirectory() as tmp:
                preflight, path, receipt, stage_dir = self.fixture(Path(tmp))
                sample = json.loads((stage_dir / 'resources.jsonl').read_text())
                samples = stage_dir / 'resources.jsonl'
                samples.write_text(''.join(json.dumps({**sample, 'elapsed_seconds': elapsed})
                                           + '\n' for elapsed in times))
                receipt.update({'unit_exit_elapsed_seconds': unit_exit,
                                'wall_seconds': unit_exit + .5,
                                'swap_observation_count': len(times),
                                'maximum_sample_gap_seconds': maximum,
                                'tail_sample_gap_seconds': tail,
                                'resource_samples_sha256': release.sha(samples)})
                with self.assertRaises(ValueError):
                    release.validate_stage(receipt, path, 'collect', preflight)


if __name__ == '__main__':
    unittest.main()
