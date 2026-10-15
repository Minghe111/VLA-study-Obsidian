"""CPU-only static tests for the dedicated system-unit stage guard.

These tests never call systemd, sudo, a simulator, or a GPU. The builder may
copy this file as evidence/preflight/test_guarded_stage.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TARGET = HERE / 'run_guarded_stage.py'
if not TARGET.exists():
    TARGET = HERE / 'run_guarded_stage_system_v6.py'
SPEC = importlib.util.spec_from_file_location('system_v6_guard_under_test', TARGET)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)
PROBE_TARGET = HERE / 'guarded_stage_cpu_probe.py'
if not PROBE_TARGET.exists():
    PROBE_TARGET = HERE / 'guarded_stage_cpu_probe_system_v6.py'
PROBE_SPEC = importlib.util.spec_from_file_location('system_v6_probe_under_test', PROBE_TARGET)
probe = importlib.util.module_from_spec(PROBE_SPEC)
PROBE_SPEC.loader.exec_module(probe)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.old = (guard.COLLECT_CPU_PERCENT, guard.UNIT_PREFIX,
                    guard.PYTHON, guard.PREFIX, guard.VARIANT)
        guard.COLLECT_CPU_PERCENT = '800'
        guard.UNIT_PREFIX = 'realman-rerecord-20261014-v6-hybrid'
        guard.PYTHON = '/opt/task/python'
        guard.PREFIX = Path('/task/deploy/realman_gen4')
        guard.VARIANT = 'rgb50_motion_quality_20261014_v6_161'

    def tearDown(self):
        (guard.COLLECT_CPU_PERCENT, guard.UNIT_PREFIX,
         guard.PYTHON, guard.PREFIX, guard.VARIANT) = self.old

    def test_stage_limits_are_exact(self):
        collect = guard.limits_for('collect')
        self.assertEqual(collect['CPUQuota'], '800%')
        self.assertEqual(collect['MemoryHigh'], str(10 * guard.GIB))
        self.assertEqual(collect['MemoryMax'], str(12 * guard.GIB))
        self.assertEqual(collect['TasksMax'], '128')
        for name, high, maximum in (('export', 3, 4), ('quality', 2, 3), ('render', 2, 3)):
            limits = guard.limits_for(name)
            self.assertEqual((limits['MemoryHigh'], limits['MemoryMax'], limits['CPUQuota']),
                             (str(high * guard.GIB), str(maximum * guard.GIB), '200%'))

    def test_unit_is_unique_scoped_and_system_service(self):
        with mock.patch.object(guard.uuid, 'uuid4', side_effect=[
            types.SimpleNamespace(hex='a' * 32), types.SimpleNamespace(hex='b' * 32)]):
            first, second = guard._new_unit('collect'), guard._new_unit('collect')
        self.assertNotEqual(first, second)
        self.assertEqual(first, 'realman-rerecord-20261014-v6-hybrid-collect-' + 'a' * 32 + '.service')

    def test_system_launch_is_argv_and_runs_as_owner(self):
        limits = guard.limits_for('collect')
        argv = guard._launch('realman-rerecord-20261014-v6-hybrid-collect-12.service',
                             Path('/task'), guard.WORKER, Path('/task/spec.json'), limits, {})
        self.assertEqual(argv[:5], ['sudo', '-n', 'systemd-run', '--system', '--wait'])
        self.assertIn('--pipe', argv)
        self.assertIn('--property=User=liminghe', argv)
        self.assertIn('--property=Group=liminghe', argv)
        self.assertIn('--property=MemoryMax=' + str(12 * guard.GIB), argv)
        self.assertIn('--property=CPUQuota=800%', argv)
        self.assertIn('--property=TasksMax=128', argv)
        self.assertIn('--property=OOMPolicy=stop', argv)
        probe_argv = guard._launch('realman-rerecord-20261014-v6-hybrid-probe-negative-13.service',
                                    Path('/task'), guard.PROBE, Path('/task/spec.json'), limits, {},
                                    allow_expected_oom=True)
        self.assertIn('--property=OOMPolicy=continue', probe_argv)
        self.assertNotIn('--user', argv)
        self.assertNotIn('shell', argv)

    def test_gate_releases_only_after_kernel_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            unit = 'realman-rerecord-20261014-v6-hybrid-collect-12.service'
            guard.atomic_json(directory / 'gate_ready.json', {
                'unit': unit, 'pid': 41, 'source_sha256': 'source',
                'adapter_sha256': 'adapter'})
            order = []
            binding = {'mode': 'v1', 'memory': directory / unit,
                       'cpu': directory / unit, 'pids': directory / unit}
            proc = types.SimpleNamespace(poll=lambda: None)

            def install(*_args):
                self.assertFalse((directory / 'gate_release.json').exists())
                order.append('limits_verified')

            with mock.patch.object(guard, 'sha', return_value='adapter'), \
                 mock.patch.object(guard, '_binding_for_pid', return_value=binding), \
                 mock.patch.object(guard, '_attest_unit', return_value={'unit': unit}), \
                 mock.patch.object(guard, '_install_limits', side_effect=install):
                actual, owner, released_at = guard._wait_gate(proc, directory, unit, {},
                                                                expected_source='source', proof_box={})
            self.assertIs(actual, binding)
            self.assertEqual(owner, {'unit': unit})
            self.assertGreater(released_at, 0.)
            self.assertEqual(order, ['limits_verified'])
            self.assertTrue((directory / 'gate_release.json').exists())

    def test_gate_never_releases_when_kernel_install_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            unit = 'realman-rerecord-20261014-v6-hybrid-collect-12.service'
            guard.atomic_json(directory / 'gate_ready.json', {
                'unit': unit, 'pid': 41, 'source_sha256': 'source',
                'adapter_sha256': 'adapter'})
            proc = types.SimpleNamespace(poll=lambda: None)
            with mock.patch.object(guard, 'sha', return_value='adapter'), \
                 mock.patch.object(guard, '_binding_for_pid', return_value={}), \
                 mock.patch.object(guard, '_attest_unit', return_value={'unit': unit}), \
                 mock.patch.object(guard, '_install_limits', side_effect=RuntimeError('write failed')):
                with self.assertRaisesRegex(RuntimeError, 'write failed'):
                    guard._wait_gate(proc, directory, unit, {}, expected_source='source', proof_box={})
            self.assertFalse((directory / 'gate_release.json').exists())

    def test_generated_unit_must_be_absent_before_launch(self):
        unit = 'realman-rerecord-20261014-v6-hybrid-collect-' + 'a' * 32 + '.service'
        with mock.patch.object(guard, 'unit_info', return_value={
            'LoadState': 'loaded', 'ActiveState': 'active', 'MainPID': '99',
            'ControlGroup': '/system.slice/' + unit, 'InvocationID': 'b' * 32}):
            with self.assertRaisesRegex(RuntimeError, 'already occupied'):
                guard._assert_unit_absent(unit)

    def test_stop_refuses_missing_or_changed_owner_without_sudo(self):
        unit = 'realman-rerecord-20261014-v6-hybrid-collect-' + 'a' * 32 + '.service'
        owner = {'unit': unit, 'pid': 99, 'invocation_id': 'a' * 32}
        with mock.patch.object(guard, '_sudo_systemctl') as sudo:
            with self.assertRaisesRegex(RuntimeError, 'outside attested'):
                guard.stop_exact_unit(unit, None)
            with mock.patch.object(guard, 'unit_info', return_value={
                    'MainPID': '99', 'ActiveState': 'active'}), \
                 mock.patch.object(guard, '_binding_for_pid', return_value={}), \
                 mock.patch.object(guard, '_attest_unit', return_value={
                    'unit': unit, 'pid': 99, 'invocation_id': 'b' * 32}):
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    guard.stop_exact_unit(unit, owner)
            sudo.assert_not_called()

    def test_launcher_failure_does_not_stop_unattested_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            script = directory / 'worker.py'
            script.write_text('pass\n')
            proc = types.SimpleNamespace(poll=lambda: 1, wait=lambda timeout: 1)
            unit = 'realman-rerecord-20261014-v6-hybrid-collect-' + 'a' * 32 + '.service'
            with mock.patch.object(guard, '_assert_unit_absent'), \
                 mock.patch.object(guard, '_launch', return_value=['launch']), \
                 mock.patch.object(guard.subprocess, 'Popen', return_value=proc), \
                 mock.patch.object(guard, '_wait_gate', side_effect=RuntimeError('launch failed')), \
                 mock.patch.object(guard, '_attest_unready_unit',
                                   side_effect=RuntimeError('wrong unit invocation')), \
                 mock.patch.object(guard, 'unit_info', return_value={'LoadState': 'not-found'}), \
                 mock.patch.object(guard, 'stop_exact_unit') as stop:
                with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                    guard._run_bounded_unit(unit, directory, script, {},
                                             {'RuntimeMaxSec': '30'}, {}, directory / 'run')
                stop.assert_not_called()
            failure = guard.json.loads((directory / 'run' / 'unattributed_startup_failure.json').read_text())
            self.assertEqual(failure['action'], 'no_stop_without_ownership_proof')

    def test_unready_unit_is_stopped_only_after_read_only_identity_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            script = directory / 'worker.py'
            script.write_text('pass\n')
            proc = types.SimpleNamespace(poll=lambda: None, wait=lambda timeout: 1)
            unit = 'realman-rerecord-20261014-v6-hybrid-collect-' + 'a' * 32 + '.service'
            owner = {'unit': unit, 'pid': 99, 'invocation_id': 'b' * 32}
            with mock.patch.object(guard, '_assert_unit_absent'), \
                 mock.patch.object(guard, '_launch', return_value=['launch']), \
                 mock.patch.object(guard.subprocess, 'Popen', return_value=proc), \
                 mock.patch.object(guard, '_wait_gate', side_effect=RuntimeError('gate timeout')), \
                 mock.patch.object(guard, '_attest_unready_unit', return_value=owner), \
                 mock.patch.object(guard, 'stop_exact_unit') as stop:
                with self.assertRaisesRegex(RuntimeError, 'gate timeout'):
                    guard._run_bounded_unit(unit, directory, script, {},
                                             {'RuntimeMaxSec': '30'}, {}, directory / 'run')
                stop.assert_called_once_with(unit, owner)

    def test_sample_row_uses_post_sampling_time(self):
        class Clock:
            value = 10.

            def monotonic(self):
                return self.value

            def sleep(self, duration):
                self.value += duration

        class Process:
            polls = 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 0

            def wait(self, timeout):
                return 0

        clock = Clock()
        resource = {'memory.peak': 128, 'memory.current': 64, 'rss_sum_bytes': 32,
                    'memory.events': {'max': 0, 'oom': 0, 'oom_kill': 0, 'oom_group_kill': 0},
                    'swap_zero_observed': True}

        def slow_sample(_binding):
            clock.value += 3.
            return resource

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            script = directory / 'worker.py'
            script.write_text('pass\n')
            run = directory / 'run'
            unit = 'realman-rerecord-20261014-v6-hybrid-export-' + 'a' * 32 + '.service'
            with mock.patch.object(guard, '_assert_unit_absent'), \
                 mock.patch.object(guard, '_launch', return_value=['launch']), \
                 mock.patch.object(guard.subprocess, 'Popen', return_value=Process()), \
                 mock.patch.object(guard, '_wait_gate',
                                   return_value=({'mode': 'v1'}, {'unit': unit}, 10.)), \
                 mock.patch.object(guard, '_sample', side_effect=slow_sample), \
                 mock.patch.object(guard, 'memory_available', return_value=100 * guard.GIB), \
                 mock.patch.object(guard, 'output_filesystems', return_value=(directory,)), \
                 mock.patch.object(guard, 'disk_free', return_value={str(directory): 100 * guard.GIB}), \
                 mock.patch.object(guard, 'unit_info', return_value={'ActiveState': 'inactive'}), \
                 mock.patch.object(guard.time, 'monotonic', side_effect=clock.monotonic), \
                 mock.patch.object(guard.time, 'sleep', side_effect=clock.sleep):
                result = guard._run_bounded_unit(unit, directory, script, {'command': []},
                                                  {'RuntimeMaxSec': '30'}, {}, run)
            row = guard.json.loads((run / 'resources.jsonl').read_text().splitlines()[0])
            self.assertEqual(row['elapsed_seconds'], 3.)
            self.assertEqual(result['maximum_sample_gap_seconds'], 3.)
            self.assertAlmostEqual(result['unit_exit_elapsed_seconds'], 3.1)

    def test_parent_only_writes_own_memory_unit_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            unit = 'realman-rerecord-20261014-v6-hybrid-collect-12.service'
            own = directory / unit
            foreign = directory / 'foreign.service'
            own.mkdir()
            foreign.mkdir()
            candidate = foreign / 'memory.soft_limit_in_bytes'
            candidate.write_text('0\n')
            with self.assertRaisesRegex(RuntimeError, 'outside own'):
                guard._sudo_write_exact(candidate, '1', {'memory': own}, unit)

    def test_source_binding_requires_release_contract(self):
        source = {'entry_sha256': 'entry', 'guard_sha256': 'guard',
                  'cgroup_adapter_sha256': 'adapter',
                  'release_contract_sha256': 'release'}
        hashes = {'script': 'entry', str(guard.WORKER): 'guard',
                  str(guard.ADAPTER): 'adapter', str(guard.RELEASE): 'release'}
        with mock.patch.object(guard, 'sha', side_effect=lambda x: hashes[str(x)]):
            guard._verify_source(source, 'script', 'export')
            source['release_contract_sha256'] = 'changed'
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                guard._verify_source(source, 'script', 'export')

    def test_negative_probe_rejects_missing_kernel_oom_kill_counter(self):
        resource = {'memory.events': {'max': 1, 'memsw_max': 0,
                                      'oom_kill': None},
                    'memory.oom_control': {'under_oom': 0}}
        passed, reason = probe.negative_oom_proof(resource, -9)
        self.assertFalse(passed)
        self.assertIn('omitted oom_kill', reason)
        resource['memory.oom_control']['oom_kill'] = 1
        resource['memory.events']['oom_kill'] = 1
        passed, reason = probe.negative_oom_proof(resource, -9)
        self.assertTrue(passed)
        self.assertIsNone(reason)

    def test_negative_probe_accepts_memsw_only_counter_with_oom_kill(self):
        resource = {'memory.events': {'max': 0, 'memsw_max': 42,
                                      'oom_kill': 1},
                    'memory.oom_control': {'under_oom': 0, 'oom_kill': 1}}
        passed, reason = probe.negative_oom_proof(resource, -9)
        self.assertTrue(passed, reason)
        resource['memory.events']['memsw_max'] = 0
        passed, reason = probe.negative_oom_proof(resource, -9)
        self.assertFalse(passed)
        resource['memory.events']['memsw_max'] = 42
        passed, reason = probe.negative_oom_proof(resource, 0)
        self.assertFalse(passed)


if __name__ == '__main__':
    unittest.main()
