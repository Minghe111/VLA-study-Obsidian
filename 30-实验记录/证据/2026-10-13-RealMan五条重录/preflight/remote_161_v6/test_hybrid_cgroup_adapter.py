"""CPU-only synthetic cgroup tests; never creates host cgroups."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import unittest

import hybrid_cgroup_adapter as adapter


UNIT = 'realman-rerecord-protected.service'


def put(directory, name, value):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(str(value) + '\n')


class HybridCgroupTest(TestCase):
    def _v1(self, root, *, dedicated=True):
        members = {}
        mounts = []
        for index, controller in enumerate(('memory', 'cpu', 'pids'), 1):
            mount = root / controller
            name = UNIT if dedicated else 'user@1002.service'
            relative = '/system.slice/' + name
            members[controller] = mount / relative.lstrip('/')
            members[controller].mkdir(parents=True)
            mounts.append(f'{index} 1 0:1 / {mount} rw,relatime - cgroup cgroup rw,{controller}')
        proc = '\n'.join(f'{index}:{controller}:/system.slice/' +
                         (UNIT if dedicated else 'user@1002.service')
                         for index, controller in enumerate(('memory', 'cpu', 'pids'), 1))
        memory, cpu, pids = (members[key] for key in ('memory', 'cpu', 'pids'))
        for name, value in {
            'memory.soft_limit_in_bytes': 10 * 1024**3,
            'memory.limit_in_bytes': 12 * 1024**3,
            'memory.memsw.limit_in_bytes': 12 * 1024**3,
            'memory.swappiness': 0,
            'memory.usage_in_bytes': 128 * 1024**2,
            'memory.max_usage_in_bytes': 256 * 1024**2,
            'memory.memsw.usage_in_bytes': 128 * 1024**2,
            'memory.memsw.max_usage_in_bytes': 256 * 1024**2,
            'memory.stat': 'swap 0\ntotal_swap 0\nrss 1048576',
            'memory.failcnt': 0, 'memory.memsw.failcnt': 0,
            'memory.oom_control': 'oom_kill_disable 0\nunder_oom 0\noom_kill 0',
        }.items():
            put(memory, name, value)
        for name, value in {'cpu.cfs_quota_us': 800000,
                            'cpu.cfs_period_us': 100000,
                            'cpu.stat': 'nr_periods 1\nnr_throttled 0\nthrottled_time 0'}.items():
            put(cpu, name, value)
        put(pids, 'pids.max', 128)
        put(pids, 'pids.current', 2)
        return proc, '\n'.join(mounts), members

    def test_v1_dedicated_unit_verified_and_clean(self):
        with TemporaryDirectory() as tmp, mock.patch.object(adapter.os, 'getpriority', return_value=10):
            proc, mounts, _dirs = self._v1(Path(tmp))
            binding = adapter.discover(proc, mounts, UNIT)
            proof = adapter.verify_limits(binding, high=10 * 1024**3,
                                          maximum=12 * 1024**3,
                                          cpu_percent=800, pids=128)
            self.assertEqual(proof['mode'], 'v1')
            resource = adapter.sample(binding)
            self.assertTrue(adapter.require_clean_terminal(resource))
            recorded = {**proof, 'controller_paths': {
                'memory': f'/sys/fs/cgroup/memory/system.slice/{UNIT}',
                'cpu': f'/sys/fs/cgroup/cpu,cpuacct/system.slice/{UNIT}',
                'pids': f'/sys/fs/cgroup/pids/system.slice/{UNIT}',
            }}
            self.assertTrue(adapter.verify_recorded(recorded, resource, unit=UNIT,
                high=10 * 1024**3, maximum=12 * 1024**3,
                cpu_percent=800, pids=128))
            wrong = {**recorded, 'controller_paths':
                     {**recorded['controller_paths'], 'cpu': '/sys/fs/cgroup/cpu/user.slice'}}
            with self.assertRaisesRegex(adapter.CgroupProofError, 'not dedicated'):
                adapter.verify_recorded(wrong, resource, unit=UNIT,
                    high=10 * 1024**3, maximum=12 * 1024**3,
                    cpu_percent=800, pids=128)

    def test_shared_v1_ancestor_rejected_even_with_plausible_values(self):
        with TemporaryDirectory() as tmp:
            proc, mounts, _dirs = self._v1(Path(tmp), dedicated=False)
            with self.assertRaisesRegex(adapter.CgroupProofError, 'not in exact dedicated'):
                adapter.discover(proc, mounts, UNIT)

    def test_unified_membership_does_not_excuse_shared_v1_controllers(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc, mounts, _dirs = self._v1(root, dedicated=False)
            unified = root / 'unified'
            target = unified / 'system.slice' / UNIT
            for name, value in {'memory.high': 10 * 1024**3,
                                'memory.max': 12 * 1024**3,
                                'memory.swap.max': 0,
                                'cpu.max': '800000 100000',
                                'pids.max': 128}.items():
                put(target, name, value)
            mounts += f'\n9 1 0:9 / {unified} rw,relatime - cgroup2 cgroup rw'
            proc += f'\n0::/system.slice/{UNIT}'
            with self.assertRaisesRegex(adapter.CgroupProofError, 'not in exact dedicated'):
                adapter.discover(proc, mounts, UNIT)

    def test_missing_swap_cap_is_fail_closed(self):
        with TemporaryDirectory() as tmp, mock.patch.object(adapter.os, 'getpriority', return_value=10):
            proc, mounts, dirs = self._v1(Path(tmp))
            (dirs['memory'] / 'memory.memsw.limit_in_bytes').unlink()
            with self.assertRaisesRegex(adapter.CgroupProofError, 'Missing kernel'):
                adapter.verify_limits(adapter.discover(proc, mounts, UNIT),
                                      high=10 * 1024**3, maximum=12 * 1024**3,
                                      cpu_percent=800, pids=128)

    def test_mismatched_soft_cpu_or_pid_limit_rejected(self):
        with TemporaryDirectory() as tmp, mock.patch.object(adapter.os, 'getpriority', return_value=10):
            proc, mounts, dirs = self._v1(Path(tmp))
            binding = adapter.discover(proc, mounts, UNIT)
            targets = (('memory', 'memory.soft_limit_in_bytes', 0),
                       ('cpu', 'cpu.cfs_quota_us', 200000), ('pids', 'pids.max', 999))
            for controller, name, value in targets:
                original = (dirs[controller] / name).read_text()
                put(dirs[controller], name, value)
                with self.assertRaises(adapter.CgroupProofError):
                    adapter.verify_limits(binding, high=10 * 1024**3,
                                          maximum=12 * 1024**3,
                                          cpu_percent=800, pids=128)
                (dirs[controller] / name).write_text(original)

    def test_v1_any_hard_or_memsw_event_rejected(self):
        with TemporaryDirectory() as tmp:
            proc, mounts, dirs = self._v1(Path(tmp))
            binding = adapter.discover(proc, mounts, UNIT)
            for name in ('memory.failcnt', 'memory.memsw.failcnt'):
                put(dirs['memory'], name, 1)
                with self.assertRaises(adapter.CgroupProofError):
                    adapter.require_clean_terminal(adapter.sample(binding))
                put(dirs['memory'], name, 0)

    def test_fuzzy_memsw_minus_usage_is_diagnostic_only(self):
        with TemporaryDirectory() as tmp:
            proc, mounts, dirs = self._v1(Path(tmp))
            memory = dirs['memory']
            used = int((memory / 'memory.usage_in_bytes').read_text())
            put(memory, 'memory.memsw.usage_in_bytes', used + 131072)
            resource = adapter.sample(adapter.discover(proc, mounts, UNIT))
            self.assertEqual(resource['memory.usage_in_bytes.raw'], used)
            self.assertEqual(resource['memory.memsw.usage_in_bytes.raw'], used + 131072)
            self.assertEqual(resource['memory.memsw_minus_memory.diagnostic_bytes'], 131072)
            self.assertEqual(resource['memory.swap.current'], 0)
            self.assertTrue(adapter.require_clean_terminal(resource))

    def test_v1_local_or_hierarchical_swap_is_never_excused(self):
        with TemporaryDirectory() as tmp:
            proc, mounts, dirs = self._v1(Path(tmp))
            binding = adapter.discover(proc, mounts, UNIT)
            for raw in ('swap 4096\ntotal_swap 4096',
                        'swap 0\ntotal_swap 4096'):
                put(dirs['memory'], 'memory.stat', raw)
                resource = adapter.sample(binding)
                self.assertEqual(resource['memory.swap.current'], 4096)
                with self.assertRaisesRegex(adapter.CgroupProofError, 'swap'):
                    adapter.require_clean_terminal(resource)

    def test_v1_missing_malformed_or_inconsistent_stat_rejected(self):
        with TemporaryDirectory() as tmp:
            proc, mounts, dirs = self._v1(Path(tmp))
            binding = adapter.discover(proc, mounts, UNIT)
            for raw in ('swap 0', 'total_swap 0', 'swap no\ntotal_swap 0',
                        'swap 4096\ntotal_swap 0'):
                put(dirs['memory'], 'memory.stat', raw)
                with self.assertRaises(adapter.CgroupProofError):
                    adapter.sample(binding)

    def test_v1_stat_bracket_detects_swap_on_second_read(self):
        with TemporaryDirectory() as tmp:
            proc, mounts, dirs = self._v1(Path(tmp))
            binding = adapter.discover(proc, mounts, UNIT)
            memory = dirs['memory']
            original_integer = adapter._integer
            changed = []

            def change_after_first_stat(directory, name):
                if directory == memory and name == 'memory.usage_in_bytes' and not changed:
                    put(memory, 'memory.stat', 'swap 0\ntotal_swap 4096')
                    changed.append(True)
                return original_integer(directory, name)

            with mock.patch.object(adapter, '_integer', side_effect=change_after_first_stat):
                resource = adapter.sample(binding)
            self.assertEqual(resource['memory.stat.total_swap_before_bytes'], 0)
            self.assertEqual(resource['memory.stat.total_swap_after_bytes'], 4096)
            self.assertEqual(resource['memory.swap.current'], 4096)
            with self.assertRaisesRegex(adapter.CgroupProofError, 'swap'):
                adapter.require_clean_terminal(resource)

    def test_missing_oom_count_is_unknown_and_rejected(self):
        with TemporaryDirectory() as tmp:
            proc, mounts, dirs = self._v1(Path(tmp))
            put(dirs['memory'], 'memory.oom_control', 'oom_kill_disable 0\nunder_oom 0')
            resource = adapter.sample(adapter.discover(proc, mounts, UNIT))
            self.assertIsNone(resource['memory.events']['oom_kill'])
            with self.assertRaisesRegex(adapter.CgroupProofError, 'hard cap or OOM'):
                adapter.require_clean_terminal(resource)

    def test_v2_dedicated_limits(self):
        with TemporaryDirectory() as tmp, mock.patch.object(adapter.os, 'getpriority', return_value=10):
            mount = Path(tmp) / 'unified'
            unit = mount / 'system.slice' / UNIT
            for name, value in {'memory.high': 10 * 1024**3,
                                'memory.max': 12 * 1024**3,
                                'memory.swap.max': 0,
                                'cpu.max': '800000 100000',
                                'pids.max': 128,
                                'memory.current': 1024,
                                'memory.peak': 2048,
                                'memory.swap.current': 0,
                                'pids.current': 2,
                                'memory.events': 'max 0\noom 0\noom_kill 0\noom_group_kill 0',
                                'cpu.stat': 'usage_usec 1'}.items():
                put(unit, name, value)
            proc = f'0::/system.slice/{UNIT}'
            mounts = f'9 1 0:9 / {mount} rw,relatime - cgroup2 cgroup rw'
            binding = adapter.discover(proc, mounts, UNIT)
            self.assertEqual(binding['mode'], 'v2')
            self.assertTrue(adapter.verify_limits(binding, high=10 * 1024**3,
                            maximum=12 * 1024**3, cpu_percent=800, pids=128))
            self.assertTrue(adapter.require_clean_terminal(adapter.sample(binding)))


if __name__ == '__main__':
    unittest.main()
