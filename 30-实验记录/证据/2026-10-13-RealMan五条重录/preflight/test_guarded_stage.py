import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

P = Path(__file__).parent
spec = importlib.util.spec_from_file_location('guard', P / 'run_guarded_stage.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

class StageGuardTest(unittest.TestCase):
    def test_stage_limits(self):
        expected = {'collect': (10, 12, 2100), 'export': (3, 4, 1200),
                    'quality': (2, 3, 1200), 'render': (2, 3, 1200)}
        for stage, (high, maximum, timeout) in expected.items():
            r = g.limits_for(stage)
            self.assertEqual(int(r['MemoryHigh']), high * g.GIB)
            self.assertEqual(int(r['MemoryMax']), maximum * g.GIB)
            self.assertEqual(int(r['RuntimeMaxSec']), timeout)
            self.assertEqual(r['MemorySwapMax'], '0')
            self.assertEqual(r['CPUQuota'], '200%')
            self.assertEqual(r['OOMPolicy'], 'stop')
            self.assertEqual(r['KillMode'], 'control-group')
            self.assertEqual(r['TasksMax'], '128')

    def test_invalid_timeouts(self):
        for timeout in (0, 29, 2101, float('inf'), float('nan'), -1):
            with self.assertRaises(ValueError):
                g.limits_for('export', timeout)
        self.assertEqual(g.limits_for('export', 60)['RuntimeMaxSec'], '60')

    def test_environment(self):
        c = g.env_for('collect', Path('/tmp/runtime'))
        e = g.env_for('export', Path('/tmp/runtime'))
        self.assertEqual(c['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(e['CUDA_VISIBLE_DEVICES'], '')
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(e[name], '2')
        self.assertNotIn('TOKEN', e)
        self.assertNotIn('PASSWORD', e)

    def test_no_generic_command(self):
        for args in (['bash', '-c', 'true'], ['/usr/bin/python3', '-c', 'print(1)'],
                     [g.PYTHON, '/tmp/anything.py'], [g.PYTHON]):
            with self.assertRaises((ValueError, FileNotFoundError)):
                g.validate_command('export', args)

    def test_stage_whitelist_exact_and_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / 'streaming_export.py'
            script.write_text('pass\n')
            alias = root / 'alias.py'
            alias.symlink_to(script)
            with patch.object(g, 'TOOLS', {'export': (script,)}):
                _, current = g.validate_command('export', [g.PYTHON, str(script)])
                self.assertEqual(current, script)
                with self.assertRaises(ValueError):
                    g.validate_command('export', [g.PYTHON, str(alias)])
                with self.assertRaises(ValueError):
                    g.validate_command('export', [g.PYTHON, str(script), '\x00'])

    def test_only_v5_per_slot_collector(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(g, 'PREFIX', root):
                good = root / g.VARIANT / 'jobs/slot34_runtime/preflight/run_staged_collector.py'
                good.parent.mkdir(parents=True)
                good.write_text('pass\n')
                runtime, _ = g.validate_command('collect', [g.PYTHON, str(good)])
                self.assertEqual(runtime, good.parent.parent)
                for relative in ('jobs/slot1_runtime/preflight/run_staged_collector.py',
                                 'jobs/slot17_runtime/preflight/run_so3_acquisition_collector.py',
                                 'preflight/run_staged_collector.py'):
                    bad = root / g.VARIANT / relative
                    bad.parent.mkdir(parents=True, exist_ok=True)
                    bad.write_text('pass\n')
                    with self.assertRaises(ValueError):
                        g.validate_command('collect', [g.PYTHON, str(bad)])

    def test_process_detection(self):
        prefix = str(g.PREFIX / g.VARIANT)
        rows = [dict(pid=8001, uid=os.getuid(), args=[g.PYTHON, prefix + '/jobs/slot34_runtime/private/derived_worker.py'], cgroup='/task', rss_bytes=1),
                dict(pid=8002, uid=os.getuid(), args=[g.PYTHON, '/other/run_randomized_collector.py'], cgroup='/other', rss_bytes=2),
                dict(pid=8003, uid=os.getuid(), args=[g.PYTHON, prefix + '/preflight/run_randomized_collector.py'], cgroup='/task', rss_bytes=3),
                dict(pid=8004, uid=os.getuid(), args=['ffmpeg', '-i', prefix + '/tmp.mp4'], cgroup='/task', rss_bytes=4)]
        self.assertEqual([r['pid'] for r in g.known_task_processes(rows)], [8001, 8003, 8004])

    def test_cgroup_scope_rejects_wrong_unit(self):
        uid = os.getuid()
        good = f'/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{g.UNIT}'
        self.assertEqual(g.scoped_cgroup({'ControlGroup': good}), g.CGROOT / good.lstrip('/'))
        for bad in ('/system.slice/other.service', f'/user.slice/user-{uid + 1}.slice/user@{uid + 1}.service/app.slice/{g.UNIT}',
                    good + '-other', '/user.slice/x/' + g.UNIT):
            with self.assertRaises(RuntimeError):
                g.scoped_cgroup({'ControlGroup': bad})

    def test_samples_only_cgroup_members(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cg = root / 'my.service'
            cg.mkdir()
            for name, value in {'memory.current': 4096, 'memory.peak': 8192, 'memory.swap.current': 0,
                                'memory.swap.peak': 0, 'memory.swap.max': 0, 'pids.current': 2,
                                'pids.peak': 2}.items():
                (cg / name).write_text(str(value))
            for name in ('memory.events', 'memory.events.local', 'memory.swap.events'):
                (cg / name).write_text('oom 0\noom_kill 0\n')
            (cg / 'cpu.stat').write_text('usage_usec 44\nnr_throttled 1\n')
            with patch.object(g, 'CGROOT', root):
                sample = g.cgroup_sample(cg, rows=[{'pid': 1, 'cgroup': '/my.service', 'rss_bytes': 12},
                                          {'pid': 2, 'cgroup': '/my.service/sub', 'rss_bytes': 15},
                                          {'pid': 3, 'cgroup': '/other', 'rss_bytes': 999}])
            self.assertEqual(sample['rss_sum_bytes'], 27)
            self.assertEqual(sample['member_pids'], [1, 2])
            self.assertEqual(sample['memory.peak'], 8192)

    def test_atomic_receipt_and_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / 'record.json'
            g.atomic_json(file, {'test': '原数据保留'})
            g.atomic_json(file, {'test': '完成'})
            self.assertEqual(__import__('json').loads(file.read_text()), {'test': '完成'})
            self.assertEqual(len(list(Path(td).iterdir())), 1)
            lines = Path(td) / 'sample.jsonl'
            g.append_jsonl(lines, {'m': 1})
            g.append_jsonl(lines, {'m': 2})
            self.assertEqual(len(lines.read_text().splitlines()), 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
