"""Whitelisted, serialized local rechecks; never controls real hardware."""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
import secrets
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path


SCENARIOS = (
    (0, "雪碧 · 左臂 · 直立"), (38, "雪碧 · 右臂 · 横放"),
    (30, "可乐 · 左臂 · 横放"), (31, "可乐 · 右臂 · 直立"),
    (17, "芬达 · 左臂 · 直立"), (7, "芬达 · 右臂 · 斜放"),
)
INDICES = tuple(index for index, _ in SCENARIOS)
SIM_PYTHON = "/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python"


class ManualRunError(Exception):
    def __init__(self, message, code=409):
        self.code = code
        super().__init__(message)


class ManualRuns:
    def __init__(self, task_root, *, process_factory=subprocess.Popen, health_probe=None):
        self.root = Path(task_root).resolve()
        self.config_path = self.root / "viewer" / "retest_config.json"
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.process_factory = process_factory
        self.health_probe = health_probe or self._probe_health
        self.active = None
        try:
            previous_manual = json.loads((self.root / "live" / "status.json").read_text()).get("manual_run")
        except (FileNotFoundError, json.JSONDecodeError):
            previous_manual = None
        self.last = (previous_manual if previous_manual and
                     previous_manual.get("phase") in {"complete", "stopped", "error"} else None)
        self.process = None
        self.batch_verified = False
        self.lock_file = None

    def _config(self):
        try:
            return json.loads(self.config_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"enabled": False, "batch_complete": False}

    def _batch_done(self):
        if self.batch_verified:
            return True
        for index in INDICES:
            file = self.root / "episodes" / f"ep{index:02d}_model80000" / "report.json"
            try:
                report = json.loads(file.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return False
            if (report.get("checkpoint_step") != 80000
                    or report.get("model_evaluated") is not True
                    or report.get("model_query_count", 0) <= 0
                    or report.get("interface_or_infrastructure_error") is not None):
                return False
        self.batch_verified = True
        return True

    def snapshot(self):
        with self.lock:
            cfg = self._config()
            enabled = (cfg.get("enabled") is True and cfg.get("batch_complete") is True
                       and self._batch_done())
            active = dict(self.active) if self.active else None
            return {
                "enabled": enabled, "available": enabled and active is None,
                "active": active, "last": dict(self.last) if self.last else None,
                "scenarios": [{"index": i, "label": label} for i, label in SCENARIOS],
                "reason": ("本次复测正在运行，可查看实时相机或停止本次复测。" if active else
                           "可选择一个原场景重新运行本机模型。" if enabled else
                           "首轮 6 个场景仍在检查，完成后可在这里复测。"),
                "hardware_io": False, "checkpoint": 80000,
            }

    def _active_runners(self):
        script = str(self.root / "scripts" / "run_model.py").encode()
        found = []
        for directory in Path("/proc").iterdir():
            if not directory.name.isdigit():
                continue
            try:
                args = (directory / "cmdline").read_bytes().split(b"\0")
            except (OSError, PermissionError):
                continue
            if script in args:
                found.append(int(directory.name))
        return found

    def _probe_health(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:8769/healthz", timeout=5) as response:
            return json.loads(response.read())

    def _patch_status(self, changes):
        target = self.root / "live" / "status.json"
        try:
            status = json.loads(target.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            status = {}
        status.update(changes)
        status["status_revision"] = int(status.get("status_revision", 0)) + 1
        status["status_wall_time"] = time.time()
        target.parent.mkdir(exist_ok=True)
        tmp = target.parent / ".manual_status.tmp"
        tmp.write_text(json.dumps(status, ensure_ascii=False, allow_nan=False))
        os.replace(tmp, target)

    def start(self, index):
        if type(index) is not int or index not in INDICES:
            raise ManualRunError("只能选择本轮六个原始示范场景。", 400)
        with self.lock:
            if not self.snapshot()["available"]:
                raise ManualRunError("当前测试尚未结束或已有复测在运行。")
            if self._active_runners():
                raise ManualRunError("另一个本轮仿真仍在运行，请稍后再试。")
            cfg = self._config()
            identity = (self.root / "policy_runtime" / "service_v1" / "model_identity.json").resolve()
            if str(identity) != cfg.get("identity_path"):
                raise ManualRunError("本机模型身份配置不匹配。", 503)
            expected = cfg.get("identity_sha256")
            try:
                raw_identity = identity.read_bytes()
                identity_json = json.loads(raw_identity)
                if hashlib.sha256(raw_identity).hexdigest() != expected:
                    raise ValueError("Identity SHA differs")
                if identity_json.get("checkpoint_step") != 80000:
                    raise ValueError("Checkpoint differs")
                health = self.health_probe()
                if (health.get("ready") is not True or health.get("checkpoint_step") != 80000
                        or health.get("model_identity_sha256") != expected):
                    raise ValueError("Ready service identity differs")
            except Exception as exc:
                raise ManualRunError("80000 步本机模型未就绪，暂时无法启动复测。", 503) from exc
            lock_path = self.root / "live" / "runner.lock"
            lock_path.parent.mkdir(exist_ok=True)
            lock_file = lock_path.open("a+")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                lock_file.close()
                raise ManualRunError("另一个本轮仿真正在使用显卡。") from exc
            job_id = uuid.uuid4().hex
            tag = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
            output = self.root / "episodes" / f"manual_{tag}_ep{index:02d}_{job_id[:8]}"
            self.active = {"job_id": job_id, "index": index,
                           "label": dict(SCENARIOS)[index], "output": str(output),
                           "phase": "starting", "started_wall_time": time.time(),
                           "stop_requested": False, "excluded_from_formal_six": True}
            self.lock_file = lock_file
            self._patch_status({"manual_run": dict(self.active), "phase": "initializing_scene",
                                "episode_index": index, "episode_label": f"{dict(SCENARIOS)[index]} / 示范 {index}",
                                "episode_position": INDICES.index(index) + 1, "episode_total": 6,
                                "model_query_count": 0, "physics_step": 0, "sim_time_s": 0.0,
                                "native_success": False, "camera_paths": {},
                                "frame_is_current_episode": False, "frame_wall_time": None,
                                "frame_episode_index": None, "frame_physics_step": None,
                                "frame_sim_time_s": None})
            command = [SIM_PYTHON, str(self.root / "scripts" / "run_model.py"),
                       "--index", str(index), "--base-url", "http://127.0.0.1:8769",
                       "--identity", str(identity), "--identity-sha256", expected,
                       "--episode-position", str(INDICES.index(index) + 1),
                       "--output", str(output), "--live-root", str(self.root / "live"),
                       "--pace", "0.03"]
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": "0", "ROBOTWIN_RT_DENOISER": "optix",
                        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
                        "PATH": "/home/admin123/liminghe/miniconda3/envs/robotwin/bin:"
                                "/home/admin123/liminghe/miniconda3/envs/robotwin-official-data/bin:"
                                "/usr/local/bin:/usr/bin:/bin"})
            threading.Thread(target=self._worker, args=(command, env, lock_file), daemon=True).start()
            return dict(self.active)

    def _worker(self, command, env, lock_file):
        returncode = None
        error = None
        with self.lock:
            job = dict(self.active)
        log_path = Path(job["output"] + ".log")
        try:
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open("xb") as log:
                with self.lock:
                    if self.active["stop_requested"]:
                        returncode = -signal.SIGINT
                    else:
                        self.process = self.process_factory(
                            command, env=env, cwd=str(self.root), stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(lock_file.fileno(),))
                        self.active.update({"pid": self.process.pid, "phase": "running"})
                if self.process is not None:
                    returncode = self.process.wait()
        except Exception as exc:
            error = repr(exc)
        finally:
            with self.lock:
                final = dict(self.active)
                stopped = final.get("stop_requested", False)
                final.update({"returncode": returncode, "error": error,
                              "phase": "stopped" if stopped else "complete" if returncode == 0 else "error",
                              "ended_wall_time": time.time(), "log": str(log_path)})
                report_path = Path(final["output"]) / "report.json"
                if report_path.is_file():
                    try:
                        report = json.loads(report_path.read_text())
                        final["model_query_count"] = report.get("model_query_count", 0)
                        final["native_success"] = report.get("native_task_success", False)
                    except (OSError, json.JSONDecodeError):
                        pass
                self.last = final
                self.active = None
                self.process = None
                receipt_path = Path(final["output"] + "_manual_receipt.json")
                receipt_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
                self._patch_status({"manual_run": final,
                                    "phase": "stopped" if stopped else "episode_complete" if returncode == 0 else "episode_error"})
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                self.lock_file = None

    def stop(self, job_id):
        with self.lock:
            if self.active is None or self.active.get("job_id") != job_id:
                raise ManualRunError("没有可停止的本次复测。")
            self.active["stop_requested"] = True
            self.active["phase"] = "stopping"
            process = self.process
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                threading.Thread(target=self._ensure_stopped, args=(process,), daemon=True).start()
            return dict(self.active)

    @staticmethod
    def _ensure_stopped(process):
        for grace, sig in ((6, signal.SIGTERM), (3, signal.SIGKILL)):
            try:
                process.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        return

