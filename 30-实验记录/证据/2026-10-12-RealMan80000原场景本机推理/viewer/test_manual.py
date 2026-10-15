"""Isolated fake-process tests; no model queries, GPU, or live-root mutations."""
import hashlib
import json
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import manual_runs
from manual_runs import INDICES, ManualRuns
from server import handler_for


class FakeProcess:
    pid = 123456789  # Never signalled: killpg is replaced by an in-process fixture.

    def __init__(self):
        self.event = threading.Event()

    def poll(self):
        return 130 if self.event.is_set() else None

    def wait(self, timeout=None):
        if not self.event.wait(timeout=timeout):
            raise subprocess.TimeoutExpired("component_fake_process", timeout)
        return 130


def main():
    with tempfile.TemporaryDirectory(prefix="realman-manual-component-") as directory:
        root = Path(directory)
        (root / "viewer").mkdir()
        (root / "live").mkdir()
        (root / "live" / "status.json").write_text(json.dumps({
            "frame_seq": 41, "model_query_count": 25, "physics_step": 6000, "sim_time_s": 24,
            "camera_paths": {"head": "old-frame.jpg"}, "frame_is_current_episode": True,
            "frame_wall_time": 1, "results_title": "formal six result fixture"}))
        identity = root / "policy_runtime" / "service_v1" / "model_identity.json"
        identity.parent.mkdir(parents=True)
        identity.write_text(json.dumps({"checkpoint_step": 80000, "component_fixture_only": True}))
        sha = hashlib.sha256(identity.read_bytes()).hexdigest()
        config = {"enabled": False, "batch_complete": False,
                  "identity_path": str(identity), "identity_sha256": sha}
        cfg_file = root / "viewer" / "retest_config.json"
        cfg_file.write_text(json.dumps(config))
        before = {}
        for index in INDICES:
            report = root / "episodes" / f"ep{index:02d}_model80000" / "report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"checkpoint_step": 80000, "model_evaluated": True,
                                          "model_query_count": 1,
                                          "interface_or_infrastructure_error": None,
                                          "component_fixture_only": True}))
            before[str(report)] = hashlib.sha256(report.read_bytes()).hexdigest()
        fake = FakeProcess()
        captured = {}

        def factory(command, **kwargs):
            captured.update({"command": command, "kwargs": kwargs})
            return fake

        manager = ManualRuns(root, process_factory=factory,
                             health_probe=lambda: {"ready": True, "checkpoint_step": 80000,
                                                   "model_identity_sha256": sha})
        manager._active_runners = lambda: []
        calls = []
        original_killpg = manual_runs.os.killpg
        manual_runs.os.killpg = lambda pid, sig: (calls.append((pid, sig)), fake.event.set())
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(root / "live", manager))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def post(path, payload, *, origin=None, token=None):
            request = urllib.request.Request(
                base + path, data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json", "Origin": origin or base,
                         "X-Run-Token": manager.token if token is None else token})
            try:
                with opener.open(request, timeout=5) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        try:
            with opener.open(base + "/api/session", timeout=5) as response:
                session = json.loads(response.read())
            assert session["request_token"] == manager.token
            assert post("/api/run", {"index": 0})[0] == 409
            assert post("/api/run", {"index": 0}, origin="http://other-site.invalid")[0] == 403
            assert post("/api/run", {"index": 0}, token="invalid")[0] == 403
            config.update({"enabled": True, "batch_complete": True})
            cfg_file.write_text(json.dumps(config))
            assert post("/api/run", {"index": 999})[0] == 400
            assert post("/api/run", {"index": True})[0] == 400
            assert post("/api/run", {"index": 0, "output": "/arbitrary"})[0] == 400
            code, accepted = post("/api/run", {"index": 0})
            assert code == 202
            initializing = json.loads((root / "live" / "status.json").read_text())
            assert initializing["frame_seq"] == 41
            assert initializing["frame_is_current_episode"] is False
            assert initializing["camera_paths"] == {}
            assert initializing["model_query_count"] == 0
            assert initializing["physics_step"] == 0
            assert initializing["sim_time_s"] == 0
            assert initializing["episode_index"] == 0
            assert initializing["results_title"] == "formal six result fixture"
            tick = time.monotonic()
            while not captured and time.monotonic() - tick < 5:
                time.sleep(.01)
            command = captured["command"]
            assert command[0] == manual_runs.SIM_PYTHON
            assert command[1] == str(root / "scripts" / "run_model.py")
            assert command[command.index("--index") + 1] == "0"
            assert "manual_" in command[command.index("--output") + 1]
            assert captured["kwargs"]["start_new_session"] is True
            assert "shell" not in captured["kwargs"]
            assert post("/api/run", {"index": 38})[0] == 409
            assert post("/api/stop", {"job_id": "unowned-job"})[0] == 409
            assert calls == []
            assert post("/api/stop", {"job_id": accepted["job"]["job_id"]})[0] == 202
            tick = time.monotonic()
            while manager.active is not None and time.monotonic() - tick < 5:
                time.sleep(.01)
            assert manager.active is None
            assert manager.last["phase"] == "stopped"
            assert manager.last["excluded_from_formal_six"] is True
            assert calls and all(pid == fake.pid for pid, _ in calls)
            assert manager.snapshot()["available"] is True
            assert all(hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
                       for file, digest in before.items())
        finally:
            manual_runs.os.killpg = original_killpg
            server.shutdown()
            server.server_close()
        print(json.dumps({"passed": True, "component_fixture_only": True,
                          "model_queries": 0, "actual_processes_spawned": 0,
                          "checks": ["first-six gate", "fixed six indices only", "origin and request token",
                                     "serialized process ownership", "stop only owned fake job",
                                     "formal six reports unchanged",
                                     "new-scene startup hides previous RGB and clears counters"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
