"""Component-only fixtures, isolated from the visible inference viewer."""
import hashlib
import json
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from publish_live import LivePublisher
from server import handler_for


def main():
    with tempfile.TemporaryDirectory(prefix="realman-viewer-component-") as directory:
        root = Path(directory) / "live"
        publisher = LivePublisher(root)
        publisher.begin_episode(0, "明确标记的组件测试，非推理结果")
        fixture = {name: np.full((48, 64, 3), value, np.uint8)
                   for name, value in (("head", 40), ("left_wrist", 100), ("right_wrist", 180))}
        first = publisher.update(fixture, {"phase": "component_test", "sim_time_s": .06,
                                           "physics_step": 15, "ui_test": True})
        publisher.update(metadata={"phase": "waiting_for_model", "ui_test": True})
        assert publisher.seq == first
        assert json.loads((root / "status.json").read_text())["frame_seq"] == first
        # A new per-episode process must continue the global sequence.
        publisher2 = LivePublisher(root)
        assert publisher2.seq == first
        publisher2.begin_episode(38, "组件测试第二个进程")
        assert json.loads((root / "status.json").read_text())["camera_paths"] == {}
        second = publisher2.update(fixture, {"phase": "component_test", "sim_time_s": .06,
                                            "physics_step": 15, "ui_test": True})
        assert second == first + 1
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(root))
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = f"http://127.0.0.1:{server.server_address[1]}"
        response = opener.open(base + "/mjpeg/head", timeout=5)

        def frame():
            assert response.readline().strip() == b"--actual_rgb_frame"
            headers = {}
            while True:
                line = response.readline().strip()
                if not line:
                    break
                name, value = line.decode().split(":", 1)
                headers[name] = value.strip()
            payload = response.read(int(headers["Content-Length"]))
            assert response.read(2) == b"\r\n"
            decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (48, 64, 3)
            return headers, hashlib.sha256(payload).hexdigest()

        h1, hash1 = frame()
        fixture["head"][:] = 220
        third = publisher2.update(fixture, {"phase": "component_test", "sim_time_s": .12,
                                           "physics_step": 30, "ui_test": True})
        h2, hash2 = frame()
        assert int(h1["X-Frame-Seq"]) == second
        assert int(h2["X-Frame-Seq"]) == third > second
        assert int(h2["X-Physics-Step"]) == 30
        assert float(h2["X-Sim-Time"]) == .12
        assert hash1 != hash2
        events = opener.open(base + "/events", timeout=5)
        assert events.readline().strip() == b"event: status"
        status = json.loads(events.readline().strip()[6:])
        assert status["frame_seq"] == third
        assert status["ui_test"] is True
        # Byte-range delivery uses an isolated transport fixture, not a video claim.
        video = Path(directory) / "clip.mp4"
        video.write_bytes(bytes(range(256)))
        requested = urllib.request.Request(
            base + "/recorded?path=" + urllib.parse.quote(str(video)),
            headers={"Range": "bytes=2-5"})
        with opener.open(requested, timeout=5) as transport:
            assert transport.status == 206
            assert transport.read() == bytes(range(2, 6))
            assert transport.headers["Content-Range"] == "bytes 2-5/256"
        try:
            opener.open(base + "/recorded?path=/etc/passwd.mp4", timeout=5)
        except urllib.error.HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("Recorded-video path escaped task root")
        response.close()
        events.close()
        server.shutdown()
        server.server_close()
        print(json.dumps({"passed": True, "component_fixture_only": True,
                          "actual_inference_not_claimed": True,
                          "checks": ["metadata updates never increment image sequence",
                                     "per-episode producer inherits monotonic frame sequence",
                                     "MJPEG delivered distinct images with advancing sequence and physical clock",
                                     "SSE delivered consistent latest status",
                                     "recorded-file byte ranges and task-directory boundary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
