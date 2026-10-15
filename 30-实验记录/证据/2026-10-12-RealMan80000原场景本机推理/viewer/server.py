"""Loopback-only MJPEG + SSE viewer for actual RealMan inference captures."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from manual_runs import ManualRuns, ManualRunError

HERE = Path(__file__).resolve().parent


def read_status(root):
    try:
        data = json.loads((root / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"phase": "preparing", "frame_seq": 0, "camera_paths": {},
                "model_query_count": 0, "physics_step": 0, "sim_time_s": 0,
                "status_revision": 0, "checkpoint": 80000}
    data["viewer_wall_time"] = time.time()
    data["frame_age_s"] = (max(0., time.time() - data["frame_wall_time"])
                           if data.get("frame_wall_time") else None)
    return data


def handler_for(root, manual=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            # Stream disconnects are normal when changing visible cameras.
            if str(args[1] if len(args) > 1 else "") not in ("200",):
                super().log_message(fmt, *args)

        def send_content_headers(self, content_type, *, length=None):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("X-Content-Type-Options", "nosniff")
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()

        def respond(self, content, content_type):
            self.send_content_headers(content_type, length=len(content))
            self.wfile.write(content)

        def state(self):
            value = read_status(root)
            if manual is not None:
                value["manual_control"] = manual.snapshot()
            return value

        def json_reply(self, code, value):
            content = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    self.respond((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/status":
                    self.respond(json.dumps(self.state(), ensure_ascii=False).encode(),
                                 "application/json; charset=utf-8")
                elif url.path == "/api/session" and manual is not None:
                    self.json_reply(200, {"request_token": manual.token,
                                          "manual_control": manual.snapshot()})
                elif url.path == "/events":
                    self.events()
                elif url.path.startswith("/mjpeg/"):
                    camera = url.path.rsplit("/", 1)[1]
                    if camera not in ("head", "left_wrist", "right_wrist", "external"):
                        self.send_error(404)
                    else:
                        self.stream(camera)
                elif url.path == "/frame":
                    query = parse_qs(url.query)
                    self.single_frame(query.get("camera", ["head"])[0],
                                      query.get("seq", [None])[0])
                elif url.path == "/recorded":
                    self.recorded_video(parse_qs(url.query).get("path", [""])[0])
                elif url.path == "/health":
                    self.respond(b'{"ok":true,"mode":"actual-capture-mjpeg","loopback_only":true}',
                                 "application/json")
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True

        def do_POST(self):
            if self.path not in ("/api/run", "/api/stop") or manual is None:
                self.json_reply(404, {"error": "该操作不可用。"})
                return
            origin = self.headers.get("Origin")
            allowed_origins = {f"http://127.0.0.1:{self.server.server_address[1]}",
                               f"http://localhost:{self.server.server_address[1]}"}
            if (origin not in allowed_origins
                    or self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json"
                    or not secrets.compare_digest(self.headers.get("X-Run-Token", ""), manual.token)):
                self.json_reply(403, {"error": "请从本机页面操作。"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2048:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected object")
                if self.path == "/api/run":
                    if set(payload) != {"index"}:
                        raise ValueError("Only fixed index accepted")
                    result = manual.start(payload["index"])
                else:
                    if set(payload) != {"job_id"} or not isinstance(payload["job_id"], str):
                        raise ValueError("Expected owned job")
                    result = manual.stop(payload["job_id"])
                self.json_reply(202, {"accepted": True, "job": result})
            except ManualRunError as exc:
                self.json_reply(exc.code, {"error": str(exc)})
            except (ValueError, json.JSONDecodeError):
                self.json_reply(400, {"error": "请求格式不正确。"})

        def stream(self, camera):
            self.send_content_headers("multipart/x-mixed-replace; boundary=actual_rgb_frame")
            last_seq = -1
            while True:
                state = self.state()
                seq = state.get("frame_seq", 0)
                rel = state.get("camera_paths", {}).get(camera)
                if seq != last_seq and rel:
                    file = (root / rel).resolve()
                    if not file.is_relative_to(root.resolve()):
                        raise ValueError("Frame path must stay in live directory")
                    try:
                        jpeg = file.read_bytes()
                    except FileNotFoundError:
                        time.sleep(.02)
                        continue
                    self.wfile.write(
                        b"--actual_rgb_frame\r\nContent-Type: image/jpeg\r\n" +
                        f"Content-Length: {len(jpeg)}\r\nX-Frame-Seq: {seq}\r\n"
                        f"X-Physics-Step: {state.get('frame_physics_step', 0)}\r\n"
                        f"X-Sim-Time: {state.get('frame_sim_time_s', 0)}\r\n\r\n".encode() +
                        jpeg + b"\r\n")
                    self.wfile.flush()
                    last_seq = seq
                time.sleep(.03)

        def events(self):
            self.send_content_headers("text/event-stream; charset=utf-8")
            last_revision = None
            heartbeat = 0.
            while True:
                state = self.state()
                revision = state.get("status_revision", 0)
                now = time.monotonic()
                if revision != last_revision or now - heartbeat > 2.:
                    blob = json.dumps(state, ensure_ascii=False).encode()
                    self.wfile.write(b"event: status\ndata: " + blob + b"\n\n")
                    self.wfile.flush()
                    last_revision = revision
                    heartbeat = now
                time.sleep(.15)

        def single_frame(self, camera, sequence=None):
            state = read_status(root)
            if camera not in ("head", "left_wrist", "right_wrist", "external"):
                self.send_error(404)
                return
            if sequence is not None:
                try:
                    sequence = int(sequence)
                    if sequence <= 0:
                        raise ValueError()
                except ValueError:
                    self.send_error(400)
                    return
                rel = f"frames/{sequence:09d}/{camera}.jpg"
            else:
                rel = state.get("camera_paths", {}).get(camera)
            if not rel:
                self.send_error(404, "No actual capture available yet")
                return
            file = (root / rel).resolve()
            if not file.is_relative_to(root.resolve()):
                self.send_error(403)
                return
            if not file.is_file():
                self.send_error(404, "Requested capture is no longer retained")
                return
            self.respond(file.read_bytes(), "image/jpeg")

        def recorded_video(self, requested):
            task_root = root.resolve().parent
            path = Path(requested)
            file = (path if path.is_absolute() else task_root / path).resolve()
            if (not file.is_relative_to(task_root) or file.suffix.lower() != ".mp4"
                    or "source" in file.relative_to(task_root).parts):
                self.send_error(403, "Only this evaluation's recorded MP4 files are available")
                return
            if not file.is_file():
                self.send_error(404)
                return
            size = file.stat().st_size
            start, end = 0, size - 1
            requested_range = self.headers.get("Range", "")
            if requested_range:
                try:
                    unit, raw_range = requested_range.split("=", 1)
                    first, last = raw_range.split("-", 1)
                    if unit != "bytes" or "," in raw_range:
                        raise ValueError()
                    if first:
                        start = int(first)
                        end = int(last) if last else end
                    else:
                        start = max(0, size - int(last))
                    if start < 0 or start > end or end >= size:
                        raise ValueError()
                except ValueError:
                    self.send_error(416)
                    return
            self.send_response(206 if requested_range else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            if requested_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with file.open("rb") as source:
                source.seek(start)
                left = end - start + 1
                while left:
                    data = source.read(min(left, 256 * 1024))
                    if not data:
                        break
                    self.wfile.write(data)
                    left -= len(data)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE.parent / "live")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    manual = ManualRuns(args.root.resolve().parent)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(args.root, manual))
    server.daemon_threads = True
    print(f"LIVE_VIEWER http://127.0.0.1:{args.port}/ root={args.root}", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
