#!/usr/bin/env python3
# Pi -> browser bridge: UDP (or recorded log) -> SSE -> operator_ui_web.html.
#   python3 puente.py --live --udp-port 5005   ->  http://localhost:8080
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HERE = os.path.dirname(os.path.abspath(__file__))
UI_FILE = os.path.join(HERE, "operator_ui_web.html")

_lock = threading.Lock()
_latest: dict | None = None
_subs: list[queue.Queue] = []
_stop = threading.Event()


def publish(pkt: dict) -> None:
    global _latest
    with _lock:
        _latest = pkt
        subs = list(_subs)
    for q in subs:
        try:
            if q.full():
                q.get_nowait()
            q.put_nowait(pkt)
        except (queue.Full, queue.Empty):
            pass


def source_udp(port: int, host: str = "0.0.0.0") -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.settimeout(0.5)
    last_seq = -1
    print(f"[bridge] listening for UDP on {host}:{port}", flush=True)
    while not _stop.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            pkt = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        seq = pkt.get("seq", 0)
        if pkt.get("type") != "meta" and seq <= last_seq:
            continue
        last_seq = max(last_seq, seq)
        publish(pkt)
    sock.close()


def source_replay(path: str, loop: bool, speed: float) -> None:
    with open(path, encoding="utf-8") as f:
        packets = [json.loads(l) for l in f if l.strip()]
    if not packets:
        print("[bridge] empty log", file=sys.stderr)
        return
    print(f"[bridge] replaying {len(packets)} packets from {path}", flush=True)
    while not _stop.is_set():
        t0_wall = time.time()
        t0_pkt = next((p["t_pi"] for p in packets if "t_pi" in p), 0.0)
        for p in packets:
            if _stop.is_set():
                return
            if "t_pi" in p:
                target = t0_wall + (p["t_pi"] - t0_pkt) / max(speed, 1e-6)
                delay = target - time.time()
                if delay > 0:
                    time.sleep(min(delay, 1.0))
            publish(p)
        if not loop:
            return
        time.sleep(0.5)


def source_demo(hz: float = 30.0) -> None:
    sys.path.insert(0, HERE)
    dt = 1.0 / hz
    seq = 0
    t_start = time.time()
    dx, dy = -180.0, 95.0
    yaw, pitch, roll = -4.2, 2.6, -3.1
    zoom = -7.0
    state = "SEARCH"
    lost_at = 9.0
    print("[bridge] demo mode (no Pi)", flush=True)
    while not _stop.is_set():
        el = time.time() - t_start
        seq += 1

        if lost_at < el < lost_at + 1.6:
            state = "LOST"
            k = 1.0
        else:
            k = 0.965
            if abs(dx) < 40 and abs(dy) < 40 and abs(yaw) < 1.0:
                state = "MICROMETRY"
            elif abs(dx) < 90:
                state = "LOCK_ON"
            else:
                state = "SEARCH"

        dx *= k
        dy *= k
        yaw *= k
        pitch *= k
        roll *= k
        zoom *= k

        tremor = 1.0 if state != "LOST" else 3.0
        dxn = dx + random.gauss(0, 2.2 * tremor)
        dyn = dy + random.gauss(0, 2.2 * tremor)
        rolln = roll + random.gauss(0, 0.482 * tremor)
        pitchn = pitch + random.gauss(0, 0.482 * tremor)
        yawn = yaw + random.gauss(0, 0.18 * tremor)

        vision_ok = state != "LOST"
        valid = ["roll", "pitch"] + (["trans", "yaw", "zoom"] if vision_ok else [])
        conf = 0.05 if not vision_ok else min(0.97, 0.35 + 0.6 * math.tanh(el / 6))

        pkt = {
            "v": 2, "seq": seq, "t_pi": time.time(), "state": state,
            "locked": bool(state == "MICROMETRY" and math.hypot(dxn, dyn) < 8
                           and abs(rolln) < 1.5 and abs(pitchn) < 1.5),
            "g": {
                "move_x_px": round(dxn, 2) if vision_ok else None,
                "move_y_px": round(dyn, 2) if vision_ok else None,
                "yaw_deg": round(yawn, 3) if vision_ok else None,
                "pitch_deg": round(pitchn, 3),
                "roll_deg": round(rolln, 3),
                "zoom_pct": round(zoom, 2) if vision_ok else None,
                "valid": valid,
            },
            "tol": {"trans_px": 8.0, "yaw_deg": 0.5, "pitch_deg": 1.5,
                    "roll_deg": 1.5, "zoom_pct": 2.0},
            "d": {
                "confidence": round(conf, 3),
                "inlier_ratio": round(conf * 0.8, 3),
                "n_matches": 240 if vision_ok else 31,
                "n_inliers": int(240 * conf * 0.8) if vision_ok else 4,
                "rho": round(0.5 + 0.45 * conf, 3) if vision_ok else 0.11,
                "psr": round(6 + 9 * conf, 2) if vision_ok else 2.1,
                "estimator": "phase_correlation" if state == "MICROMETRY" else "orb_search",
                "gyro_mag_dps": round(abs(random.gauss(0, 0.06 * tremor)), 4),
                "still": tremor == 1.0,
                "ms_total": round(7.9 + random.gauss(0, 1.4), 2),
                "fps": round(hz - random.random(), 1),
                "temp_c": round(58 + 6 * math.tanh(el / 40) + random.gauss(0, .3), 1),
                "throttled": "0x0",
            },
        }
        publish(pkt)
        time.sleep(dt)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            return self._sse()
        if self.path in ("/", "/index.html"):
            return self._file(UI_FILE, "text/html; charset=utf-8")
        self.send_error(404)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404, "operator_ui_web.html not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self):
        q: queue.Queue = queue.Queue(maxsize=1)
        with _lock:
            _subs.append(q)
            first = _latest
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            if first:
                self._emit(first)
            while not _stop.is_set():
                try:
                    pkt = q.get(timeout=2.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self._emit(pkt)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _lock:
                if q in _subs:
                    _subs.remove(q)

    def _emit(self, pkt: dict):
        self.wfile.write(b"data: " + json.dumps(pkt).encode("utf-8") + b"\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--live", action="store_true", help="listen to UDP from the Pi")
    src.add_argument("--replay", metavar="LOG.jsonl", help="replay a log")
    src.add_argument("--demo", action="store_true", help="synthetic session")
    ap.add_argument("--udp-port", type=int, default=5005)
    ap.add_argument("--port", type=int, default=8080, help="HTTP port")
    ap.add_argument("--loop", action="store_true", help="loop the log")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    args = ap.parse_args()

    if args.live:
        t = threading.Thread(target=source_udp, args=(args.udp_port,), daemon=True)
    elif args.replay:
        t = threading.Thread(target=source_replay,
                             args=(args.replay, args.loop, args.speed), daemon=True)
    else:
        t = threading.Thread(target=source_demo, daemon=True)
    t.start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    print(f"UI IN http://localhost:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        srv.server_close()


if __name__ == "__main__":
    main()
