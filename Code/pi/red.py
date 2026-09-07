# Remote output: packet schema, UDP/JSONL emitter, TCP command server,
# SETUP mode and the per-frame logging.
from __future__ import annotations

import json
import math
import os
import queue
import re
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


PACKET_VERSION = 2




def _f(x: Any, nd: int = 4) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


@dataclass
class Guidance:

    move_x_px: Optional[float] = None
    move_y_px: Optional[float] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    zoom_pct: Optional[float] = None

    valid: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "move_x_px": _f(self.move_x_px),
            "move_y_px": _f(self.move_y_px),
            "yaw_deg": _f(self.yaw_deg),
            "pitch_deg": _f(self.pitch_deg),
            "roll_deg": _f(self.roll_deg),
            "zoom_pct": _f(self.zoom_pct),
            "valid": list(self.valid),
        }


@dataclass
class Tolerance:

    trans_px: float = 8.0
    yaw_deg: float = 0.5
    pitch_deg: float = 1.5
    roll_deg: float = 1.5
    zoom_pct: float = 2.0

    def as_dict(self) -> Dict[str, Any]:
        return {k: _f(v) for k, v in asdict(self).items()}


@dataclass
class Diagnostics:

    homography: Optional[List[float]] = None
    confidence: Optional[float] = None
    inlier_ratio: Optional[float] = None
    n_matches: Optional[int] = None
    n_inliers: Optional[int] = None
    rho: Optional[float] = None
    psr: Optional[float] = None
    estimator: Optional[str] = None

    roll_raw_deg: Optional[float] = None
    pitch_raw_deg: Optional[float] = None
    ref_roll_deg: Optional[float] = None
    ref_pitch_deg: Optional[float] = None
    gyro_mag_dps: Optional[float] = None
    still: Optional[bool] = None

    t_capture: Optional[float] = None
    ms_grab: Optional[float] = None
    ms_estimate: Optional[float] = None
    ms_map: Optional[float] = None
    ms_total: Optional[float] = None
    fps: Optional[float] = None
    temp_c: Optional[float] = None
    throttled: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = _f(v)
            elif isinstance(v, list):
                d[k] = [_f(x) for x in v]
        return d


@dataclass
class TelemetryPacket:
    seq: int
    t_pi: float
    state: str
    locked: bool
    guidance: Guidance
    tolerance: Tolerance
    diagnostics: Optional[Diagnostics] = None
    nota: Optional[str] = None
    v: int = PACKET_VERSION

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "v": self.v,
            "seq": int(self.seq),
            "t_pi": _f(self.t_pi, 4),
            "state": self.state,
            "locked": bool(self.locked),
            "g": self.guidance.as_dict(),
            "tol": self.tolerance.as_dict(),
        }
        if self.nota:
            d["nota"] = self.nota
        if self.diagnostics is not None:
            d["d"] = self.diagnostics.as_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"))


@dataclass
class SessionMeta:

    session_id: str
    started_at: float
    reference_path: Optional[str] = None
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    sensor_mode: Optional[str] = None
    config_hash: Optional[str] = None
    mount_signs: Optional[List[int]] = None
    ref_roll_deg: Optional[float] = None
    ref_pitch_deg: Optional[float] = None
    notes: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"v": PACKET_VERSION, "type": "meta", **asdict(self)}


class OperatorOutput:

    def render(self, packet: TelemetryPacket) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NetworkOutput(OperatorOutput):

    def __init__(self, host: str = "127.0.0.1", port: int = 5005,
                 with_diagnostics: bool = True):
        self.addr = (host, port)
        self.with_diagnostics = with_diagnostics
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self._seq = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def render(self, packet: TelemetryPacket) -> None:
        if not self.with_diagnostics:
            packet = TelemetryPacket(**{**packet.__dict__, "diagnostics": None})
        try:
            self.sock.sendto(packet.to_json().encode("utf-8"), self.addr)
        except (BlockingIOError, OSError):
            pass

    def send_meta(self, meta: SessionMeta) -> None:
        try:
            self.sock.sendto(json.dumps(meta.as_dict()).encode("utf-8"), self.addr)
        except (BlockingIOError, OSError):
            pass

    def close(self) -> None:
        self.sock.close()


class LogWriter(OperatorOutput):

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")

    def render(self, packet: TelemetryPacket) -> None:
        self.f.write(packet.to_json() + "\n")

    def write_meta(self, meta: SessionMeta) -> None:
        self.f.write(json.dumps(meta.as_dict()) + "\n")

    def close(self) -> None:
        self.f.close()


class FanOut(OperatorOutput):

    def __init__(self, *outputs: OperatorOutput):
        self.outputs = outputs

    def render(self, packet: TelemetryPacket) -> None:
        for o in self.outputs:
            o.render(packet)

    def close(self) -> None:
        for o in self.outputs:
            o.close()


def now() -> float:
    return time.time()


def make_packet(seq: int, state: str, *, move_x_px=None, move_y_px=None,
                yaw_deg=None, pitch_deg=None, roll_deg=None, zoom_pct=None,
                valid=None, tolerance: Optional[Tolerance] = None,
                diagnostics: Optional[Diagnostics] = None,
                locked: bool = False, nota=None) -> TelemetryPacket:
    tol = tolerance or Tolerance()
    g = Guidance(move_x_px, move_y_px, yaw_deg, pitch_deg, roll_deg, zoom_pct,
                 list(valid or []))
    return TelemetryPacket(seq=seq, t_pi=now(), state=state, locked=locked,
                           guidance=g, tolerance=tol, diagnostics=diagnostics,
                           nota=nota)


_META_CADA_S = 1.0


class EmisorTelemetria:

    def __init__(self, cfg, *, host=None, port=5005, log_path=None,
                 zoom_medido=False, cada_n=1, con_diagnostico=True,
                 ref_path=None):
        salidas = []
        self._red = NetworkOutput(host, port, with_diagnostics=con_diagnostico) if host else None
        self._jsonl = LogWriter(log_path) if log_path else None
        if self._red:
            salidas.append(self._red)
        if self._jsonl:
            salidas.append(self._jsonl)
        self._out = FanOut(*salidas) if salidas else None

        self._cada_n = max(1, int(cada_n))
        self._zoom_medido = bool(zoom_medido)
        self._diag = bool(con_diagnostico)
        self._tol = Tolerance(
            trans_px=float(getattr(cfg, "lock_px_threshold", 15.0)),
            yaw_deg=float(getattr(cfg, "lock_turn_deg_threshold", 2.0)),
            pitch_deg=float(getattr(cfg, "lock_deg_threshold", 5.0)),
            roll_deg=float(getattr(cfg, "lock_deg_threshold", 5.0)),
            zoom_pct=float(getattr(cfg, "lock_zoom_pct", 2.0)),
        )
        self._ref_path = str(ref_path) if ref_path else None
        self._sesion = uuid.uuid4().hex[:8]
        self._t0 = time.time()
        self._n = self._seq = 0
        self._t_meta = 0.0
        self._t_prev = None
        self._muerto = False

    def push(self, signal, *, result=None, frame=None, roll=None, pitch=None,
             ref_model=None, proc_ms=None, yaw_medido=False, nota=None):
        if self._out is None or self._muerto:
            return
        self._n += 1
        if self._n % self._cada_n:
            return
        try:
            self._emitir(signal, result, frame, roll, pitch, ref_model,
                         proc_ms, yaw_medido, nota)
        except Exception as e:
            self._muerto = True
            print(f"[emitter] TELEMETRY DISABLED after a failure: {e!r}",
                  flush=True)

    def _emitir(self, signal, result, frame, roll, pitch, ref_model,
                proc_ms, yaw_medido, nota=None):
        ahora = time.time()
        if ahora - self._t_meta >= _META_CADA_S:
            self._t_meta = ahora
            self._meta(frame, ref_model)

        hay_trans = getattr(result, "transform", None) is not None
        hay_imu = roll is not None

        valid = []
        if hay_trans:
            valid.append("trans")
            if self._zoom_medido:
                valid.append("zoom")
        if yaw_medido:
            valid.append("yaw")
        if hay_imu:
            valid += ["roll", "pitch"]

        fps = None
        if self._t_prev is not None:
            dt = ahora - self._t_prev
            fps = 1.0 / dt if dt > 1e-6 else None
        self._t_prev = ahora

        self._seq += 1
        self._out.render(make_packet(
            self._seq, signal.state.upper(),
            move_x_px=signal.pan_px,
            move_y_px=-signal.tilt_px,
            yaw_deg=signal.turn_deg,
            pitch_deg=math.degrees(signal.pitch_err),
            roll_deg=-math.degrees(signal.roll_err),
            zoom_pct=(signal.zoom_rel - 1.0) * 100.0,
            valid=valid,
            tolerance=self._tol,
            locked=bool(signal.locked),
            nota=nota or None,
            diagnostics=self._diagnostico(signal, result, frame, roll, pitch,
                                          proc_ms, fps) if self._diag else None,
        ))

    def _diagnostico(self, signal, result, frame, roll, pitch, proc_ms, fps):
        er = getattr(result, "estimator_result", None)
        t = getattr(result, "timing", None) or {}
        return Diagnostics(
            confidence=signal.confidence,
            n_inliers=getattr(er, "n_inliers", None),
            n_matches=getattr(er, "n_matches", None),
            estimator=getattr(er, "method", None) or None,
            rho=getattr(er, "raw_score", None),
            roll_raw_deg=None if roll is None else math.degrees(roll),
            pitch_raw_deg=None if pitch is None else math.degrees(pitch),
            t_capture=getattr(frame, "timestamp", None),
            ms_estimate=t.get("search_ms", t.get("micro_ms")),
            ms_total=proc_ms,
            fps=fps,
        )

    def _meta(self, frame, ref_model):
        img = getattr(frame, "image", None)
        h, w = (img.shape[0], img.shape[1]) if img is not None else (None, None)
        m = SessionMeta(
            session_id=self._sesion,
            started_at=self._t0,
            reference_path=self._ref_path,
            frame_w=w, frame_h=h,
            ref_roll_deg=(math.degrees(ref_model.ref_roll)
                          if ref_model is not None else None),
            ref_pitch_deg=(math.degrees(ref_model.ref_pitch)
                           if ref_model is not None else None),
        )
        if self._red:
            self._red.send_meta(m)
        if self._jsonl:
            self._jsonl.write_meta(m)

    def close(self):
        if self._out is not None:
            try:
                self._out.close()
            except Exception:
                pass
            self._out = None


PUERTO = 8081
PATRON = re.compile(r"^S(?P<escena>[A-Za-z0-9]+)_(?P<secuencia>[A-Za-z0-9]+)"
                    r"_(?P<plano>[A-Za-z0-9]+)\.png$")
COLA = {"capturar", "guardar", "reintentar", "descartar", "guiar", "parar"}
DIRECTOS = {"borrar", "renombrar"}
COMANDOS = COLA | DIRECTOS


def nombre_de(escena, secuencia, plano) -> str:
    return f"S{escena}_{secuencia}_{plano}.png"


def catalogo(refs_dir: Path) -> dict:
    arbol: dict = {}
    planos = []
    for f in sorted(refs_dir.glob("*.png")):
        m = PATRON.match(f.name)
        if not m:
            continue
        e, s, p = m["escena"], m["secuencia"], m["plano"]
        arbol.setdefault(e, {}).setdefault(s, []).append(p)
        planos.append({"file": f.name, "escena": e, "secuencia": s, "plano": p,
                       "imu": (f.parent / (f.name + ".imu.json")).exists()})
    return {"arbol": arbol, "planos": planos}


class ServidorPi:
    def __init__(self, refs_dir, puerto: int = PUERTO):
        self.refs = Path(refs_dir)
        self.refs.mkdir(parents=True, exist_ok=True)
        self.cola: "queue.Queue[dict]" = queue.Queue()
        self.en_uso = lambda: None
        self._srv = ThreadingHTTPServer(("0.0.0.0", puerto), self._handler())
        self._srv.daemon_threads = True
        self.puerto = puerto

    def start(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        print(f"[server] commands and thumbnails at http://0.0.0.0:{self.puerto}")

    def close(self):
        try:
            self._srv.shutdown()
            self._srv.server_close()
        except Exception:
            pass

    def pendiente(self):
        try:
            return self.cola.get_nowait()
        except queue.Empty:
            return None

    def _fichero(self, nombre):
        if not nombre or not PATRON.match(str(nombre)):
            return None
        f = self.refs / nombre
        return f if f.is_file() else None

    @staticmethod
    def _sidecar(f: Path) -> Path:
        return f.with_suffix(f.suffix + ".imu.json")

    def borrar(self, nombre):
        f = self._fichero(nombre)
        if f is None:
            return False, "does not exist"
        actual = self.en_uso()
        if actual is not None and Path(actual).resolve() == f.resolve():
            return False, "it is the shot being guided right now"
        f.unlink()
        s = self._sidecar(f)
        if s.exists():
            s.unlink()
        print(f"[server] deleted {f.name}", flush=True)
        return True, ""

    def renombrar(self, nombre, nuevo):
        f = self._fichero(nombre)
        if f is None:
            return False, "does not exist"
        if not nuevo or not PATRON.match(str(nuevo)):
            return False, "new name is not valid"
        destino = self.refs / nuevo
        if destino.exists():
            return False, f"{nuevo} already exists"
        actual = self.en_uso()
        if actual is not None and Path(actual).resolve() == f.resolve():
            return False, "it is the shot being guided right now"
        s = self._sidecar(f)
        if s.exists():
            s.replace(self._sidecar(destino))
        f.replace(destino)
        print(f"[server] {f.name} -> {destino.name}", flush=True)
        return True, ""

    def _handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *a):
                pass

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def _json(self, obj, code=200):
                cuerpo = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(cuerpo)))
                self._cors()
                self.end_headers()
                self.wfile.write(cuerpo)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                if self.path in ("/health", "/"):
                    return self._json({"ok": True, "refs": str(srv.refs)})
                if self.path == "/shots":
                    return self._json(catalogo(srv.refs))
                if self.path.startswith("/refs/"):
                    nombre = self.path[len("/refs/"):].split("?")[0]
                    if not PATRON.match(nombre):
                        return self._json({"error": "name is not valid"}, 400)
                    f = srv.refs / nombre
                    if not f.is_file():
                        return self._json({"error": "does not exist"}, 404)
                    datos = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(datos)))
                    self._cors()
                    self.end_headers()
                    self.wfile.write(datos)
                    return
                self._json({"error": "?"}, 404)

            def do_POST(self):
                if self.path != "/cmd":
                    return self._json({"error": "?"}, 404)
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    cmd = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self._json({"error": "invalid json"}, 400)
                c = cmd.get("cmd")
                if c not in COMANDOS:
                    return self._json({"error": f"unknown command: {c!r}"}, 400)
                if c in DIRECTOS:
                    ok, motivo = (srv.borrar(cmd.get("file")) if c == "borrar"
                                  else srv.renombrar(cmd.get("file"),
                                                     cmd.get("nuevo")))
                    if not ok:
                        return self._json({"error": motivo}, 409)
                    return self._json({"ok": True, "catalogo": catalogo(srv.refs)})
                srv.cola.put(cmd)
                print(f"[server] command {cmd}", flush=True)
                return self._json({"ok": True})

        return H


CUENTA_S = 1.0


class ModoSetup:
    def __init__(self, cfg, servidor, refs_dir, *, ref_inicial=None):
        self.cfg = cfg
        self.srv = servidor
        self.refs = Path(refs_dir)
        self.estado = "GUIANDO" if ref_inicial else "SETUP"
        self.ref_actual = ref_inicial
        self._t0 = 0.0
        self._digito = 3
        self._roll_antes = self._pitch_antes = None
        self._pend = None
        self._aviso = ""

    def en_setup(self) -> bool:
        return self.estado != "GUIANDO"

    def etiqueta(self) -> str:
        if self.estado == "ARMANDO":
            return f"ARMANDO_{self._digito}"
        return self.estado

    def aviso(self) -> str:
        return self._aviso

    def rechazar_referencia(self, motivo: str) -> None:
        self.estado = "SETUP"
        self.ref_actual = None
        self._aviso = motivo
        print(f"[setup] reference rejected: {motivo}", flush=True)

    def tick(self, frame, roll, pitch):
        cmd = self.srv.pendiente()
        if cmd is not None:
            ruta = self._comando(cmd, roll, pitch)
            if ruta is not None:
                return ruta

        if self.estado == "ARMANDO":
            t = time.monotonic() - self._t0
            self._digito = max(1, 3 - int(t))
            if t >= 3 * CUENTA_S:
                self.estado = "DISPARO"
                self._roll_antes, self._pitch_antes = roll, pitch
        elif self.estado == "DISPARO":
            self._disparar(frame, roll, pitch)
        return None

    def _comando(self, cmd, roll, pitch):
        c = cmd.get("cmd")
        if c == "capturar":
            self._aviso = ""
            self._pend = None
            self.estado = "ARMANDO"
            self._t0 = time.monotonic()
            self._digito = 3
        elif c == "reintentar":
            self._aviso = ""
            self.estado = "ARMANDO"
            self._t0 = time.monotonic()
            self._digito = 3
        elif c == "descartar":
            self._pend = None
            self._aviso = ""
            self.estado = "SETUP"
        elif c == "guardar":
            self._guardar(cmd)
        elif c == "parar":
            self.estado = "SETUP"
            self.ref_actual = None
        elif c == "guiar":
            ruta = self.refs / str(cmd.get("file", ""))
            if ruta.is_file():
                self.ref_actual = ruta
                self.estado = "GUIANDO"
                self._aviso = ""
                return ruta
            self._aviso = f"{ruta.name} does not exist"
        return None

    def _disparar(self, frame, roll, pitch):
        if roll is not None and self._roll_antes is not None:
            movido = max(abs(math.degrees(roll - self._roll_antes)),
                         abs(math.degrees(pitch - self._pitch_antes)))
            if movido > self.cfg.capture_move_tolerance_deg:
                self._aviso = (f"TOO MUCH MOVEMENT -- {movido:.1f} degrees "
                               f"between the two IMU readings (limit "
                               f"{self.cfg.capture_move_tolerance_deg:.1f}). "
                               f"Nothing has been saved.")
                self.estado = "SETUP"
                print(f"[setup] {self._aviso}", flush=True)
                return
        self._pend = (frame.image.copy(), roll, pitch)
        self.estado = "NOMBRAR"
        print("[setup] capture held in memory, waiting for a name", flush=True)

    def _guardar(self, cmd):
        if self._pend is None:
            self._aviso = "there is nothing captured"
            return
        imagen, roll, pitch = self._pend
        nombre = nombre_de(cmd.get("escena", "1"), cmd.get("secuencia", "A"),
                           cmd.get("plano", "1"))
        destino = self.refs / nombre
        sidecar = destino.with_suffix(destino.suffix + ".imu.json")

        tmp_s = str(sidecar) + ".tmp"
        tmp_p = str(destino.with_name(destino.stem + ".tmp.png"))
        if roll is not None:
            Path(tmp_s).write_text(json.dumps({"ref_roll": roll,
                                               "ref_pitch": pitch}, indent=2))
            os.replace(tmp_s, sidecar)
        elif sidecar.exists():
            sidecar.unlink()
        cv2.imwrite(tmp_p, imagen)
        os.replace(tmp_p, destino)

        self._pend = None
        self.estado = "SETUP"
        self._aviso = f"saved {nombre}"
        print(f"[setup] saved {destino}"
              f"{'' if roll is not None else '  (NO sidecar: no IMU)'}",
              flush=True)


class RunLogger:
    SCHEMA = 1

    def __init__(
        self,
        config,
        meta: Optional[dict] = None,
        *,
        enabled: Optional[bool] = None,
        path=None,
    ) -> None:
        self._every = max(1, int(getattr(config, "log_every_n_frames", 1)))

        if enabled is None:
            enabled = bool(getattr(config, "log_enabled", True)) and bool(
                getattr(config, "log_dir", "")
            )
        self._enabled = bool(enabled)

        self._fh = None
        self._seen = 0
        self._written = 0
        self._path: Optional[Path] = None

        if not self._enabled:
            return

        try:
            if path is None:
                log_dir = Path(getattr(config, "log_dir", "logs"))
                log_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = log_dir / f"run_{stamp}.jsonl"
            self._path = Path(path)
            self._fh = open(self._path, "w", buffering=1)
            self._write({
                "type": "meta",
                "schema": self.SCHEMA,
                "started": datetime.now().isoformat(timespec="seconds"),
                "log_every_n_frames": self._every,
                **(meta or {}),
            })
        except OSError as exc:
            print(f"[RunLogger] disabled (could not open log: {exc})")
            self._enabled = False
            self._fh = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def log(self, frame, result, signal, imu_sample=None, proc_ms=None,
            extra=None) -> None:
        if not self._enabled:
            return
        self._seen += 1
        if (self._seen - 1) % self._every != 0:
            return
        try:
            rec = self._build_record(frame, result, signal, imu_sample, proc_ms,
                                     extra)
        except Exception as exc:
            print(f"[RunLogger] record build failed, disabling: {exc}")
            self._enabled = False
            self._safe_close()
            return
        self._write(rec)
        self._written += 1

    def close(self) -> None:
        if self._fh is not None:
            self._write({
                "type": "end",
                "frames_seen": self._seen,
                "frames_logged": self._written,
                "ended": datetime.now().isoformat(timespec="seconds"),
            })
            self._safe_close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _build_record(self, frame, result, signal, imu_sample, proc_ms,
                      extra=None) -> dict:
        tx = ty = scale = None
        T = getattr(result, "transform", None)
        if T is not None:
            T = np.asarray(T, dtype=float)
            if T.shape == (2, 3):
                T = np.vstack([T, [0.0, 0.0, 1.0]])
            tx = float(T[0, 2])
            ty = float(T[1, 2])
            scale = float(math.sqrt(abs(np.linalg.det(T[:2, :2]))))

        st = getattr(result, "state", None)
        state = getattr(st, "name", str(st)).lower()

        er = getattr(result, "estimator_result", None)
        method = getattr(er, "method", "") if er is not None else ""
        n_inliers = int(getattr(er, "n_inliers", 0) or 0) if er is not None else 0

        gyro_mag = None
        if imu_sample is not None:
            try:
                gyro_mag = float(np.linalg.norm(np.asarray(imu_sample.gyro, dtype=float)))
            except Exception:
                gyro_mag = None

        rec = {
            "type": "frame",
            "idx": int(getattr(frame, "index", self._seen - 1)),
            "t": _r(getattr(frame, "timestamp", 0.0), 4),
            "state": state,
            "lost": bool(getattr(result, "lost", T is None)),
            "conf": _r(getattr(result, "confidence", 0.0)),
            "method": method,
            "n_inliers": n_inliers,
            "tx": _r(tx), "ty": _r(ty), "scale": _r(scale, 4),
            "pan_px": _r(getattr(signal, "pan_px", 0.0)),
            "tilt_px": _r(getattr(signal, "tilt_px", 0.0)),
            "zoom_rel": _r(getattr(signal, "zoom_rel", 1.0), 4),
            "roll_err_deg": _r(math.degrees(getattr(signal, "roll_err", 0.0))),
            "pitch_err_deg": _r(math.degrees(getattr(signal, "pitch_err", 0.0))),
            "locked": bool(getattr(signal, "locked", False)),
            "gyro_mag": _r(gyro_mag, 5),
            "turn_deg": _r(getattr(signal, "turn_deg", 0.0)),
            "ess_yaw_deg": _r(getattr(er, "yaw_deg", None)),
            "ess_pitch_deg": _r(getattr(er, "ess_pitch_deg", None)),
            "ess_roll_deg": _r(getattr(er, "ess_roll_deg", None)),
            "ess_inliers": int(getattr(er, "ess_inliers", 0) or 0),
            "sel_r_h": _r(getattr(er, "sel_r_h", None)),
            "sel_gana": getattr(er, "sel_gana", None),
            "sel_s_h": _r(getattr(er, "sel_s_h", None)),
            "sel_s_e": _r(getattr(er, "sel_s_e", None)),
            "yaw_vetado_deg": _r(getattr(er, "yaw_vetado_deg", None)),
            **{k: (_r(v) if isinstance(v, float) else v)
               for k, v in (extra or {}).items()},
        }
        if proc_ms is not None:
            rec["proc_ms"] = _r(proc_ms, 2)

        timing = getattr(result, "timing", None) or {}
        if "search_ms" in timing:
            rec["search_ms"] = _r(timing["search_ms"], 2)
        if "micro_ms" in timing:
            rec["micro_ms"] = _r(timing["micro_ms"], 2)
        return rec

    def _write(self, obj: dict) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(obj) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            print(f"[RunLogger] write failed, disabling: {exc}")
            self._enabled = False
            self._safe_close()

    def _safe_close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except OSError:
            pass
        self._fh = None


def _r(x, nd: int = 3):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(xf):
        return None
    return round(xf, nd)
