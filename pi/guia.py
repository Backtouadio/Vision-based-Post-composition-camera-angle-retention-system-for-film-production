# Merge of the two channels: homography + IMU -> GuidanceSignal, and the
# smoothing of the orders the operator sees.
from __future__ import annotations

import math
import time as _time
from collections import deque
from typing import Optional

import numpy as np

from nucleo import Config, GuidanceSignal


class GuidanceMapper:

    def __init__(self, config: Config, suavizador=None) -> None:
        self.last_pitch_correction_px: float = 0.0
        self.last_yaw_correction_px: float = 0.0
        self.last_raw_pan_px: float = 0.0
        self.last_raw_tilt_px: float = 0.0
        self.last_raw_zoom_rel: float = 1.0
        self.last_raw_turn_deg: float = 0.0
        self._suav = suavizador
        self._cfg = config

    def map(
        self,
        transform: Optional[np.ndarray],
        confidence: float,
        state: str,
        imu_sample=None,
        ref_model=None,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        frame_size: Optional[tuple[int, int]] = None,
        yaw_deg: Optional[float] = None,
        t: Optional[float] = None,
        sel_gana: Optional[str] = None,
    ) -> GuidanceSignal:
        signal = GuidanceSignal(state=state, confidence=confidence, sel_gana=sel_gana)

        if frame_size is None and ref_model is not None:
            img = getattr(ref_model, "image_gray", None)
            if img is not None:
                frame_size = (img.shape[1], img.shape[0])

        ref_roll  = ref_model.ref_roll  if ref_model else 0.0
        ref_pitch = ref_model.ref_pitch if ref_model else 0.0

        if roll is not None and pitch is not None:
            signal.roll_err  = roll  - ref_roll
            signal.pitch_err = pitch - ref_pitch
        elif imu_sample is not None:
            r, p = _imu_to_roll_pitch(imu_sample)
            signal.roll_err  = r - ref_roll
            signal.pitch_err = p - ref_pitch

        if transform is not None:
            pan_px, tilt_px, zoom_rel = _decompose_homography(transform, frame_size)

            tilt_px, self.last_pitch_correction_px = _strip_pitch(
                tilt_px, signal.pitch_err, self._cfg)
            pan_px, self.last_yaw_correction_px = _strip_yaw(
                pan_px, yaw_deg, self._cfg)

            turn = 0.0 if yaw_deg is None else -float(yaw_deg)

            self.last_raw_pan_px = pan_px
            self.last_raw_tilt_px = tilt_px
            self.last_raw_zoom_rel = zoom_rel
            self.last_raw_turn_deg = turn

            if self._suav is not None:
                pan_px, tilt_px, zoom_rel, turn = self._suav.aplicar(
                    _time.monotonic() if t is None else float(t),
                    pan_px, tilt_px, zoom_rel, turn)

            if abs(turn) <= self._cfg.yaw_deadband_deg:
                turn = 0.0
            signal.turn_deg = turn

            db = self._cfg.deadband_px
            signal.pan_px  = pan_px  if abs(pan_px)  > db else 0.0
            signal.tilt_px = tilt_px if abs(tilt_px) > db else 0.0
            signal.zoom_rel = zoom_rel

            turn_ok = (abs(signal.turn_deg) <= self._cfg.lock_turn_deg_threshold
                       if getattr(self._cfg, "yaw_gates_lock", True) else True)
            framing_locked = (
                abs(pan_px)  <= self._cfg.lock_px_threshold and
                abs(tilt_px) <= self._cfg.lock_px_threshold and
                turn_ok
            )
        else:
            framing_locked = False
            self.last_pitch_correction_px = 0.0
            self.last_yaw_correction_px = 0.0
            if self._suav is not None:
                self._suav.reset()

        leveling_locked = (
            abs(math.degrees(signal.roll_err))  <= self._cfg.lock_deg_threshold and
            abs(math.degrees(signal.pitch_err)) <= self._cfg.lock_deg_threshold
        )

        signal.locked = framing_locked and leveling_locked

        return signal


def _strip_pitch(tilt_px: float, pitch_err: float, cfg) -> tuple[float, float]:
    if not getattr(cfg, "pitch_compensation", False):
        return tilt_px, 0.0
    if abs(math.degrees(pitch_err)) > cfg.pitch_comp_max_deg:
        return tilt_px, 0.0
    correccion = cfg.focal_length_px * pitch_err
    return tilt_px + correccion, correccion


def _strip_yaw(pan_px: float, yaw_deg, cfg) -> tuple[float, float]:
    if yaw_deg is None:
        return pan_px, 0.0
    if not getattr(cfg, "yaw_compensation", False):
        return pan_px, 0.0
    if abs(yaw_deg) > cfg.yaw_max_deg:
        return pan_px, 0.0
    correccion = cfg.focal_length_px * math.tan(math.radians(yaw_deg))
    return pan_px + correccion, correccion


def _decompose_homography(
    H: np.ndarray, frame_size: Optional[tuple[int, int]] = None
) -> tuple[float, float, float]:
    if H.shape == (2, 3):
        H = np.vstack([H, [0, 0, 1]])

    M = H[:2, :2]
    scale = math.sqrt(abs(np.linalg.det(M)))

    if frame_size is not None:
        w, h = frame_size
        cx, cy = w / 2.0, h / 2.0
        p = H @ np.array([cx, cy, 1.0], dtype=float)
        if abs(p[2]) < 1e-12:
            dx, dy = float(H[0, 2]), float(H[1, 2])
        else:
            dx, dy = float(p[0] / p[2] - cx), float(p[1] / p[2] - cy)
    else:
        dx, dy = float(H[0, 2]), float(H[1, 2])

    zoom_rel = 1.0 / scale if scale > 1e-9 else 1.0
    return dx, dy, zoom_rel


def _imu_to_roll_pitch(imu_sample) -> tuple[float, float]:
    ax, ay, az = imu_sample.accel
    roll  = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    return roll, pitch


class VentanaMediana:

    def __init__(self, ventana_s: float = 0.5, max_muestras: int = 512) -> None:
        self.ventana_s = float(ventana_s)
        self._buf: deque = deque(maxlen=max_muestras)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, t: float, v: float) -> float:
        self._buf.append((float(t), float(v)))
        corte = float(t) - self.ventana_s
        while self._buf and self._buf[0][0] < corte:
            self._buf.popleft()
        vals = sorted(x[1] for x in self._buf)
        n = len(vals)
        if n == 0:
            return float(v)
        m = n // 2
        return vals[m] if n % 2 else vals[m - 1]

    def __len__(self) -> int:
        return len(self._buf)




class SentidoEstable:

    def __init__(self, margen: float) -> None:
        self.margen = abs(float(margen))
        self._signo = 0

    def reset(self) -> None:
        self._signo = 0

    def apply(self, v: float) -> float:
        s = 0 if v == 0 else (1 if v > 0 else -1)
        if s == 0 or self._signo == 0 or s == self._signo:
            if s != 0:
                self._signo = s
            return float(v)
        if abs(v) >= self.margen:
            self._signo = s
            return float(v)
        return 0.0


class BandaMuerta:

    def __init__(self, umbral_on: float, umbral_off: float) -> None:
        self.on = abs(float(umbral_on))
        self.off = abs(float(umbral_off))
        if self.off > self.on:
            self.on, self.off = self.off, self.on
        self._activa = False

    def reset(self) -> None:
        self._activa = False

    def apply(self, v: float) -> float:
        a = abs(v)
        if self._activa:
            if a <= self.off:
                self._activa = False
                return 0.0
            return float(v)
        if a >= self.on:
            self._activa = True
            return float(v)
        return 0.0


class FiltroEje:

    def __init__(self, ventana_s: float, banda_on: float, banda_off: float,
                 margen_sentido: float, paso: Optional[float] = None) -> None:
        self.mediana = VentanaMediana(ventana_s)
        self.sentido = SentidoEstable(margen_sentido)
        self.banda = BandaMuerta(banda_on, banda_off)
        self.paso = paso
        self.ultimo_crudo = 0.0
        self.ultimo = 0.0

    def reset(self) -> None:
        self.mediana.reset()
        self.sentido.reset()
        self.banda.reset()
        self.ultimo_crudo = 0.0
        self.ultimo = 0.0

    def __call__(self, t: float, v: Optional[float]) -> float:
        if v is None:
            return self.ultimo
        self.ultimo_crudo = float(v)
        x = self.mediana.push(t, float(v))
        x = self.sentido.apply(x)
        x = self.banda.apply(x)
        if self.paso:
            x = round(x / self.paso) * self.paso
        self.ultimo = float(x)
        return self.ultimo


class SuavizadorGuia:

    def __init__(self, cfg) -> None:
        g = lambda n, d: float(getattr(cfg, n, d))
        self.pan = FiltroEje(g("suav_ventana_px_s", 0.4),
                             g("suav_banda_on_px", 10.0),
                             g("suav_banda_off_px", 5.0),
                             g("suav_margen_px", 12.0),
                             getattr(cfg, "suav_paso_px", None))
        self.tilt = FiltroEje(g("suav_ventana_px_s", 0.4),
                              g("suav_banda_on_px", 10.0),
                              g("suav_banda_off_px", 5.0),
                              g("suav_margen_px", 12.0),
                              getattr(cfg, "suav_paso_px", None))
        self.zoom = FiltroEje(g("suav_ventana_zoom_s", 0.8),
                              g("suav_banda_on_zoom_pct", 6.0),
                              g("suav_banda_off_zoom_pct", 3.0),
                              g("suav_margen_zoom_pct", 8.0),
                              getattr(cfg, "suav_paso_zoom_pct", None))
        self.turn = FiltroEje(g("suav_ventana_turn_s", 0.6),
                              g("suav_banda_on_turn_deg", 2.5),
                              g("suav_banda_off_turn_deg", 1.5),
                              g("suav_margen_turn_deg", 3.0),
                              getattr(cfg, "suav_paso_turn_deg", None))

    def reset(self) -> None:
        for f in (self.pan, self.tilt, self.zoom, self.turn):
            f.reset()

    def aplicar(self, t: float, pan_px: float, tilt_px: float,
                zoom_rel: float, turn_deg: float):
        pan = self.pan(t, pan_px)
        tilt = self.tilt(t, tilt_px)
        zpct = self.zoom(t, (float(zoom_rel) - 1.0) * 100.0)
        turn = self.turn(t, turn_deg)
        return pan, tilt, 1.0 + zpct / 100.0, turn


class CoherenciaYaw:

    def __init__(self, salto_max_deg: float = 8.0, confirmaciones: int = 2,
                 caducidad_s: float = 0.4) -> None:
        self.salto_max = abs(float(salto_max_deg))
        self.confirmaciones = max(1, int(confirmaciones))
        self.caducidad_s = float(caducidad_s)
        self._ultimo: Optional[float] = None
        self._t_ultimo: float = -1e9
        self._pendientes: list = []
        self.n_rechazos = 0
        self.n_aceptados = 0

    def reset(self) -> None:
        self._ultimo = None
        self._t_ultimo = -1e9
        self._pendientes = []

    def filtra(self, t: float, yaw_deg: Optional[float]) -> Optional[float]:
        if yaw_deg is None:
            return None
        y = float(yaw_deg)
        if self._ultimo is None or (t - self._t_ultimo) > self.caducidad_s:
            self._aceptar(t, y)
            return y
        if abs(y - self._ultimo) <= self.salto_max:
            self._pendientes = []
            self._aceptar(t, y)
            return y
        self._pendientes.append(y)
        if len(self._pendientes) >= self.confirmaciones:
            if max(self._pendientes) - min(self._pendientes) <= self.salto_max:
                self._pendientes = []
                self._aceptar(t, y)
                return y
            self._pendientes = self._pendientes[-(self.confirmaciones - 1):] \
                if self.confirmaciones > 1 else []
        self.n_rechazos += 1
        return None

    def _aceptar(self, t: float, y: float) -> None:
        self._ultimo = y
        self._t_ultimo = float(t)
        self.n_aceptados += 1
