# Local output: OpenCV window (OperatorUI) and terminal output (HeadlessUI).
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from nucleo import Config, GuidanceSignal


_COL = {
    "search":      (0,   60,  220),
    "lock_on":     (0,  165,  255),
    "micrometry":  (200, 180,   0),
    "locked":      (30,  200,   30),
    "arrow":       (255, 255, 255),
    "crosshair":   (180, 180, 180),
    "text_dark":   (20,   20,  20),
    "text_light":  (240, 240, 240),
    "overlay_bg":  (10,   10,  10),
}

_STATE_COL = {
    "search":     _COL["search"],
    "lock_on":    _COL["lock_on"],
    "micrometry": _COL["micrometry"],
}

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FONTB = cv2.FONT_HERSHEY_DUPLEX


class OperatorUI:

    def __init__(
        self,
        config: Config,
        window_name: str = "TFG Guidance",
        max_arrow_px: int = 120,
        scale: float = 0.18,
        rotate: int | None = None,
    ) -> None:
        self._cfg = config
        self._win = window_name
        self._max_arrow = max_arrow_px
        self._scale = scale
        self._rotate = rotate
        self._win_open = False


    def __enter__(self) -> "OperatorUI":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._win_open:
            cv2.destroyWindow(self._win)
            self._win_open = False


    def draw(self, frame: np.ndarray, signal: GuidanceSignal) -> np.ndarray:
        canvas = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        state_col = _state_colour(signal)

        _draw_border(canvas, state_col, thickness=5)

        _draw_crosshair(canvas, cx, cy)

        if signal.pan_px != 0.0 or signal.tilt_px != 0.0:
            _draw_guidance_arrow(
                canvas, cx, cy,
                signal.pan_px, signal.tilt_px,
                self._scale, self._max_arrow,
            )

        if signal.locked:
            _draw_locked_banner(canvas, h, w)

        _draw_state_badge(canvas, signal, state_col)

        _draw_confidence(canvas, w, signal)

        _draw_leveling(canvas, h, signal)

        _draw_zoom(canvas, w, h, signal)

        return canvas

    def show(self, frame: np.ndarray, signal: GuidanceSignal) -> bool:
        if self._rotate is not None:
            frame = cv2.rotate(frame, self._rotate)
        canvas = self.draw(frame, signal)
        cv2.imshow(self._win, canvas)
        self._win_open = True
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")


def _state_colour(signal: GuidanceSignal):
    if signal.locked:
        return _COL["locked"]
    return _STATE_COL.get(signal.state.lower(), _COL["search"])


def _draw_border(img: np.ndarray, colour, thickness: int = 5) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), colour, thickness)


def _draw_crosshair(img: np.ndarray, cx: int, cy: int, size: int = 18) -> None:
    col = _COL["crosshair"]
    cv2.line(img, (cx - size, cy), (cx + size, cy), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy + size), col, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, col, 1, cv2.LINE_AA)


class HeadlessUI:

    def __init__(self, cfg, rate_hz: float = 5.0) -> None:
        self._cfg = cfg
        self._period = 1.0 / max(0.1, rate_hz)
        self._last = 0.0
        self._frames = 0
        self._states = {}

    def __enter__(self) -> "HeadlessUI":
        print("Headless mode: no window. Ctrl-C to stop.")
        print("pan/tilt/turn are ORDERS (what to do); roll/pitch are ERRORS "
              "(where you already are).")
        if getattr(self._cfg, "essential_yaw", False):
            print("turn: 'planar/rot' = H won the selection: planar scene "
                  "or pure rotation,\n      where E is degenerate. It does "
                  "NOT mean you are aligned.")
            print("turn: 'unmeasured' = the essential does not run in this "
                  "state, not that you are aligned.")
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._states:
            seen = "  ".join(f"{k}={v}" for k, v in sorted(self._states.items()))
            print(f"\n{self._frames} frames.  states seen: {seen}")

    def show(self, frame, signal: GuidanceSignal) -> bool:
        self._frames += 1
        self._states[signal.state] = self._states.get(signal.state, 0) + 1

        now = time.monotonic()
        if now - self._last < self._period:
            return True
        self._last = now

        dead = float(getattr(self._cfg, "deadband_px", 3.0))
        pan = ("hold      " if abs(signal.pan_px) <= dead else
               "move RIGHT" if signal.pan_px > 0 else "move LEFT ")
        tilt = ("hold     " if abs(signal.tilt_px) <= dead else
                "move DOWN" if signal.tilt_px > 0 else "move UP  ")

        roll_deg = math.degrees(signal.roll_err)
        pitch_deg = math.degrees(signal.pitch_err)
        lock_deg = float(getattr(self._cfg, "lock_deg_threshold", 1.5))
        roll_order = ("level     " if abs(roll_deg) <= lock_deg else
                      "roll LEFT " if roll_deg > 0 else "roll RIGHT")
        pitch_order = ("level    " if abs(pitch_deg) <= lock_deg else
                       "nose UP  " if pitch_deg > 0 else "nose DOWN")

        turn_deg = float(getattr(signal, "turn_deg", 0.0))
        turn_thr = float(getattr(self._cfg, "lock_turn_deg_threshold", 2.0))
        if not getattr(self._cfg, "essential_yaw", False):
            turn_txt = ""
        elif turn_deg == 0.0:
            if getattr(signal, "sel_gana", None) == "H":
                turn_txt = f" | {'  --  ':>6} {'planar/rot':<10}"
            else:
                turn_txt = f" | {'  --  ':>6} {'unmeasured':<10}"
        else:
            turn_word = ("aimed     " if abs(turn_deg) <= turn_thr else
                         "turn RIGHT" if turn_deg > 0 else "turn LEFT ")
            turn_txt = f" | {turn_deg:+6.2f}d {turn_word}"

        zoom_pct = (signal.zoom_rel - 1.0) * 100.0
        zoom_order = ("ok    " if abs(zoom_pct) < 2.0 else
                      "closer" if zoom_pct > 0 else "back  ")

        held = ""
        if (signal.state == "search" and turn_deg != 0.0
                and abs(turn_deg) > turn_thr):
            held = "  <- SEARCH held: turn first"

        lock = " [LOCKED]" if signal.locked else ""
        print(f"{signal.state.upper():<11} c{signal.confidence:4.2f} | "
              f"{signal.pan_px:+7.1f}px {pan:<10} | "
              f"{signal.tilt_px:+7.1f}px {tilt:<9}"
              f"{turn_txt} | "
              f"{zoom_pct:+5.1f}% {zoom_order} | "
              f"roll {roll_deg:+6.2f} {roll_order} | "
              f"pitch {pitch_deg:+6.2f} {pitch_order}{lock}{held}")
        return True


def _draw_guidance_arrow(
    img: np.ndarray,
    cx: int, cy: int,
    pan_px: float, tilt_px: float,
    scale: float, max_len: int,
) -> None:
    dx = pan_px  * scale
    dy = tilt_px * scale

    mag = math.hypot(dx, dy)
    if mag < 1.0:
        return

    if mag > max_len:
        dx = dx / mag * max_len
        dy = dy / mag * max_len

    ex = int(cx + dx)
    ey = int(cy + dy)

    cv2.arrowedLine(img, (cx, cy), (ex, ey), (0, 0, 0),      6, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(img, (cx, cy), (ex, ey), _COL["arrow"],  3, cv2.LINE_AA, tipLength=0.25)

    label = f"{math.hypot(pan_px, tilt_px):.0f}px"
    ox = 8 if dx >= 0 else -60
    oy = -10
    _put_text_shadowed(img, label, (ex + ox, ey + oy), scale=0.45, thickness=1)


def _draw_locked_banner(img: np.ndarray, h: int, w: int) -> None:
    text = "LOCKED"
    font_scale = 1.4
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, _FONTB, font_scale, thickness)
    tx = (w - tw) // 2
    ty = 70

    overlay = img.copy()
    pad = 12
    cv2.rectangle(overlay, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad),
                  _COL["locked"], -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.putText(img, text, (tx, ty), _FONTB, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_state_badge(img: np.ndarray, signal: GuidanceSignal, col) -> None:
    state_label = signal.state.upper().replace("_", " ")
    text = f"  {state_label}"
    font_scale = 0.6
    thickness = 1

    (tw, th), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad = 6
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2), col, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (10 + pad, 10 + th + pad // 2), _FONT, font_scale,
                _COL["text_light"], thickness, cv2.LINE_AA)


def _draw_confidence(img: np.ndarray, w: int, signal: GuidanceSignal) -> None:
    text = f"conf {signal.confidence * 100:.0f}%"
    font_scale = 0.5
    thickness = 1
    (tw, _), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x = w - tw - 14
    _put_text_shadowed(img, text, (x, 28), scale=font_scale, thickness=thickness)


def _draw_leveling(img: np.ndarray, h: int, signal: GuidanceSignal) -> None:
    roll_deg  = math.degrees(signal.roll_err)
    pitch_deg = math.degrees(signal.pitch_err)
    lines = [
        f"roll  {roll_deg:+.1f} deg",
        f"pitch {pitch_deg:+.1f} deg",
    ]
    y0 = h - 14 - 22 * (len(lines) - 1) - 8
    for i, line in enumerate(lines):
        _put_text_shadowed(img, line, (14, y0 + i * 22), scale=0.47, thickness=1)


def _draw_zoom(img: np.ndarray, w: int, h: int, signal: GuidanceSignal) -> None:
    z = signal.zoom_rel
    if abs(z - 1.0) < 0.02:
        label = "ZOOM OK"
        col = _COL["locked"]
    elif z > 1.0:
        label = f"ZOOM IN  x{z:.2f}"
        col = (0, 200, 255)
    else:
        label = f"ZOOM OUT x{z:.2f}"
        col = (0, 200, 255)

    font_scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, _FONT, font_scale, thickness)
    tx = (w - tw) // 2
    ty = h - 14

    _put_text_shadowed(img, label, (tx, ty), scale=font_scale, thickness=thickness,
                       fg=col)


def _put_text_shadowed(
    img: np.ndarray,
    text: str,
    org: tuple,
    scale: float = 0.5,
    thickness: int = 1,
    fg=(240, 240, 240),
) -> None:
    ox, oy = org
    cv2.putText(img, text, (ox + 1, oy + 1), _FONT, scale, (0, 0, 0),
                thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (ox, oy), _FONT, scale, fg, thickness, cv2.LINE_AA)
