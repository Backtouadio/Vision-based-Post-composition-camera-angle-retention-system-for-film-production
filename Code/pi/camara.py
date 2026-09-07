# Input: locked IMX219 camera (Picamera2) and construction of the reference model.
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from nucleo import Config, Frame, FrameSource, ReferenceModel


_CONTROL_SETTLE_FRAMES = 8

_VERIFY_SAMPLES = 5


class LiveCamera(FrameSource):

    def __init__(self, config: Config) -> None:
        self._cfg          = config
        self._picam2       = None
        self._frame_index  = 0
        self.unsupported_controls: list = []


    @property
    def picam2(self):
        return self._picam2


    def _select_sensor_mode(self) -> dict:
        cfg   = self._cfg
        want  = tuple(cfg.sensor_mode_size)
        depth = cfg.sensor_mode_bit_depth
        modes = self._picam2.sensor_modes

        for mode in modes:
            if tuple(mode["size"]) == want and mode.get("bit_depth") == depth:
                return mode

        available = ", ".join(
            f'{tuple(m["size"])}@{m.get("bit_depth")}bit' for m in modes
        )
        raise RuntimeError(
            f"No sensor mode {want} at {depth}-bit. Available: {available}"
        )

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is not available on this machine. "
                "Use ClipReader for laptop development."
            ) from exc

        cfg = self._cfg
        self._picam2 = Picamera2()

        raw_mode = self._select_sensor_mode()

        camera_config = self._picam2.create_video_configuration(
            main={"size": cfg.lores_resolution, "format": "BGR888"},
            raw=raw_mode,
            controls={
                "FrameDurationLimits": (
                    cfg.frame_duration_us,
                    cfg.frame_duration_us,
                ),
            },
        )
        self._picam2.configure(camera_config)

        actual = self._picam2.camera_configuration()["raw"]["size"]
        if tuple(actual) != tuple(cfg.sensor_mode_size):
            raise RuntimeError(
                f"Sensor mode mismatch: asked for {tuple(cfg.sensor_mode_size)}, "
                f"got {tuple(actual)}. Reference and live capture MUST use the "
                f"same sensor mode or the fields of view will not match."
            )
        self._picam2.start()

        wanted = {
            "AeEnable":     False,
            "ExposureTime": cfg.exposure_us,
            "AnalogueGain": cfg.analogue_gain,
            "AwbEnable":    False,
            "ColourGains":  tuple(cfg.colour_gains),
            "AfMode":       0,
            "LensPosition": cfg.lens_position,
        }

        available = set(self._picam2.camera_controls)
        controls  = {k: v for k, v in wanted.items() if k in available}
        skipped   = sorted(set(wanted) - available)

        self._picam2.set_controls(controls)

        if skipped:
            self.unsupported_controls = skipped
            print(f"[LiveCamera] sensor does not advertise: {', '.join(skipped)} "
                  f"-- not locked (fixed-focus module is normal for IMX219)")

        for _ in range(_CONTROL_SETTLE_FRAMES):
            self._picam2.capture_array("main")

        meta = self._verify_lock(cfg)
        if meta["bad"]:
            self._picam2.set_controls(controls)
            for _ in range(_CONTROL_SETTLE_FRAMES):
                self._picam2.capture_array("main")
            meta = self._verify_lock(cfg)

        if meta["bad"]:
            print("[LiveCamera] WARNING: the radiometric lock did NOT take, "
                  "after two attempts:")
            for line in meta["bad"]:
                print(f"[LiveCamera]   {line}")
            print(f"[LiveCamera]   over {len(meta['traj'])} frames, "
                  f"gain went {' -> '.join(f'{g:.3f}' for g in meta['traj'])}")
            spread = max(meta["traj"]) - min(meta["traj"]) if meta["traj"] else 0.0
            if spread < 0.01:
                print("[LiveCamera]   ...but it is STABLE. A constant wrong value "
                      "is survivable:")
                print("[LiveCamera]   reference and live share it, so matching "
                      "still works. Note it and move on.")
            else:
                print(f"[LiveCamera]   ...and it is STILL MOVING (spread "
                      f"{spread:.3f}). Do not trust this run.")
            dg = meta.get("digital")
            if dg is not None and meta.get("analogue"):
                total = meta["analogue"] * dg
                print(f"[LiveCamera]   analogue {meta['analogue']:.3f} x digital "
                      f"{dg:.3f} = {total:.3f} total "
                      f"(asked {float(cfg.analogue_gain):.3f})")
                if abs(total - float(cfg.analogue_gain)) < 0.05:
                    print("[LiveCamera]   -> the ISP simply SPLIT the gain "
                          "differently. Total is right; not a fault.")
        else:
            print(f"[LiveCamera] locked: exposure {meta['exposure']} us, "
                  f"gain {meta['analogue']:.3f}")

        lux = meta.get("lux")
        if lux is not None:
            verdict = ("DIM -- at or below the measured floor" if lux < 100 else
                       "workable" if lux >= 220 else
                       "between the measured 84 and 234 lux, un-bracketed")
            print(f"[LiveCamera] scene: {lux:.0f} lux  ({verdict})")

    def _verify_lock(self, cfg) -> dict:
        traj = []
        meta = {}
        for _ in range(_VERIFY_SAMPLES):
            meta = self._picam2.capture_metadata()
            g = meta.get("AnalogueGain")
            if g is not None:
                traj.append(float(g))

        exp = meta.get("ExposureTime")
        gain = meta.get("AnalogueGain")
        want_gain = float(cfg.analogue_gain)

        bad = []
        if exp is not None and abs(exp - cfg.exposure_us) > max(200, 0.05 * cfg.exposure_us):
            bad.append(f"ExposureTime: asked {cfg.exposure_us} us, got {exp} us")
        if gain is not None and abs(gain - want_gain) > 0.05:
            bad.append(f"AnalogueGain: asked {want_gain:g}, got {gain:.6f}")

        return {
            "exposure": exp,
            "analogue": float(gain) if gain is not None else None,
            "digital": float(meta["DigitalGain"]) if meta.get("DigitalGain") else None,
            "lux": meta.get("Lux"),
            "traj": traj,
            "bad": bad,
            "meta": meta,
        }

    def close(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2 = None

    def frames(self) -> Iterator[Frame]:
        if self._picam2 is None:
            raise RuntimeError("Call open() before iterating frames.")
        while True:
            bgr = self._picam2.capture_array("main")
            yield Frame(
                image=bgr,
                timestamp=time.monotonic(),
                index=self._frame_index,
            )
            self._frame_index += 1


class InsufficientTextureError(RuntimeError):

    def __init__(self, n_keypoints: int, minimum: int, source: str = ""):
        self.n_keypoints = int(n_keypoints)
        self.minimum = int(minimum)
        self.source = source
        where = f" in {source}" if source else ""
        super().__init__(
            f"Insufficient number of descriptors{where}: ORB found "
            f"{self.n_keypoints}, minimum is {self.minimum}. The scene is "
            f"probably too low-texture, too dark or blurry."
        )


def build_reference_model(
    image_path: str | Path,
    target_size: Tuple[int, int],
    config: Config,
    ref_roll: float = 0.0,
    ref_pitch: float = 0.0,
) -> ReferenceModel:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read: {image_path}")

    src_h, src_w = bgr.shape[:2]
    tgt_w, tgt_h = target_size

    src_ratio = src_w / src_h
    tgt_ratio = tgt_w / tgt_h

    if abs(src_ratio - tgt_ratio) > 0.02:
        if src_ratio > tgt_ratio:
            new_w = int(src_h * tgt_ratio)
            x0 = (src_w - new_w) // 2
            bgr = bgr[:, x0:x0 + new_w]
        else:
            new_h = int(src_w / tgt_ratio)
            y0 = (src_h - new_h) // 2
            bgr = bgr[y0:y0 + new_h, :]

        crop_h, crop_w = bgr.shape[:2]
        print(
            f"[reference_model] Cropped reference from {src_w}x{src_h} "
            f"to {crop_w}x{crop_h} to match target aspect ratio {tgt_ratio:.3f}."
        )

    interpolation = cv2.INTER_AREA if (bgr.shape[1] > tgt_w or bgr.shape[0] > tgt_h) else cv2.INTER_LINEAR
    resized_bgr = cv2.resize(bgr, (tgt_w, tgt_h), interpolation=interpolation)

    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    return build_reference_model_from_gray(
        gray,
        config,
        ref_roll=ref_roll,
        ref_pitch=ref_pitch,
        _src_desc=f"{image_path.name} ({src_w}x{src_h} -> {tgt_w}x{tgt_h})",
    )


def build_reference_model_from_gray(
    gray: np.ndarray,
    config: Config,
    ref_roll: float = 0.0,
    ref_pitch: float = 0.0,
    _src_desc: str = "",
) -> ReferenceModel:
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    tgt_h, tgt_w = gray.shape[:2]

    pyramid: list[np.ndarray] = [gray]
    for _ in range(config.pyramid_levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))

    orb = cv2.ORB_create(nfeatures=config.orb_n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    n_kp = 0 if descriptors is None else len(keypoints)
    minimo = int(getattr(config, "ref_min_keypoints", 50))
    if n_kp < minimo:
        raise InsufficientTextureError(n_kp, minimo, _src_desc)

    S = config.phase_crop_size
    fft_ref = None
    if tgt_h >= S and tgt_w >= S:
        cy, cx = tgt_h // 2, tgt_w // 2
        crop = gray[cy - S//2 : cy + S//2, cx - S//2 : cx + S//2].astype(np.float32)
        window = np.outer(np.hanning(S), np.hanning(S)).astype(np.float32)
        fft_ref = cv2.dft(crop * window, flags=cv2.DFT_COMPLEX_OUTPUT)

    print(
        f"[reference_model] Built {('from ' + _src_desc) if _src_desc else 'from array'}, "
        f"{len(keypoints)} keypoints, {config.pyramid_levels}-level pyramid."
    )

    return ReferenceModel(
        image_gray=gray,
        pyramid=pyramid,
        keypoints=list(keypoints),
        descriptors=descriptors,
        ref_roll=ref_roll,
        ref_pitch=ref_pitch,
        fft_ref=fft_ref,
    )
