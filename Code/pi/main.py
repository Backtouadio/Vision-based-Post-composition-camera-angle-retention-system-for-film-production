#!/usr/bin/env python3
# Entry point on the Pi. Full pipeline with the KLT + PnP fine stage.
#   python3 main.py --setup --headless --telemetria 192.168.100.166:5005
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import cv2

from nucleo import Config, GuidanceSignal
from camara import LiveCamera, build_reference_model, InsufficientTextureError
from vision import KLTAlignmentEngine
from guia import CoherenciaYaw, GuidanceMapper, SuavizadorGuia
from imu import OrientationSource
from ui_local import HeadlessUI, OperatorUI
from red import EmisorTelemetria, ModoSetup, RunLogger, ServidorPi


_ROTATE_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _imu_sidecar_path(ref_path: Path) -> Path:
    return ref_path.with_suffix(ref_path.suffix + ".imu.json")


def _save_imu_sidecar(ref_path: Path, roll: float, pitch: float) -> None:
    path = _imu_sidecar_path(ref_path)
    path.write_text(json.dumps({"ref_roll": roll, "ref_pitch": pitch}, indent=2))


def _load_imu_sidecar(ref_path: Path, warn: bool = True) -> tuple[float, float]:
    path = _imu_sidecar_path(ref_path)
    if not path.exists():
        if warn:
            print(f"WARNING: no tilt sidecar next to {ref_path.name} "
                  f"({path.name} missing).")
            print("         Leveling arrows will aim at the sensor's own zero, "
                  "NOT at this photo's tilt.")
            print("         Re-take the reference with --capture-ref, or run "
                  "with --no-imu to hide the leveling arrows.")
        return 0.0, 0.0
    data = json.loads(path.read_text())
    return float(data.get("ref_roll", 0.0)), float(data.get("ref_pitch", 0.0))


def capture_reference(ref_path: Path, cfg: Config, use_imu: bool) -> None:
    orient = None
    if use_imu:
        try:
            orient = OrientationSource(cfg)
            print(f"Opening IMU. HOLD THE RIG STILL for ~{cfg.gyro_bias_seconds:g}s "
                  f"(gyro-bias calibration) ...")
            orient.open()
            for _ in range(max(1, int(cfg.imu_odr_hz * 0.5))):
                orient.orientation()
                time.sleep(1.0 / cfg.imu_odr_hz)
            print("IMU ready.")
        except RuntimeError as exc:
            print(f"IMU not available, capturing image only: {exc}")
            orient = None

    print()
    print("=" * 62)
    print("  HOLD THE SHOT NOW -- keep holding until you see 'done'.")
    print("  The image AND the tilt are both taken in this pose.")
    print("=" * 62)
    for n in (3, 2, 1):
        print(f"  {n} ...")
        time.sleep(1.0)

    roll_before = pitch_before = None
    if orient is not None:
        roll_before, pitch_before = orient.orientation()

    image = None
    with LiveCamera(cfg) as cam:
        for i, frame in enumerate(cam.frames()):
            if i < cfg.capture_warmup_frames:
                continue
            image = frame.image
            break

    roll = pitch = None
    if orient is not None:
        roll, pitch = orient.orientation()
        moved = max(abs(math.degrees(roll - roll_before)),
                    abs(math.degrees(pitch - pitch_before)))
        orient.close()
        if moved > cfg.capture_move_tolerance_deg:
            print()
            print(f"REJECTED: the rig moved {moved:.2f} deg during capture "
                  f"(limit {cfg.capture_move_tolerance_deg:.2f} deg).")
            print("Nothing was saved. Brace the camera and run --capture-ref again.")
            return
        print(f"  (rig held to within {moved:.2f} deg through the capture)")

    if image is None:
        print("ERROR: camera produced no frame; nothing saved.")
        return
    cv2.imwrite(str(ref_path), image)

    if roll is not None:
        _save_imu_sidecar(ref_path, roll, pitch)
        print(f"done -- image + tilt saved (roll {math.degrees(roll):+.2f} deg, "
              f"pitch {math.degrees(pitch):+.2f} deg -> "
              f"{_imu_sidecar_path(ref_path).name})")
    else:
        print("done -- image saved (no tilt sidecar: IMU unavailable or --no-imu).")


def run(args: argparse.Namespace, cfg_overrides: dict | None = None,
        sink=None, modo=None) -> None:
    cfg = Config.from_yaml(args.config) if args.config else Config()

    for _k, _v in (cfg_overrides or {}).items():
        if not hasattr(cfg, _k):
            raise SystemExit(f"ERROR: unknown config override: {_k}")
        setattr(cfg, _k, _v)

    ref_path = Path(args.reference) if args.reference else None

    if args.capture_ref:
        capture_reference(ref_path, cfg, use_imu=not args.no_imu)
        return

    hay_ref = ref_path is not None and ref_path.exists()
    if not hay_ref and modo is None:
        print(f"ERROR: reference image not found: {ref_path}")
        sys.exit(1)

    rotate = _ROTATE_MAP.get(args.rotate) if args.rotate else None

    print("Opening camera ...")
    cam = LiveCamera(cfg)
    cam.open()

    first = next(cam.frames())
    h, w  = first.image.shape[:2]
    print(f"Camera resolution: {w}×{h}")

    orient = None
    if not args.no_imu:
        orient = OrientationSource(cfg)
        try:
            print(f"Calibrating IMU (hold still ~{cfg.gyro_bias_seconds:g}s) ...")
            orient.open()
            bias_dps = [math.degrees(b) for b in orient.gyro_bias]
            print(f"IMU ready. Gyro bias (deg/s): "
                  f"[{bias_dps[0]:+.3f}, {bias_dps[1]:+.3f}, {bias_dps[2]:+.3f}]")
        except RuntimeError as exc:
            print(f"IMU unavailable, leveling channel disabled: {exc}")
            orient = None

    def _cargar_referencia(path: Path):
        rr, rp = _load_imu_sidecar(path)
        if orient is not None and (rr or rp):
            print(f"Reference tilt loaded: roll={math.degrees(rr):+.2f}°, "
                  f"pitch={math.degrees(rp):+.2f}°")
        print(f"Building reference model ({path.name}) ...")
        rm = build_reference_model(image_path=path, target_size=(w, h),
                                   config=cfg, ref_roll=rr, ref_pitch=rp)
        return rm, KLTAlignmentEngine(rm, cfg)

    def _rechazar(nombre: str, motivo: str) -> None:
        print(f"[main] reference rejected ({nombre}): {motivo}", flush=True)
        rechaza = getattr(modo, "rechazar_referencia", None) if modo else None
        if callable(rechaza):
            rechaza(f"{nombre}: {motivo}")

    ref_model = engine = None
    if hay_ref:
        try:
            ref_model, engine = _cargar_referencia(ref_path)
        except InsufficientTextureError as exc:
            if modo is None:
                sys.exit(f"ERROR: reference {ref_path.name} is unusable. {exc}")
            _rechazar(ref_path.name,
                      f"insufficient descriptors ({exc.n_keypoints}, "
                      f"minimum {exc.minimum}). Pick another shot.")
            ref_model = engine = None
    suav = None
    if getattr(cfg, "suavizado_enabled", False):
        suav = SuavizadorGuia(cfg)
    mapper = GuidanceMapper(cfg, suav)

    coh = None
    if getattr(cfg, "yaw_coherencia", True) and getattr(cfg, "essential_yaw", False):
        coh = CoherenciaYaw(cfg.yaw_salto_max_deg, cfg.yaw_confirmaciones,
                            cfg.yaw_caducidad_s)
    logger = RunLogger(cfg, meta={"reference": str(ref_path), "resolution": [w, h],
                                  "imu": orient is not None})

    if args.headless:
        print("Starting guidance loop (headless). Ctrl-C to stop.")
    else:
        print("Starting guidance loop. Press 'q' to quit.")

    try:
        ui_ctx = (HeadlessUI(cfg) if args.headless
                  else OperatorUI(cfg, rotate=rotate))
        with ui_ctx as ui:
            for frame in _prepend(first, cam.frames()):
                t0 = time.perf_counter()

                roll = pitch = None
                imu_sample = None
                if orient is not None:
                    roll, pitch = orient.orientation()
                    imu_sample = orient.last_sample

                if modo is not None:
                    nueva = modo.tick(frame, roll, pitch)
                    if nueva is not None:
                        try:
                            ref_model, engine = _cargar_referencia(Path(nueva))
                        except InsufficientTextureError as exc:
                            _rechazar(Path(nueva).name,
                                      f"insufficient descriptors "
                                      f"({exc.n_keypoints}, minimum "
                                      f"{exc.minimum}). Pick a shot with "
                                      f"more texture.")
                        except Exception as exc:
                            traceback.print_exc()
                            _rechazar(Path(nueva).name,
                                      f"could not be loaded "
                                      f"({type(exc).__name__}). Pick another shot.")

                if modo is not None and modo.en_setup():
                    if sink is not None:
                        sink.push(GuidanceSignal(state=modo.etiqueta()),
                                  frame=frame, nota=modo.aviso())
                    continue
                if engine is None:
                    continue

                result = engine.update(frame, imu_sample=imu_sample)

                _yaw_raw = getattr(
                    getattr(result, "estimator_result", None), "yaw_deg", None)
                _yaw_ok = (coh.filtra(frame.timestamp, _yaw_raw)
                           if coh is not None else _yaw_raw)

                signal = mapper.map(
                    transform=result.transform,
                    confidence=result.confidence,
                    state=result.state.name.lower(),
                    imu_sample=imu_sample,
                    ref_model=ref_model,
                    roll=roll,
                    pitch=pitch,
                    yaw_deg=_yaw_ok,
                    t=frame.timestamp,
                    sel_gana=getattr(
                        getattr(result, "estimator_result", None), "sel_gana", None),
                )
                proc_ms = (time.perf_counter() - t0) * 1000.0
                logger.log(frame, result, signal, imu_sample=imu_sample,
                           proc_ms=proc_ms,
                           extra={"pitch_corr_px": mapper.last_pitch_correction_px,
                                  "yaw_corr_px": mapper.last_yaw_correction_px,
                                  "pan_px_raw": mapper.last_raw_pan_px,
                                  "tilt_px_raw": mapper.last_raw_tilt_px,
                                  "zoom_rel_raw": mapper.last_raw_zoom_rel,
                                  "turn_deg_raw": mapper.last_raw_turn_deg,
                                  "yaw_raw_deg": _yaw_raw,
                                  "yaw_rechazado": (_yaw_raw is not None
                                                    and _yaw_ok is None)})

                if sink is not None:
                    sink.push(signal, result=result, frame=frame, roll=roll,
                              pitch=pitch, ref_model=ref_model,
                              proc_ms=proc_ms, yaw_medido=_yaw_ok is not None)

                if not ui.show(frame.image, signal):
                    break
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        logger.close()
        if sink is not None:
            sink.close()
        if orient is not None:
            orient.close()
        cam.close()
        print("Camera closed.")


def _prepend(first_frame, rest):
    yield first_frame
    yield from rest




def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real pipeline with the KLT + PnP fine stage instead of "
                    "phase correlation.")
    p.add_argument("reference", nargs="?", default=None,
                   help="reference captured with main.py --capture-ref. "
                        "Optional with --setup: chosen from the browser.")
    p.add_argument("--config", default=None, metavar="PATH")
    p.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270])
    p.add_argument("--headless", action="store_true",
                   help="required over SSH")
    p.add_argument("--no-imu", action="store_true")
    p.add_argument("--no-gric-gate", action="store_true",
                   help="seed the cloud even if GRIC did not pick E (or "
                        "GRIC is off). For debugging, not for guiding.")
    p.add_argument("--no-gric", action="store_true",
                   help="turns GRIC off entirely. Implies --no-gric-gate.")
    p.add_argument("--no-turn", action="store_true",
                   help="LOOK BUT DON'T TOUCH the PnP yaw: it is still "
                        "computed and logged (turn_deg in the log/UI), but it "
                        "is not subtracted from pan_px nor does it gate "
                        "LOCKED. Isolates whether the KLT homography converges "
                        "on its own. Start here if PnP diverges from the IMU.")
    p.add_argument("--telemetria", nargs="?", const="127.0.0.1:5005",
                   default=None, metavar="HOST[:PORT]",
                   help="emits the operator packet over UDP "
                        "(pc/puente.py --live).")
    p.add_argument("--telemetria-log", default=None, metavar="PATH",
                   help="also writes the SAME packet to JSONL; "
                        "replayable with --replay.")
    p.add_argument("--setup", action="store_true",
                   help="starts in SETUP mode with the command server: "
                        "capture, name and pick the shot from the browser.")
    p.add_argument("--refs-dir", default="Data/refs", metavar="DIR",
                   help="where the shots live (default Data/refs).")
    p.add_argument("--puerto-cmd", type=int, default=8081, metavar="PORT")
    p.add_argument("--capture-ref", action="store_true",
                   help="captures reference + tilt and exits.")
    args = p.parse_args()
    if not args.reference and not args.setup:
        p.error("a reference is needed, or --setup to pick one in the browser")
    return args


def main() -> None:
    args = _parse_args()

    overrides = {
        "essential_yaw": True,
        "gric_enabled": not args.no_gric,
        "klt_require_gric_e": not (args.no_gric_gate or args.no_gric),
        "lock_on_requires_yaw": True,
        "yaw_compensation": not args.no_turn,
        "yaw_gates_lock": not args.no_turn,
    }

    cfg = Config.from_yaml(args.config) if args.config else Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)

    print("=" * 70)
    print("  KLT+PnP GUIDANCE -- fine stage: frozen 3D cloud, tracked with KLT")
    print("=" * 70)
    print(f"  reference    : {args.reference}")
    print(f"  GRIC         : {'YES' if cfg.gric_enabled else 'NO'}")
    print(f"  requires E   : {'YES' if cfg.klt_require_gric_e else 'NO (debug)'}")
    print(f"  K            : fx {cfg.fx:.2f} fy {cfg.fy:.2f} "
          f"cx {cfg.cx:.2f} cy {cfg.cy:.2f}")
    print(f"  seeding      : >= {cfg.klt_seed_min_points} inliers from E, "
          f">= {cfg.klt_min_points} live points to avoid recovery")
    if args.no_turn:
        print("  --no-turn: PnP is computed and logged but does NOT touch "
              "pan_px or LOCKED.")
    print()
    print("  First check: ess_pitch_deg/ess_roll_deg from the log against")
    print("  pitch_err_deg/roll_err_deg from the IMU on MICROMETRY frames --")
    print("  if they diverge, this estimator's turn_deg is not reliable yet.")
    print("=" * 70)
    print()

    sink = None
    if args.telemetria or args.telemetria_log:
        host, port = None, 5005
        if args.telemetria:
            host, _, _p = args.telemetria.partition(":")
            port = int(_p) if _p else 5005
        sink = EmisorTelemetria(cfg, host=host, port=port,
                                log_path=args.telemetria_log,
                                zoom_medido=True, ref_path=args.reference)
        print(f"  telemetry    : {'udp %s:%d' % (host, port) if host else '-'}"
              f"{' + ' + args.telemetria_log if args.telemetria_log else ''}")

    modo = servidor = None
    if args.setup:
        servidor = ServidorPi(args.refs_dir, args.puerto_cmd)
        modo = ModoSetup(cfg, servidor, args.refs_dir,
                         ref_inicial=args.reference)
        servidor.en_uso = lambda: modo.ref_actual
        servidor.start()

    try:
        run(args, cfg_overrides=overrides, sink=sink, modo=modo)
    finally:
        if servidor is not None:
            servidor.close()


if __name__ == "__main__":
    main()
