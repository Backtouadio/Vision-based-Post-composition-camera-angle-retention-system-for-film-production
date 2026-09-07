#!/usr/bin/env python3
"""
Temperature/throttling sampler, independent from the pipeline.
It is launched in ANOTHER SSH session while main.py runs in the first one.
It touches nothing in the project -- it only calls vcgencmd and writes a CSV.

IMPORTANT: the "has occurred" bits (16-19) of vcgencmd are only cleared
by a reboot. Reboot the Pi before EVERY thermal run, or a previous run
will contaminate this one's reading.

Usage (on the Pi, in a second SSH terminal, while main.py runs in the first):
    python3 medir_temperatura.py --out temp_search.csv
    python3 medir_temperatura.py --out temp_micrometry.csv

SEARCH-only run: in main.py, use a --config with a very high
inlier_ratio_threshold (e.g. 0.99) so that it never locks onto LOCK_ON and
stays in SEARCH for the whole run.
MICROMETRY-only run: let it lock on normally and then do NOT move the
camera -- it stays in MICROMETRY for the rest of the run by construction.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

RE_TEMP = re.compile(r"temp=([\d.]+)")
RE_THR = re.compile(r"throttled=(0x[0-9a-fA-F]+)")


def leer_temp() -> float | None:
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        m = RE_TEMP.search(out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def leer_throttled() -> int | None:
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
        m = RE_THR.search(out)
        return int(m.group(1), 16) if m else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, metavar="CSV")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--max-duration", type=float, default=1200,
                     help="safety cap in seconds (default 20 min)")
    ap.add_argument("--tail-after", type=float, default=60,
                     help="extra seconds to record after the first throttle")
    args = ap.parse_args()

    t0 = leer_temp()
    if t0 is None:
        raise SystemExit("could not read vcgencmd -- are you on the Pi?")

    thr0 = leer_throttled()
    if thr0:
        print(f"WARNING: throttled=0x{thr0:x} already has bits set BEFORE starting. "
              f"Reboot the Pi (sudo reboot) and launch this again from clean.")

    out = Path(args.out)
    started = datetime.now().isoformat(timespec="seconds")
    inicio = time.monotonic()
    onset_t = None

    with open(out, "w", buffering=1) as f:
        f.write(f"# started_iso={started}\n")
        f.write("elapsed_s,temp_c,throttled_hex,undervolt_now,freqcap_now,"
                "throttled_now,softlimit_now\n")
        print(f"[temp] recording to {out} every {args.interval}s. Ctrl-C to stop.")
        while True:
            elapsed = time.monotonic() - inicio
            temp = leer_temp()
            thr = leer_throttled() or 0
            uv_now = bool(thr & 0x1)
            fc_now = bool(thr & 0x2)
            th_now = bool(thr & 0x4)
            sl_now = bool(thr & 0x8)

            f.write(f"{elapsed:.2f},{temp if temp is not None else ''},"
                    f"0x{thr:x},{int(uv_now)},{int(fc_now)},{int(th_now)},{int(sl_now)}\n")
            print(f"  t={elapsed:6.1f}s  temp={temp:.1f}C  throttled=0x{thr:x}"
                  f"{'  <-- THROTTLING' if (th_now or sl_now) else ''}")

            if (th_now or sl_now) and onset_t is None:
                onset_t = elapsed
                print(f"[temp] *** throttling detected at {elapsed:.1f}s, "
                      f"{temp:.1f}C. Recording {args.tail_after}s more and stopping. ***")

            if onset_t is not None and elapsed - onset_t >= args.tail_after:
                break
            if elapsed >= args.max_duration:
                print("[temp] safety cap reached without throttling.")
                break
            time.sleep(args.interval)

    print(f"[temp] done -> {out}")


if __name__ == "__main__":
    main()
