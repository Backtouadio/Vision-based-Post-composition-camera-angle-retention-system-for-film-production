#!/usr/bin/env python3
"""
Reads the two CSVs from medir_temperatura.py (one for SEARCH, one for MICROMETRY)
and plots a chart comparing the temperature rise up to throttling.

Since each run is isolated so that ~all the time is spent in a single state (SEARCH
forced with thresholds impossible to lock onto / MICROMETRY staying still
after the lock), the duration of each CSV IS the time in that state -- there is no
need to cross-reference it with main.py's telemetry log.

Usage (on the PC):
    python graficar_termico.py --search temp_search.csv --micro temp_micrometry.csv --out termico.png
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cargar(path):
    t, temp, thr_now = [], [], []
    with open(path, newline="") as f:
        for row in f:
            if row.startswith("#") or row.startswith("elapsed_s"):
                continue
            r = row.strip().split(",")
            if len(r) < 7 or r[1] == "":
                continue
            t.append(float(r[0]))
            temp.append(float(r[1]))
            thr_now.append(bool(int(r[5])) or bool(int(r[6])))
    return np.array(t), np.array(temp), np.array(thr_now)


def onset(t, thr_now):
    idx = np.argmax(thr_now) if thr_now.any() else None
    return float(t[idx]) if idx is not None else None


def pendiente_c_por_min(t, temp, hasta):
    mask = t <= (hasta if hasta is not None else t[-1])
    if mask.sum() < 2:
        return float("nan")
    p = np.polyfit(t[mask], temp[mask], 1)
    return p[0] * 60.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True, metavar="CSV")
    ap.add_argument("--micro", required=True, metavar="CSV")
    ap.add_argument("--out", default="termico.png")
    args = ap.parse_args()

    t_s, temp_s, thr_s = cargar(args.search)
    t_m, temp_m, thr_m = cargar(args.micro)
    onset_s, onset_m = onset(t_s, thr_s), onset(t_m, thr_m)
    slope_s = pendiente_c_por_min(t_s, temp_s, onset_s)
    slope_m = pendiente_c_por_min(t_m, temp_m, onset_m)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t_s, temp_s, color="#d1495b", lw=1.8,
            label=f"SEARCH  ({slope_s:.2f} °C/min until throttle)")
    ax.plot(t_m, temp_m, color="#2e86ab", lw=1.8,
            label=f"MICROMETRY  ({slope_m:.2f} °C/min until throttle)")

    for x, c, nombre in [(onset_s, "#d1495b", "SEARCH"), (onset_m, "#2e86ab", "MICROMETRY")]:
        if x is not None:
            ax.axvline(x, color=c, ls="--", lw=1, alpha=0.7)
            ax.annotate(f"throttle {nombre}\nt={x:.0f}s", xy=(x, ax.get_ylim()[1]),
                        xytext=(4, -4), textcoords="offset points",
                        fontsize=8, color=c, va="top")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("CM4 temperature (°C)")
    ax.set_title("Thermal rise up to throttling: SEARCH vs MICROMETRY")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")
    print(f"SEARCH:     onset={onset_s} s   slope={slope_s:.2f} °C/min   dur={t_s[-1]:.0f}s")
    print(f"MICROMETRY: onset={onset_m} s   slope={slope_m:.2f} °C/min   dur={t_m[-1]:.0f}s")


if __name__ == "__main__":
    main()
