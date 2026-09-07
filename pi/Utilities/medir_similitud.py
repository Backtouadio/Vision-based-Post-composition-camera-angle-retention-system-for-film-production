#!/usr/bin/env python3
"""
Compares every photo of a measurement session against the reference:
SSIM (structural similarity), pitch/roll (deg) and their error against the
reference. It only uses cv2 + numpy (already dependencies of the project).

Usage (on the Pi, inside FINAL/pi):
    python3 medir_similitud.py --ref Data/refs/NOMBRE_REFERENCIA.png
    python3 medir_similitud.py --ref Data/refs/REF.png --refs-dir Data/refs --out resultados.csv

If your reference has no .imu.json sidecar, pitch/roll_error come out empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

# Session table (group, try, time_s, photo). Edit here if the data changes.
TABLA = [
    ("WITHOUT device (memory)",                1, 21.37, "S1_A_2"),
    ("WITHOUT device (memory)",                2, 18.43, "S1_A_3"),
    ("WITHOUT device (memory)",                3, 23.47, "S1_A_4"),
    ("WITHOUT device (memory)",                4, 23.79, "S1_A_5"),
    ("WITHOUT device (memory)",                5, 23.71, "S1_A_6"),
    ("WITHOUT device (memory)",                6, 22.21, "S1_A_7"),
    ("WITH device, LARGE baseline (Search)",          1, 33.70, "S1_B_1"),
    ("WITH device, LARGE baseline (Search)",          2, 11.40, "S1_B_2"),
    ("WITH device, LARGE baseline (Search)",          3, 18.20, "S1_B_3"),
    ("WITH device, LARGE baseline (Search)",          4, 15.23, "S1_B_4"),
    ("WITH device, LARGE baseline (Search)",          5, 25.54, "S1_B_5"),
    ("WITH device, LARGE baseline (Search)",          6, 28.10, "S1_B_6"),
    ("WITH device, LARGE baseline (Search)",          7, 16.80, "S1_B_7"),
    ("WITH device, SMALL baseline (Micrometry)",       1, 17.49, "S1_C_1"),
    ("WITH device, SMALL baseline (Micrometry)",       2, 15.45, "S1_C_2"),
    ("WITH device, SMALL baseline (Micrometry)",       3, 12.28, "S1_C_3"),
    ("WITH device, SMALL baseline (Micrometry)",       4,  9.63, "S1_C_4"),
    ("WITH device, SMALL baseline (Micrometry)",       5, 15.30, "S1_C_5"),
]


def ssim(img1: np.ndarray, img2: np.ndarray, L: int = 255) -> float:
    """Single-scale SSIM (Wang et al.), 11x11 gaussian window."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.T)

    def filt(x):
        return cv2.filter2D(x, -1, win)[5:-5, 5:-5]

    mu1, mu2 = filt(img1), filt(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = filt(img1 * img1) - mu1_sq
    sigma2_sq = filt(img2 * img2) - mu2_sq
    sigma12 = filt(img1 * img2) - mu1_mu2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((num / den).mean())


def cargar_gris(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def cargar_imu(path_imagen: Path) -> tuple[float | None, float | None]:
    sidecar = path_imagen.with_suffix(path_imagen.suffix + ".imu.json")
    if not sidecar.is_file():
        return None, None
    d = json.loads(sidecar.read_text())
    return d.get("ref_roll"), d.get("ref_pitch")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, metavar="PATH",
                     help="reference photo (the one everything is compared against)")
    ap.add_argument("--refs-dir", default="Data/refs", metavar="DIR",
                     help="folder where the photos of the table live (default Data/refs)")
    ap.add_argument("--out", default="resultados_validacion.csv", metavar="CSV")
    args = ap.parse_args()

    ref_path = Path(args.ref)
    refs_dir = Path(args.refs_dir)

    ref_gray = cargar_gris(ref_path)
    if ref_gray is None:
        raise SystemExit(f"could not read the reference: {ref_path}")
    ref_roll, ref_pitch = cargar_imu(ref_path)
    if ref_roll is None:
        print(f"WARNING: {ref_path.name} has no .imu.json sidecar -> "
              f"pitch/roll_error will come out empty")

    filas = []
    for grupo, ntry, tiempo, foto in TABLA:
        foto_path = refs_dir / f"{foto}.png"
        img = cargar_gris(foto_path)
        nota = ""
        sim_pct = ""
        pitch_deg = roll_deg = pitch_err = roll_err = ""

        if img is None:
            nota = "image not found"
        else:
            img_cmp = img
            if img.shape != ref_gray.shape:
                img_cmp = cv2.resize(img, (ref_gray.shape[1], ref_gray.shape[0]),
                                      interpolation=cv2.INTER_AREA)
            sim_pct = round(ssim(ref_gray, img_cmp) * 100, 2)

            roll, pitch = cargar_imu(foto_path)
            if roll is not None:
                roll_deg = round(math.degrees(roll), 2)
                pitch_deg = round(math.degrees(pitch), 2)
                if ref_roll is not None:
                    roll_err = round(math.degrees(roll - ref_roll), 2)
                    pitch_err = round(math.degrees(pitch - ref_pitch), 2)
            else:
                nota = "no imu sidecar"

        filas.append([grupo, ntry, tiempo, foto, sim_pct, pitch_deg, roll_deg,
                      pitch_err, roll_err, nota])

    cabecera = ["Group", "Try", "Time_s", "Photo", "SSIM_%",
                "Pitch_deg", "Roll_deg", "Pitch_err_deg", "Roll_err_deg", "Note"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        w.writerows(filas)

    ancho = [max(len(str(c)) for c in [h, *[r[i] for r in filas]]) for i, h in enumerate(cabecera)]
    print("  ".join(h.ljust(w) for h, w in zip(cabecera, ancho)))
    for r in filas:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, ancho)))
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
