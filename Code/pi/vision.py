# Framing channel: model selection, essential matrix, ORB search,
# KLT+PnP fine stage and the SEARCH -> LOCK_ON -> MICROMETRY state machine.
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from nucleo import Config, Estimator, EstimatorResult, Frame, ReferenceModel
from guia import _decompose_homography


TH_H = 5.991
TH_E = 3.841
SCORE_CONST = 5.991


def _homog(p: np.ndarray) -> np.ndarray:
    return np.hstack([p, np.ones((len(p), 1))])


def score_homography(ref_pts, live_pts, H, sigma=1.0) -> float:
    if H is None:
        return 0.0
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0.0
    inv_s2 = 1.0 / (sigma * sigma)
    total = 0.0
    for A, B, M in ((ref_pts, live_pts, H), (live_pts, ref_pts, Hi)):
        p = (M @ _homog(A).T).T
        w = p[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        d2 = np.sum((p[:, :2] / w - B) ** 2, axis=1) * inv_s2
        total += float(np.sum(np.where(d2 < TH_H, SCORE_CONST - d2, 0.0)))
    return total


def score_essential(ref_pts, live_pts, E, K, sigma=1.0) -> float:
    if E is None:
        return 0.0
    Ki = np.linalg.inv(K)
    F = Ki.T @ E @ Ki
    inv_s2 = 1.0 / (sigma * sigma)
    total = 0.0
    for A, B, M in ((ref_pts, live_pts, F), (live_pts, ref_pts, F.T)):
        l = (M @ _homog(A).T).T
        num = np.sum(l[:, :2] * B, axis=1) + l[:, 2]
        den = l[:, 0] ** 2 + l[:, 1] ** 2
        den = np.where(den < 1e-12, 1e-12, den)
        d2 = (num * num / den) * inv_s2
        total += float(np.sum(np.where(d2 < TH_E, SCORE_CONST - d2, 0.0)))
    return total


def seleccionar(ref_pts, live_pts, H, E, K, umbral=0.40, sigma=1.0) -> dict:
    if ref_pts is None or live_pts is None or len(ref_pts) < 8:
        return {"s_h": 0.0, "s_e": 0.0, "r_h": None, "gana": None}
    s_h = score_homography(ref_pts, live_pts, H, sigma)
    s_e = score_essential(ref_pts, live_pts, E, K, sigma)
    tot = s_h + s_e
    if tot <= 1e-9:
        return {"s_h": s_h, "s_e": s_e, "r_h": None, "gana": None}
    r_h = s_h / tot
    return {"s_h": s_h, "s_e": s_e, "r_h": float(r_h),
            "gana": "H" if r_h > umbral else "E"}


DEG = 180.0 / math.pi


def sin_distorsion(pts, K, dist):
    if dist is None:
        return pts
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, dist, P=K).reshape(-1, 2)


def solve_essential(ref_pts, live_pts, K):
    E, mask = cv2.findEssentialMat(ref_pts, live_pts, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None
    n_in, R, t, mask_pose = cv2.recoverPose(E, ref_pts, live_pts, K, mask=mask)
    rvec, _ = cv2.Rodrigues(R)
    pitch = +float(rvec[0]) * DEG
    yaw = -float(rvec[1]) * DEG
    roll = -float(rvec[2]) * DEG
    t = t.ravel() / (np.linalg.norm(t) or 1.0)
    return {"yaw": yaw, "pitch": pitch, "roll": roll,
            "t_right": float(t[0]), "t_down": float(t[1]), "t_fwd": float(t[2]),
            "inliers": int(n_in), "ratio": int(n_in) / max(1, len(ref_pts)),
            "E": E,
            "R": R, "mask_pose": mask_pose}


def homography_to_ypr(H, K, ref_pts, live_pts):
    if ref_pts is None or live_pts is None or len(ref_pts) < 4:
        return None
    try:
        n_sol, Rs, _, Ns = cv2.decomposeHomographyMat(H, K)
        if not n_sol:
            return None
        if n_sol == 1:
            idx = [0]
        else:
            ref32 = np.asarray(ref_pts, dtype=np.float32).reshape(-1, 1, 2)
            live32 = np.asarray(live_pts, dtype=np.float32).reshape(-1, 1, 2)
            keep = cv2.filterHomographyDecompByVisibleRefpoints(Rs, Ns, ref32, live32)
            idx = [int(i) for i in np.asarray(keep).ravel()] if keep is not None else []
    except cv2.error:
        return None
    if len(idx) != 1:
        return None
    rvec, _ = cv2.Rodrigues(Rs[idx[0]])
    pitch = +float(rvec[0]) * DEG
    yaw   = -float(rvec[1]) * DEG
    roll  = -float(rvec[2]) * DEG
    if not all(math.isfinite(v) for v in (yaw, pitch, roll)):
        return None
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


class EssentialYawEstimator:

    def __init__(self, config) -> None:
        self._cfg = config
        self._orb = cv2.ORB_create(nfeatures=getattr(config, "orb_n_features", 1000))
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._gms_ok = (hasattr(cv2, "xfeatures2d")
                        and hasattr(cv2.xfeatures2d, "matchGMS"))
        self._since = 10 ** 9
        self._aviso_recorte = False
        self.reutilizados = 0
        self.emparejados = 0
        self._last = None
        self._last_age = 0

    def match(self, ref_model, live_gray):
        cfg = self._cfg
        ref_kps, ref_desc = ref_model.keypoints, ref_model.descriptors
        if ref_desc is None or len(ref_kps) == 0:
            return None, None, 0

        live_kps, live_desc = self._orb.detectAndCompute(live_gray, None)
        if live_desc is None or len(live_kps) < cfg.gms_min_matches:
            return None, None, 0

        knn = self._bf.knnMatch(live_desc, ref_desc, k=2)
        good = [q[0] for q in knn
                if len(q) == 2 and q[0].distance < cfg.lowe_ratio * q[1].distance]
        n_raw = len(good)
        if n_raw < cfg.gms_min_matches:
            return None, None, n_raw

        if self._gms_ok:
            try:
                g = cv2.xfeatures2d.matchGMS(
                    size1=live_gray.shape[::-1],
                    size2=ref_model.image_gray.shape[::-1],
                    keypoints1=live_kps, keypoints2=ref_kps,
                    matches1to2=good,
                    withRotation=cfg.gms_with_rotation,
                    withScale=cfg.gms_with_scale,
                    thresholdFactor=cfg.gms_threshold_factor)
                if len(g) >= cfg.min_matches_for_homography:
                    good = g
            except cv2.error:
                pass

        if len(good) < cfg.min_matches_for_homography:
            return None, None, n_raw

        live_pts = np.float64([live_kps[m.queryIdx].pt for m in good])
        ref_pts = np.float64([ref_kps[m.trainIdx].pt for m in good])
        return ref_pts, live_pts, n_raw

    def _puede_reutilizar(self) -> bool:
        cfg = self._cfg
        if not getattr(cfg, "essential_reuse_search_points", True):
            return False
        entero = float(getattr(cfg, "search_roi_fraction", 1.0)) >= 0.999
        completa = float(getattr(cfg, "search_scale", 1.0)) >= 0.999
        if not (entero and completa):
            if not self._aviso_recorte:
                self._aviso_recorte = True
                print("WARNING: search crops or downscales "
                      f"(roi {getattr(cfg, 'search_roi_fraction', 1.0)}, "
                      f"scale {getattr(cfg, 'search_scale', 1.0)}), so the "
                      "essential\n       does NOT reuse its points and it "
                      "matches separately. That is correct: with cropping\n       the "
                      "correspondences are not spread out and E dies.")
            return False
        return True

    def estimate(self, live_gray, ref_model, pts=None):
        cfg = self._cfg
        every = max(1, int(getattr(cfg, "essential_every_n_frames", 3)))
        max_age = int(getattr(cfg, "essential_max_age_frames", 9))

        self._since += 1
        if self._since < every:
            if self._last is not None:
                self._last_age += 1
                if self._last_age > max_age:
                    self._last = None
                    return None
                out = dict(self._last)
                out["age"] = self._last_age
                return out
            return None

        self._since = 0
        r = self._compute(live_gray, ref_model, pts)
        if r is None:
            self._last = None
            return None
        self._last = r
        self._last_age = 0
        out = dict(r)
        out["age"] = 0
        return out

    def _compute(self, live_gray, ref_model, pts=None):
        try:
            cfg = self._cfg
            if pts is not None and pts[0] is not None and self._puede_reutilizar():
                ref_pts, live_pts = pts
                self.reutilizados += 1
            else:
                ref_pts, live_pts, _ = self.match(ref_model, live_gray)
                self.emparejados += 1
            if ref_pts is None or len(ref_pts) < 8:
                return None
            K, dist = cfg.camera_matrix(), cfg.distortion()
            rp = sin_distorsion(ref_pts, K, dist)
            lp = sin_distorsion(live_pts, K, dist)
            r = solve_essential(rp, lp, K)
            if r is None:
                return None
            r["ref_pts"] = rp
            r["live_pts"] = lp
            r["ref_pts_raw"] = ref_pts
            r["live_pts_raw"] = live_pts

            if getattr(cfg, "gric_enabled", False):
                try:
                    Hc, _ = cv2.findHomography(
                        rp, lp, cv2.USAC_MAGSAC,
                        ransacReprojThreshold=cfg.magsac_reproj_px,
                        maxIters=cfg.magsac_max_iters,
                        confidence=cfg.magsac_confidence)
                    r.update(seleccionar(rp, lp, Hc, r.get("E"), K,
                                         getattr(cfg, "gric_rh_threshold", 0.40)))
                    if (r.get("gana") == "H"
                            and getattr(cfg, "gric_yaw_from_h", False)):
                        h_ypr = homography_to_ypr(Hc, K, rp, lp)
                        if h_ypr is not None:
                            r.update(h_ypr)
                            r["yaw_de_h"] = True
                except Exception:
                    r.update({"s_h": 0.0, "s_e": 0.0, "r_h": None, "gana": None})
            if not all(math.isfinite(r[k]) for k in ("yaw", "pitch", "roll")):
                return None
            if r["inliers"] < getattr(cfg, "yaw_min_inliers", 40):
                return None
            return r
        except Exception:
            return None


class ORBSearchEstimator(Estimator):

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._scale = config.search_scale
        self._orb = cv2.ORB_create(nfeatures=config.search_n_features)
        self._ref_orb = cv2.ORB_create(nfeatures=config.orb_n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._cached_ref: ReferenceModel | None = None
        self._cached_hw: tuple[int, int] | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self._roi_small_size: tuple[int, int] | None = None
        self._ref_kps_small: list | None = None
        self._ref_descs_small: np.ndarray | None = None

        self._has_gms = hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "matchGMS")


    @property
    def name(self) -> str:
        return "ORB+GMS+USAC_MAGSAC"

    def _ensure_reference_cache(
        self, reference_model: ReferenceModel, h: int, w: int
    ) -> None:
        if (
            reference_model is self._cached_ref
            and self._cached_hw == (h, w)
            and self._roi is not None
        ):
            return

        cfg = self._cfg
        x0, y0, x1, y1 = _central_roi(w, h, cfg.search_roi_fraction)
        ref_roi = reference_model.image_gray[y0:y1, x0:x1]
        ref_small = _downscale(ref_roi, self._scale)
        rh, rw = ref_small.shape[:2]

        ref_kps, ref_descs = self._ref_orb.detectAndCompute(ref_small, None)

        self._roi = (x0, y0, x1, y1)
        self._roi_small_size = (rw, rh)
        self._ref_kps_small = ref_kps
        self._ref_descs_small = ref_descs
        self._cached_ref = reference_model
        self._cached_hw = (h, w)

    def estimate(
        self,
        live_frame: Frame,
        reference_model: ReferenceModel,
    ) -> EstimatorResult:
        cfg = self._cfg

        live_gray = _to_gray(live_frame.image)
        h, w = live_gray.shape[:2]

        self._ensure_reference_cache(reference_model, h, w)
        ref_kps_small = self._ref_kps_small
        ref_descs_small = self._ref_descs_small
        if ref_descs_small is None or len(ref_kps_small) < 8:
            return EstimatorResult(transform=None, confidence=0.0, method=self.name)

        x0, y0, x1, y1 = self._roi
        live_roi = live_gray[y0:y1, x0:x1]
        roi_small = _downscale(live_roi, self._scale)
        roi_w, roi_h = self._roi_small_size

        live_kps, live_descs = self._orb.detectAndCompute(roi_small, None)
        if live_descs is None or len(live_kps) < 8:
            return EstimatorResult(transform=None, confidence=0.0, method=self.name,
                                   reject_reason="too_few_live_keypoints")

        knn_matches = self._matcher.knnMatch(live_descs, ref_descs_small, k=2)

        ratio = cfg.lowe_ratio
        good_matches = [
            p[0] for p in knn_matches
            if len(p) == 2 and p[0].distance < ratio * p[1].distance
        ]

        if len(good_matches) < cfg.gms_min_matches:
            return EstimatorResult(transform=None, confidence=0.0, method=self.name,
                                   n_matches=len(good_matches),
                                   reject_reason="too_few_ratio_matches")

        gms_matches = good_matches
        if self._has_gms:
            try:
                gms_matches = cv2.xfeatures2d.matchGMS(
                    size1=(roi_w, roi_h),
                    size2=(roi_w, roi_h),
                    keypoints1=live_kps,
                    keypoints2=ref_kps_small,
                    matches1to2=good_matches,
                    withRotation=cfg.gms_with_rotation,
                    withScale=cfg.gms_with_scale,
                    thresholdFactor=cfg.gms_threshold_factor,
                )
                if len(gms_matches) < cfg.min_matches_for_homography:
                    gms_matches = good_matches
            except cv2.error:
                gms_matches = good_matches

        if len(gms_matches) < cfg.min_matches_for_homography:
            return EstimatorResult(transform=None, confidence=0.0, method=self.name,
                                   n_matches=len(gms_matches),
                                   reject_reason="too_few_matches_for_homography")

        live_pts_small = np.float32([live_kps[m.queryIdx].pt for m in gms_matches])
        ref_pts_small = np.float32([ref_kps_small[m.trainIdx].pt for m in gms_matches])

        inv_scale = 1.0 / self._scale

        def _to_full(pts_small: np.ndarray) -> np.ndarray:
            pts = pts_small * inv_scale
            pts[:, 0] += x0
            pts[:, 1] += y0
            return pts

        ref_pts = _to_full(ref_pts_small)
        live_pts = _to_full(live_pts_small)

        if len(live_pts) < 4:
            return EstimatorResult(transform=None, confidence=0.0, method=self.name,
                                   n_matches=len(gms_matches),
                                   reject_reason="too_few_points")

        homography, mask = cv2.findHomography(
            ref_pts,
            live_pts,
            cv2.USAC_MAGSAC,
            ransacReprojThreshold=cfg.magsac_reproj_px,
            maxIters=cfg.magsac_max_iters,
            confidence=cfg.magsac_confidence,
        )

        if homography is None or mask is None:
            return EstimatorResult(transform=None, confidence=0.0, method=self.name,
                                   n_matches=len(gms_matches),
                                   reject_reason="no_homography")

        n_inliers = int(mask.sum())
        inlier_ratio = n_inliers / len(gms_matches)

        reason = _implausible(homography, cfg)
        if reason:
            return EstimatorResult(
                transform=None,
                confidence=0.0,
                n_inliers=n_inliers,
                method=self.name,
                raw_score=inlier_ratio,
                raw_transform=homography,
                n_matches=len(gms_matches),
                reject_reason=reason,
            )

        return EstimatorResult(
            transform=homography,
            ref_pts=ref_pts,
            live_pts=live_pts,
            confidence=inlier_ratio,
            n_inliers=n_inliers,
            method=self.name,
            raw_score=inlier_ratio,
            raw_transform=homography,
            n_matches=len(gms_matches),
        )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _downscale(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    return cv2.resize(
        img,
        (int(img.shape[1] * scale), int(img.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _implausible(H: np.ndarray, cfg: Config) -> str:
    if cfg.max_translation_px is not None:
        if abs(H[0, 2]) > cfg.max_translation_px or abs(H[1, 2]) > cfg.max_translation_px:
            return "implausible_translation"

    if cfg.max_scale_deviation is not None:
        det = abs(float(np.linalg.det(H[:2, :2])))
        scale = math.sqrt(det) if det > 0 else 0.0
        if abs(scale - 1.0) > cfg.max_scale_deviation:
            return "implausible_scale"

    if cfg.max_perspective is not None:
        if abs(H[2, 0]) > cfg.max_perspective or abs(H[2, 1]) > cfg.max_perspective:
            return "implausible_perspective"

    return ""


def _central_roi(w: int, h: int, fraction: float) -> tuple[int, int, int, int]:
    rw = int(w * fraction)
    rh = int(h * fraction)
    x0 = (w - rw) // 2
    y0 = (h - rh) // 2
    return x0, y0, x0 + rw, y0 + rh


def _gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class KLTPnPEstimator(Estimator):

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._cloud = None
        self._ref2d = None
        self._orig_n = 0
        self._prev_gray = None
        self._prev_pts = None
        self.reseeds = 0
        self.starvations = 0

    @property
    def name(self) -> str:
        return "KLT+PnP"

    def has_cloud(self) -> bool:
        return self._cloud is not None

    def seed(self, cloud_3d: np.ndarray, ref2d_undist: np.ndarray,
             live_gray_raw: np.ndarray, live_pts_raw: np.ndarray) -> None:
        n = len(cloud_3d)
        self._cloud = np.asarray(cloud_3d, dtype=np.float64).reshape(n, 3)
        self._ref2d = np.asarray(ref2d_undist, dtype=np.float64).reshape(n, 2)
        self._orig_n = n
        self._prev_gray = live_gray_raw
        self._prev_pts = np.asarray(live_pts_raw, dtype=np.float32).reshape(n, 1, 2)
        self.reseeds += 1

    def estimate(self, live_frame, reference_model) -> EstimatorResult:
        cfg = self._cfg
        if self._cloud is None:
            return EstimatorResult(transform=None, confidence=0.0, method=self.name,
                                    reject_reason="no_cloud")

        gray = _gray(live_frame.image)
        win = int(getattr(cfg, "klt_lk_win_size", 21))
        lk_params = dict(
            winSize=(win, win),
            maxLevel=int(getattr(cfg, "klt_lk_max_level", 3)),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **lk_params)
        back_pts, st_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, new_pts, None, **lk_params)

        fb_err = np.linalg.norm((self._prev_pts - back_pts).reshape(-1, 2), axis=1)
        good = ((st.reshape(-1) == 1) & (st_b.reshape(-1) == 1) &
               (fb_err < float(getattr(cfg, "klt_fb_threshold_px", 2.0))))
        n_alive = int(good.sum())

        if n_alive < int(getattr(cfg, "klt_min_points", 25)):
            self.starvations += 1
            self._cloud = None
            return EstimatorResult(transform=None, confidence=0.0, method=self.name,
                                    n_inliers=n_alive, reject_reason="klt_starvation")

        self._cloud = self._cloud[good]
        self._ref2d = self._ref2d[good]
        live_pts_raw = new_pts[good].reshape(-1, 2)
        self._prev_pts = live_pts_raw.reshape(-1, 1, 2).astype(np.float32)
        self._prev_gray = gray

        K, dist = cfg.camera_matrix(), cfg.distortion()
        live_undist = sin_distorsion(live_pts_raw, K, dist)

        transform = None
        n_matches = len(self._ref2d)
        if n_matches >= 4:
            transform, _ = cv2.findHomography(
                self._ref2d, live_undist, cv2.USAC_MAGSAC,
                ransacReprojThreshold=cfg.magsac_reproj_px,
                maxIters=cfg.magsac_max_iters, confidence=cfg.magsac_confidence)

        confidence = n_alive / max(1, self._orig_n)
        result = EstimatorResult(
            transform=transform,
            confidence=float(confidence) if transform is not None else 0.0,
            n_inliers=n_alive, n_matches=n_matches, method=self.name,
        )

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                self._cloud, live_undist, K, None,
                reprojectionError=float(getattr(cfg, "klt_pnp_reproj_px", 4.0)),
                confidence=float(getattr(cfg, "klt_pnp_confidence", 0.99)),
                flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok, inliers = False, None

        if ok and inliers is not None and len(inliers) >= int(getattr(cfg, "klt_pnp_min_inliers", 15)):
            pitch = +float(rvec[0]) * DEG
            yaw   = -float(rvec[1]) * DEG
            roll  = -float(rvec[2]) * DEG
            if all(math.isfinite(v) for v in (yaw, pitch, roll)):
                result = replace(result, yaw_deg=yaw, ess_pitch_deg=pitch,
                                 ess_roll_deg=roll, ess_inliers=int(len(inliers)))
        return result


class State(Enum):
    SEARCH     = auto()
    LOCK_ON    = auto()
    MICROMETRY = auto()


@dataclass
class AlignmentResult:
    transform: Optional[np.ndarray]
    confidence: float
    state: State
    lost: bool
    estimator_result: EstimatorResult
    timing: dict = field(default_factory=dict)


class StillnessTracker:

    def __init__(self, threshold: float, required_frames: int) -> None:
        self._threshold = threshold
        self._required = required_frames
        self._count = 0

    def update(self, imu_sample) -> None:
        if imu_sample is None:
            self._count = self._required
            return
        gyro_mag = float(np.linalg.norm(imu_sample.gyro))
        if gyro_mag < self._threshold:
            self._count = min(self._count + 1, self._required)
        else:
            self._count = 0

    def is_still(self) -> bool:
        return self._count >= self._required

    def reset(self) -> None:
        self._count = 0


class AlignmentEngine:

    def __init__(
        self,
        reference_model: ReferenceModel,
        config: Config,
        search_estimator=None,
        fine_estimator=None,
    ) -> None:
        self._ref   = reference_model
        self._cfg   = config
        self._state = State.SEARCH

        self._search_estimator = search_estimator or ORBSearchEstimator(config)
        self._fine_estimator   = fine_estimator

        self._yaw_estimator = (EssentialYawEstimator(config)
                               if getattr(config, "essential_yaw", False) else None)

        self._stillness = StillnessTracker(
            threshold=config.gyro_stillness_threshold,
            required_frames=config.stillness_frames,
        )

        self._last_transform: Optional[np.ndarray] = None
        self._last_confidence: float = 0.0

        self._state_frame_count: int = 0

        self._big_err_frames: int = 0


    def update(self, live_frame: Frame, imu_sample=None) -> AlignmentResult:
        self._stillness.update(imu_sample)
        self._state_frame_count += 1

        if self._state == State.SEARCH:
            return self._tick_search(live_frame)
        elif self._state == State.LOCK_ON:
            return self._tick_lock_on(live_frame)
        elif self._state == State.MICROMETRY:
            return self._tick_micrometry(live_frame)


    def _tick_search(self, live_frame: Frame) -> AlignmentResult:
        t0 = time.perf_counter()
        est = self._search_estimator.estimate(live_frame, self._ref)
        search_ms = (time.perf_counter() - t0) * 1000.0

        ess_ms = 0.0
        if self._yaw_estimator is not None:
            t1 = time.perf_counter()
            ess = self._yaw_estimator.estimate(
                _gray(live_frame.image), self._ref,
                pts=(getattr(est, "ref_pts", None), getattr(est, "live_pts", None)))
            ess_ms = (time.perf_counter() - t1) * 1000.0
            if ess is not None:
                est = replace(est,
                              yaw_deg=ess["yaw"],
                              ess_pitch_deg=ess["pitch"],
                              ess_roll_deg=ess["roll"],
                              ess_inliers=ess["inliers"])
                est = self._aplicar_seleccion(est, ess)

        if est.transform is not None:
            self._last_transform  = est.transform
            self._last_confidence = est.confidence

        aimed = True
        if getattr(self._cfg, "lock_on_requires_yaw", False):
            y = getattr(est, "yaw_deg", None)
            if y is not None:
                aimed = abs(y) <= self._cfg.lock_turn_deg_threshold

        if (
            est.transform is not None
            and est.confidence >= self._cfg.inlier_ratio_threshold
            and est.n_inliers >= self._cfg.min_inliers
            and self._stillness.is_still()
            and aimed
        ):
            self._transition_to(State.LOCK_ON)

        return AlignmentResult(
            transform=self._last_transform,
            confidence=self._last_confidence,
            state=State.SEARCH,
            lost=(self._last_transform is None),
            estimator_result=est,
            timing={"search_ms": search_ms, "ess_ms": ess_ms},
        )

    def _tick_lock_on(self, live_frame: Frame) -> AlignmentResult:
        t0 = time.perf_counter()
        est = self._search_estimator.estimate(live_frame, self._ref)
        search_ms = (time.perf_counter() - t0) * 1000.0
        if est.transform is not None:
            self._last_transform  = est.transform
            self._last_confidence = est.confidence


        self._transition_to(State.MICROMETRY)

        return AlignmentResult(
            transform=self._last_transform,
            confidence=self._last_confidence,
            state=State.LOCK_ON,
            lost=(self._last_transform is None),
            estimator_result=est,
            timing={"search_ms": search_ms},
        )

    def _observar_yaw(self, live_frame: Frame, est):
        if self._yaw_estimator is None:
            return est
        if not getattr(self._cfg, "essential_all_states", False):
            return est
        ess = self._yaw_estimator.estimate(_gray(live_frame.image), self._ref)
        if ess is None:
            return est
        est = replace(est, yaw_deg=ess["yaw"], ess_pitch_deg=ess["pitch"],
                      ess_roll_deg=ess["roll"], ess_inliers=ess["inliers"])
        return self._aplicar_seleccion(est, ess)

    def _aplicar_seleccion(self, est, ess):
        if ess.get("r_h") is None and ess.get("gana") is None:
            return est
        est = replace(est,
                      sel_s_h=ess.get("s_h"), sel_s_e=ess.get("s_e"),
                      sel_r_h=ess.get("r_h"), sel_gana=ess.get("gana"))
        if (getattr(self._cfg, "gric_gates_yaw", False)
                and ess.get("gana") == "H" and not ess.get("yaw_de_h")):
            est = replace(est, yaw_vetado_deg=est.yaw_deg, yaw_deg=None)
        return est

    def _tick_micrometry(self, live_frame: Frame) -> AlignmentResult:
        t0 = time.perf_counter()
        est = self._fine_estimator.estimate(live_frame, self._ref)
        micro_ms = (time.perf_counter() - t0) * 1000.0

        if est.transform is not None:
            self._last_transform  = est.transform
            self._last_confidence = est.confidence

        if est.transform is None:
            self._recover()

        elif self._error_too_large():
            self._big_err_frames += 1
            if self._big_err_frames >= self._cfg.recovery_err_frames:
                self._recover()
        else:
            self._big_err_frames = 0

        return AlignmentResult(
            transform=self._last_transform,
            confidence=self._last_confidence,
            state=State.MICROMETRY,
            lost=(self._last_transform is None),
            estimator_result=self._observar_yaw(live_frame, est),
            timing={"micro_ms": micro_ms},
        )


    def _error_too_large(self) -> bool:
        T = self._last_transform
        if T is None:
            return False
        try:
            img = getattr(self._ref, "image_gray", None)
            size = (img.shape[1], img.shape[0]) if img is not None else None
            dx, dy, _ = _decompose_homography(np.asarray(T, dtype=float), size)
            return math.hypot(dx, dy) > self._cfg.recovery_err_px
        except Exception:
            return False

    def _transition_to(self, new_state: State) -> None:
        self._state = new_state
        self._state_frame_count = 0
        self._big_err_frames = 0
        if new_state == State.SEARCH:
            self._stillness.reset()

    def _recover(self) -> None:
        self._transition_to(State.SEARCH)


    @property
    def state(self) -> State:
        return self._state


class KLTAlignmentEngine(AlignmentEngine):

    def __init__(self, reference_model, config, klt_estimator=None):
        self.klt = klt_estimator or KLTPnPEstimator(config)
        super().__init__(reference_model, config, fine_estimator=self.klt)
        if self._yaw_estimator is None:
            raise ValueError(
                "KLTAlignmentEngine requires config.essential_yaw = True "
                "(the cloud is triangulated from the essential matrix).")
        self.seed_failures = 0

    def _tick_lock_on(self, live_frame):
        result = super()._tick_lock_on(live_frame)
        self._seed_cloud(live_frame)
        return result

    def _seed_cloud(self, live_frame) -> None:
        cfg = self._cfg
        gray = _gray(live_frame.image)

        info = self._yaw_estimator._compute(gray, self._ref)
        if not info or info.get("R") is None or info.get("mask_pose") is None:
            self.seed_failures += 1
            return
        if (getattr(cfg, "klt_require_gric_e", True)
                and info.get("gana") not in (None, "E")):
            self.seed_failures += 1
            return

        mask = np.asarray(info["mask_pose"]).reshape(-1).astype(bool)
        if mask.sum() < getattr(cfg, "klt_seed_min_points", 40):
            self.seed_failures += 1
            return

        rp = info["ref_pts"][mask]
        lp = info["live_pts"][mask]
        lp_raw = info["live_pts_raw"][mask]
        R = info["R"]
        t = np.array([info["t_right"], info["t_down"], info["t_fwd"]], dtype=np.float64)

        K = cfg.camera_matrix()
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t.reshape(3, 1)])
        pts4d = cv2.triangulatePoints(P1, P2,
                                      rp.T.astype(np.float64), lp.T.astype(np.float64))
        pts3d = (pts4d[:3] / pts4d[3]).T

        depth_ref = pts3d[:, 2]
        depth_live = (R @ pts3d.T + t.reshape(3, 1)).T[:, 2]
        good = (depth_ref > 0) & (depth_live > 0) & np.isfinite(pts3d).all(axis=1)
        if good.sum() < getattr(cfg, "klt_min_points", 25):
            self.seed_failures += 1
            return

        self.klt.seed(pts3d[good], rp[good], gray, lp_raw[good])
