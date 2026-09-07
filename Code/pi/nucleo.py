# Shared types (contracts) and every tunable parameter.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import yaml


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    index: int


@dataclass
class IMUSample:
    accel: np.ndarray
    gyro: np.ndarray
    timestamp: float


@dataclass
class ReferenceModel:
    image_gray: np.ndarray
    pyramid: list
    keypoints: list
    descriptors: np.ndarray

    ref_roll: float = 0.0
    ref_pitch: float = 0.0

    fft_ref: Optional[np.ndarray] = None


@dataclass
class EstimatorResult:
    transform: Optional[np.ndarray]
    confidence: float
    n_inliers: int = 0
    method: str = ""

    raw_score: float = 0.0
    raw_transform: Optional[np.ndarray] = None
    n_matches: int = 0
    reject_reason: str = ""

    yaw_deg: Optional[float] = None
    ess_pitch_deg: Optional[float] = None
    ess_roll_deg: Optional[float] = None
    ess_inliers: int = 0

    ref_pts: Optional[np.ndarray] = None
    live_pts: Optional[np.ndarray] = None

    sel_s_h: Optional[float] = None
    sel_s_e: Optional[float] = None
    sel_r_h: Optional[float] = None
    sel_gana: Optional[str] = None
    yaw_vetado_deg: Optional[float] = None


@dataclass
class GuidanceSignal:
    pan_px: float = 0.0
    tilt_px: float = 0.0
    zoom_rel: float = 1.0

    turn_deg: float = 0.0

    roll_err: float = 0.0
    pitch_err: float = 0.0

    locked: bool = False
    state: str = "search"
    confidence: float = 0.0

    sel_gana: Optional[str] = None


class FrameSource(ABC):

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        pass

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


class Estimator(ABC):

    @abstractmethod
    def estimate(
        self,
        live_frame: Frame,
        reference_model: ReferenceModel,
    ) -> EstimatorResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


@dataclass
class Config:
    lores_resolution: Tuple[int, int] = (640, 480)
    main_resolution: Tuple[int, int] = (1920, 1080)

    sensor_mode_size: Tuple[int, int] = (1640, 1232)
    sensor_mode_bit_depth: int = 10

    exposure_us: int = 30_000
    analogue_gain: float = 3.0
    colour_gains: Tuple[float, float] = (1.5, 1.5)
    lens_position: float = 0.0
    frame_duration_us: int = 33_333

    pyramid_levels: int = 3
    orb_n_features: int = 1000

    ref_min_keypoints: int = 50

    search_roi_fraction: float = 1.0
    search_scale: float = 1.0
    search_n_features: int = 500
    gms_min_matches: int = 20

    lowe_ratio: float = 0.85

    gms_threshold_factor: float = 6.0
    gms_with_rotation: bool = True
    gms_with_scale: bool = False

    magsac_reproj_px: float = 5.0
    magsac_max_iters: int = 2000
    magsac_confidence: float = 0.995

    min_matches_for_homography: int = 15

    inlier_ratio_threshold: float = 0.35
    min_inliers: int = 15

    max_translation_px: Optional[float] = None
    max_scale_deviation: Optional[float] = None
    max_perspective: Optional[float] = None

    gyro_stillness_threshold: float = 0.10
    stillness_frames: int = 3
    lockon_n_corners: int = 80

    phase_crop_size: int = 256
    phase_psr_threshold: float = 10.0
    phase_psr_exclusion: int = 11
    psr_norm_divisor: float = 30.0

    ibvs_depth: float = 2.0

    intrinsics_resolution: Tuple[int, int] = (640, 480)
    fx: float = 499.024
    fy: float = 498.933
    cx: float = 307.743
    cy: float = 252.843
    dist_coeffs: Tuple[float, ...] = (0.14772, -0.26011, 0.00185, -0.00058, 0.00000)

    focal_length_px: float = 499.0

    deadband_px: float = 3.0

    pitch_compensation: bool = True

    pitch_comp_max_deg: float = 4.0


    essential_yaw: bool = False

    essential_every_n_frames: int = 3

    essential_max_age_frames: int = 9

    essential_all_states: bool = False

    yaw_compensation: bool = True

    yaw_min_inliers: int = 40

    yaw_max_deg: float = 25.0

    yaw_deadband_deg: float = 1.5

    essential_reuse_search_points: bool = True

    gric_enabled: bool = False
    gric_gates_yaw: bool = False
    gric_rh_threshold: float = 0.40

    gric_yaw_from_h: bool = False

    lock_turn_deg_threshold: float = 2.0

    yaw_gates_lock: bool = True

    recovery_err_px: float = 40.0
    recovery_err_frames: int = 15

    lock_on_requires_yaw: bool = True

    lock_px_threshold: float = 15.0

    lock_deg_threshold: float = 5.0

    imu_filter_alpha: float = 0.98

    imu_tau_seconds: Optional[float] = 0.5

    gyro_bias_seconds: float = 2.0
    imu_odr_hz: int = 100

    capture_warmup_frames: int = 10
    capture_move_tolerance_deg: float = 1.0

    suavizado_enabled: bool = False

    suav_ventana_px_s: float = 0.4
    suav_banda_on_px: float = 14.0
    suav_banda_off_px: float = 7.0
    suav_margen_px: float = 16.0
    suav_paso_px: Optional[float] = None

    suav_ventana_zoom_s: float = 0.8
    suav_banda_on_zoom_pct: float = 8.0
    suav_banda_off_zoom_pct: float = 4.0
    suav_margen_zoom_pct: float = 10.0
    suav_paso_zoom_pct: Optional[float] = None

    suav_ventana_turn_s: float = 0.6
    suav_banda_on_turn_deg: float = 2.5
    suav_banda_off_turn_deg: float = 1.5
    suav_margen_turn_deg: float = 3.0
    suav_paso_turn_deg: Optional[float] = None


    yaw_coherencia: bool = True
    yaw_salto_max_deg: float = 8.0
    yaw_confirmaciones: int = 2
    yaw_caducidad_s: float = 0.4

    klt_min_points: int = 25
    klt_seed_min_points: int = 40
    klt_require_gric_e: bool = True
    klt_fb_threshold_px: float = 2.0
    klt_lk_win_size: int = 21
    klt_lk_max_level: int = 3
    klt_pnp_reproj_px: float = 4.0
    klt_pnp_confidence: float = 0.99
    klt_pnp_min_inliers: int = 15

    log_dir: str = "logs"
    log_every_n_frames: int = 1
    log_enabled: bool = True

    def camera_matrix(self):
        import numpy as np
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=float)

    def distortion(self):
        import numpy as np
        return np.array(self.dist_coeffs, dtype=float)

    def assert_intrinsics_match(self, resolution=None) -> None:
        res = tuple(resolution or self.lores_resolution)
        if res != tuple(self.intrinsics_resolution):
            raise ValueError(
                f"The intrinsics were measured at {tuple(self.intrinsics_resolution)} "
                f"and are about to be used at {res}. A K only describes ITS OWN resolution, and "
                f"rescaling it by hand would put an assumption back where there is "
                f"a measurement. Recalibrate at the new resolution "
                f"(calib_capture.py + calibrate_charuco.py)."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        tuple_fields = {"lores_resolution", "main_resolution", "colour_gains", "lk_win_size",
                        "sensor_mode_size", "intrinsics_resolution", "dist_coeffs"}
        for key in tuple_fields:
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_yaml(self, path: str | Path) -> None:
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
