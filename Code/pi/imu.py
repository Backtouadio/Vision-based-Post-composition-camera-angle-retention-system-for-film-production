# Leveling channel: BMI160 over I2C + complementary filter -> roll/pitch.
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np

from nucleo import Config, IMUSample


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def accel_to_roll_pitch(accel) -> Tuple[float, float]:
    ax, ay, az = float(accel[0]), float(accel[1]), float(accel[2])
    roll  = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    return roll, pitch


def estimate_gyro_bias(gyro_samples) -> np.ndarray:
    arr = np.asarray(gyro_samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("gyro_samples must be a non-empty (N, 3) array")
    return arr.mean(axis=0)


class ComplementaryFilter:

    def __init__(self, alpha: float = 0.98, tau: Optional[float] = None) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        if tau is not None and float(tau) <= 0.0:
            raise ValueError("tau must be positive (or None for fixed alpha)")
        self.tau = None if tau is None else float(tau)
        self.roll: float = 0.0
        self.pitch: float = 0.0
        self._initialized = False

    def alpha_for(self, dt: float) -> float:
        if self.tau is None:
            return self.alpha
        return self.tau / (self.tau + dt)

    def reset(self) -> None:
        self.roll = 0.0
        self.pitch = 0.0
        self._initialized = False

    def update(self, accel, gyro, dt: float) -> Tuple[float, float]:
        roll_acc, pitch_acc = accel_to_roll_pitch(accel)

        if not self._initialized:
            self.roll = roll_acc
            self.pitch = pitch_acc
            self._initialized = True
            return self.roll, self.pitch

        if dt <= 0.0:
            b = 1.0 - self.alpha
            self.roll = wrap_pi(self.roll + b * wrap_pi(roll_acc - self.roll))
            self.pitch = wrap_pi(self.pitch + b * wrap_pi(pitch_acc - self.pitch))
            return self.roll, self.pitch

        gx, gy = float(gyro[0]), float(gyro[1])
        roll_gyro = self.roll + gx * dt
        pitch_gyro = self.pitch + gy * dt

        a = self.alpha_for(dt)

        b = 1.0 - a
        self.roll = wrap_pi(roll_gyro + b * wrap_pi(roll_acc - roll_gyro))
        self.pitch = wrap_pi(pitch_gyro + b * wrap_pi(pitch_acc - pitch_gyro))
        return self.roll, self.pitch


class _BMI160I2C:

    _ADDR_DEFAULT = 0x69
    _REG_CHIP_ID = 0x00
    _CHIP_ID = 0xD1
    _REG_CMD = 0x7E
    _CMD_ACC_NORMAL = 0x11
    _CMD_GYR_NORMAL = 0x15
    _REG_DATA_GYR = 0x0C
    _REG_ACC_RANGE = 0x41
    _REG_GYR_RANGE = 0x43

    _ACC_RANGE = (0x03, 2.0)
    _GYR_RANGE = (0x03, 250.0)
    _G = 9.80665

    _MOUNT_MAP = ((2, +1.0), (1, +1.0), (0, -1.0))

    def __init__(self, bus_id: int = 1, address: Optional[int] = None,
                 apply_mount: bool = True) -> None:
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise RuntimeError(
                "smbus2 not available — BMI160 read is Raspberry Pi only. "
                "Use the ComplementaryFilter directly with recorded/synthetic "
                "samples for laptop development."
            ) from exc
        self._addr = address if address is not None else self._ADDR_DEFAULT
        self._bus = SMBus(bus_id)
        self.apply_mount = bool(apply_mount)
        self._acc_lsb_to_ms2 = (self._ACC_RANGE[1] / 32768.0) * self._G
        self._gyr_lsb_to_rads = math.radians(self._GYR_RANGE[1] / 32768.0)

    def start(self) -> None:
        chip = self._bus.read_byte_data(self._addr, self._REG_CHIP_ID)
        if chip != self._CHIP_ID:
            raise RuntimeError(
                f"BMI160 not found at 0x{self._addr:02X} "
                f"(chip id 0x{chip:02X}, expected 0x{self._CHIP_ID:02X})"
            )
        self._bus.write_byte_data(self._addr, self._REG_CMD, self._CMD_ACC_NORMAL)
        time.sleep(0.005)
        self._bus.write_byte_data(self._addr, self._REG_CMD, self._CMD_GYR_NORMAL)
        time.sleep(0.08)

        self._write_range(self._REG_ACC_RANGE, self._ACC_RANGE[0], "ACC_RANGE")
        self._write_range(self._REG_GYR_RANGE, self._GYR_RANGE[0], "GYR_RANGE")

    def _write_range(self, reg: int, value: int, name: str) -> None:
        self._bus.write_byte_data(self._addr, reg, value)
        time.sleep(0.002)
        got = self._bus.read_byte_data(self._addr, reg)
        if got != value:
            raise RuntimeError(
                f"{name} (0x{reg:02X}) did not take: wrote 0x{value:02X}, "
                f"read back 0x{got:02X}. The scale factors in _BMI160I2C would "
                f"be wrong, so refusing to continue."
            )

    @staticmethod
    def _s16(lsb: int, msb: int) -> int:
        v = (msb << 8) | lsb
        return v - 65536 if v >= 32768 else v

    def read(self) -> IMUSample:
        d = self._bus.read_i2c_block_data(self._addr, self._REG_DATA_GYR, 12)
        g_raw = (self._s16(d[0], d[1]),
                 self._s16(d[2], d[3]),
                 self._s16(d[4], d[5]))
        a_raw = (self._s16(d[6], d[7]),
                 self._s16(d[8], d[9]),
                 self._s16(d[10], d[11]))
        if self.apply_mount:
            gyro = [sgn * g_raw[i] * self._gyr_lsb_to_rads
                    for i, sgn in self._MOUNT_MAP]
            accel = [sgn * a_raw[i] * self._acc_lsb_to_ms2
                     for i, sgn in self._MOUNT_MAP]
        else:
            gyro = [v * self._gyr_lsb_to_rads for v in g_raw]
            accel = [v * self._acc_lsb_to_ms2 for v in a_raw]
        return IMUSample(
            accel=np.array(accel, dtype=np.float32),
            gyro=np.array(gyro, dtype=np.float32),
            timestamp=time.monotonic(),
        )

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass


class OrientationSource:

    def __init__(self, config: Config, driver=None) -> None:
        self._cfg = config
        self._filter = ComplementaryFilter(
            alpha=config.imu_filter_alpha,
            tau=getattr(config, "imu_tau_seconds", None),
        )
        self._driver = driver
        self._bias = np.zeros(3, dtype=np.float64)
        self._last_ts: Optional[float] = None
        self.last_sample: Optional[IMUSample] = None
        self.roll: float = 0.0
        self.pitch: float = 0.0


    def open(self) -> None:
        if self._driver is None:
            self._driver = _BMI160I2C()
        self._driver.start()
        self._calibrate_bias()
        self._filter.reset()
        self._last_ts = None

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def __enter__(self) -> "OrientationSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


    def _calibrate_bias(self) -> None:
        n = max(1, int(self._cfg.gyro_bias_seconds * self._cfg.imu_odr_hz))
        period = 1.0 / max(1, self._cfg.imu_odr_hz)
        samples = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            samples[i] = self._driver.read().gyro
            time.sleep(period)
        self._bias = estimate_gyro_bias(samples)


    def orientation(self) -> Tuple[float, float]:
        raw = self._driver.read()
        gyro_corr = np.asarray(raw.gyro, dtype=np.float64) - self._bias

        ts = raw.timestamp
        dt = 0.0 if self._last_ts is None else (ts - self._last_ts)
        self._last_ts = ts

        self.roll, self.pitch = self._filter.update(raw.accel, gyro_corr, dt)

        self.last_sample = IMUSample(
            accel=np.asarray(raw.accel, dtype=np.float32),
            gyro=gyro_corr.astype(np.float32),
            timestamp=ts,
        )
        return self.roll, self.pitch

    @property
    def gyro_bias(self) -> np.ndarray:
        return self._bias.copy()
