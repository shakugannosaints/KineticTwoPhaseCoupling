from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalize(v: Iterable[float], eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return np.zeros_like(arr)
    return arr / norm


def quat_normalize(q: Iterable[float]) -> np.ndarray:
    return normalize(q)


def quat_multiply(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_from_axis_angle(axis: Iterable[float], angle: float) -> np.ndarray:
    axis_n = normalize(axis)
    s = math.sin(angle * 0.5)
    return quat_normalize([math.cos(angle * 0.5), axis_n[0] * s, axis_n[1] * s, axis_n[2] * s])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return quat_normalize(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_rotate(q: Iterable[float], v: Iterable[float]) -> np.ndarray:
    return quat_to_matrix(q) @ np.asarray(v, dtype=np.float64)


def integrate_quaternion(q: Iterable[float], angular_velocity: Iterable[float], dt: float) -> np.ndarray:
    wx, wy, wz = np.asarray(angular_velocity, dtype=np.float64)
    omega = np.array([0.0, wx, wy, wz], dtype=np.float64)
    dq = 0.5 * quat_multiply(q, omega)
    return quat_normalize(np.asarray(q, dtype=np.float64) + dq * dt)


def transform_points(q: Iterable[float], translation: Iterable[float], points: np.ndarray) -> np.ndarray:
    rot = quat_to_matrix(q)
    t = np.asarray(translation, dtype=np.float64)
    return points @ rot.T + t
