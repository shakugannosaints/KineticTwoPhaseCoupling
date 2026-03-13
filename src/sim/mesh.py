from __future__ import annotations

import numpy as np
from skimage import measure


def extract_surface_mesh(phi: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume = np.asarray(phi, dtype=np.float32)
    if volume.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float32)
    if float(volume.max()) < 0.5 or float(volume.min()) > 0.5:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float32)
    occupied = np.argwhere(volume >= 0.5)
    if occupied.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float32)
    pad = np.array([2, 2, 2], dtype=np.int32)
    lower = np.maximum(occupied.min(axis=0) - pad, 0)
    upper = np.minimum(occupied.max(axis=0) + pad + 1, np.array(volume.shape, dtype=np.int32))
    cropped = volume[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
    vertices, faces, normals, _ = measure.marching_cubes(cropped, level=0.5, spacing=(cell_size, cell_size, cell_size))
    vertices += lower.astype(np.float32) * cell_size
    return vertices.astype(np.float32), faces.astype(np.int32), normals.astype(np.float32)


def open_tank_wireframe(extent: tuple[float, float, float]) -> np.ndarray:
    x, y, z = extent
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [x, 0.0, 0.0],
            [x, 0.0, z],
            [0.0, 0.0, z],
            [0.0, y, 0.0],
            [x, y, 0.0],
            [x, y, z],
            [0.0, y, z],
        ],
        dtype=np.float32,
    )
    edge_indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (7, 4),
    )
    segments = []
    for a, b in edge_indices:
        segments.append(corners[a])
        segments.append(corners[b])
    return np.asarray(segments, dtype=np.float32)
