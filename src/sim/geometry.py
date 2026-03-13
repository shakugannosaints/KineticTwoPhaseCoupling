from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
import trimesh

from sim.math3d import quat_to_matrix, transform_points


class OccupancyShape(Protocol):
    def signed_distance(self, point: np.ndarray) -> float:
        ...

    def surface_velocity(self, point: np.ndarray) -> np.ndarray:
        ...


@lru_cache(maxsize=1)
def _base_icosahedron() -> trimesh.Trimesh:
    return trimesh.creation.icosahedron()


@lru_cache(maxsize=1)
def _base_plane_mesh() -> trimesh.Trimesh:
    points = np.array(
        [
            [1.15, 0.0, 0.0],
            [0.45, 0.12, 0.12],
            [0.45, 0.12, -0.12],
            [0.45, -0.12, 0.12],
            [0.45, -0.12, -0.12],
            [-0.85, 0.10, 0.10],
            [-0.85, 0.10, -0.10],
            [-0.85, -0.10, 0.10],
            [-0.85, -0.10, -0.10],
            [-1.18, 0.0, 0.0],
            [0.02, 0.0, 1.25],
            [0.02, 0.0, -1.25],
            [-0.68, 0.0, 0.42],
            [-0.68, 0.0, -0.42],
            [-0.76, 0.55, 0.0],
            [-0.98, 0.06, 0.0],
            [0.0, -0.18, 0.0],
        ],
        dtype=np.float64,
    )
    return trimesh.convex.convex_hull(points)


def _planes_from_mesh(mesh: trimesh.Trimesh, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = []
    offsets = []
    for face in mesh.faces:
        tri = vertices[face]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal /= max(np.linalg.norm(normal), 1e-8)
        offset = float(np.dot(normal, tri[0]))
        normals.append(normal)
        offsets.append(offset)
    return np.asarray(normals, dtype=np.float64), np.asarray(offsets, dtype=np.float64)


class ConvexShapeMixin:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    local_vertices: np.ndarray
    local_faces: np.ndarray
    face_normals: np.ndarray
    face_offsets: np.ndarray

    def _finalize_mesh(self, mesh: trimesh.Trimesh, local_vertices: np.ndarray) -> None:
        self.local_vertices = np.asarray(local_vertices, dtype=np.float64)
        self.local_faces = np.asarray(mesh.faces, dtype=np.int32)
        self.face_normals, self.face_offsets = _planes_from_mesh(mesh, self.local_vertices)
        self.bounding_radius = float(np.max(np.linalg.norm(self.local_vertices, axis=1)))

    def world_vertices(self) -> np.ndarray:
        return transform_points(self.orientation, self.position, self.local_vertices)

    def rotation_matrix(self) -> np.ndarray:
        return quat_to_matrix(self.orientation)

    def signed_distance(self, point: np.ndarray) -> float:
        rot = self.rotation_matrix()
        local = rot.T @ (np.asarray(point, dtype=np.float64) - self.position)
        return float(np.max(self.face_normals @ local - self.face_offsets))

    def surface_velocity(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        return self.linear_velocity + np.cross(self.angular_velocity, p - self.position)

    def transformed_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.Trimesh(vertices=self.local_vertices.copy(), faces=self.local_faces.copy(), process=False)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation_matrix()
        transform[:3, 3] = self.position
        mesh.apply_transform(transform)
        return mesh


@dataclass
class D20Shape(ConvexShapeMixin):
    radius: float
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        self.linear_velocity = np.asarray(self.linear_velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        base = _base_icosahedron()
        raw_vertices = base.vertices.astype(np.float64)
        scale = self.radius / float(np.max(np.linalg.norm(raw_vertices, axis=1)))
        self._finalize_mesh(base, raw_vertices * scale)


@dataclass
class PlaneShape(ConvexShapeMixin):
    length: float
    height: float
    span: float
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        self.linear_velocity = np.asarray(self.linear_velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        base = _base_plane_mesh()
        raw_vertices = base.vertices.astype(np.float64).copy()
        scaled = raw_vertices * np.array([self.length * 0.5, self.height * 0.5, self.span * 0.5], dtype=np.float64)
        self._finalize_mesh(base, scaled)


def make_body_shape(
    shape_kind: str,
    shape_size: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> ConvexShapeMixin:
    if shape_kind == "d20":
        return D20Shape(
            radius=float(shape_size[0]),
            position=position,
            orientation=orientation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
    if shape_kind == "plane":
        return PlaneShape(
            length=float(shape_size[0]),
            height=float(shape_size[1]),
            span=float(shape_size[2]),
            position=position,
            orientation=orientation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
    raise ValueError(f"Unsupported shape kind: {shape_kind}")



@dataclass(frozen=True)
class TankShape:
    extent: tuple[float, float, float]

    def interior_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3, dtype=np.float64), np.asarray(self.extent, dtype=np.float64)
