from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sim.geometry import TankShape, make_body_shape
from sim.math3d import integrate_quaternion, quat_normalize


@dataclass
class RigidBodyState:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    mass: float
    inv_mass: float
    inertia_body: np.ndarray
    inertia_body_inv: np.ndarray
    radius: float
    shape_kind: str
    shape_size: np.ndarray

    @classmethod
    def create_d20(cls, radius: float, density: float, position: tuple[float, float, float], orientation: tuple[float, float, float, float]) -> "RigidBodyState":
        # Icosahedron is isotropic enough that a diagonal approximation works well for this prototype.
        volume = 2.53615071012041 * (radius**3)
        mass = max(0.05, volume * density * 0.012)
        inertia_scalar = 0.4 * mass * (radius**2)
        inertia_body = np.eye(3, dtype=np.float64) * inertia_scalar
        inertia_body_inv = np.eye(3, dtype=np.float64) / inertia_scalar
        return cls(
            position=np.asarray(position, dtype=np.float64),
            orientation=quat_normalize(orientation),
            linear_velocity=np.zeros(3, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            mass=mass,
            inv_mass=1.0 / mass,
            inertia_body=inertia_body,
            inertia_body_inv=inertia_body_inv,
            radius=radius,
            shape_kind="d20",
            shape_size=np.array([radius, radius, radius], dtype=np.float64),
        )

    @classmethod
    def create_plane(
        cls,
        length: float,
        height: float,
        span: float,
        density: float,
        position: Iterable[float],
        orientation: Iterable[float],
    ) -> "RigidBodyState":
        volume = max(length * height * span * 0.22, 1e-3)
        mass = max(0.05, volume * density * 0.006)
        radius = 0.5 * float(np.linalg.norm([length, height, span]))
        inertia_scalar = 0.4 * mass * (radius**2)
        inertia_body = np.eye(3, dtype=np.float64) * inertia_scalar
        inertia_body_inv = np.eye(3, dtype=np.float64) / inertia_scalar
        return cls(
            position=np.asarray(position, dtype=np.float64),
            orientation=quat_normalize(orientation),
            linear_velocity=np.zeros(3, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            mass=mass,
            inv_mass=1.0 / mass,
            inertia_body=inertia_body,
            inertia_body_inv=inertia_body_inv,
            radius=radius,
            shape_kind="plane",
            shape_size=np.array([length, height, span], dtype=np.float64),
        )

    @classmethod
    def create_body(
        cls,
        shape_kind: str,
        shape_size: Iterable[float],
        density: float,
        position: Iterable[float],
        orientation: Iterable[float],
    ) -> "RigidBodyState":
        shape = tuple(float(v) for v in shape_size)
        if shape_kind == "d20":
            return cls.create_d20(radius=shape[0], density=density, position=tuple(position), orientation=tuple(orientation))
        if shape_kind == "plane":
            return cls.create_plane(length=shape[0], height=shape[1], span=shape[2], density=density, position=position, orientation=orientation)
        raise ValueError(f"Unsupported shape kind: {shape_kind}")


class RigidBodySolver:
    def __init__(self, tank: TankShape, restitution: float = 0.18, friction: float = 0.92):
        self.tank = tank
        self.restitution = restitution
        self.friction = friction

    def make_shape(self, state: RigidBodyState):
        return make_body_shape(
            shape_kind=state.shape_kind,
            shape_size=state.shape_size,
            position=state.position.copy(),
            orientation=state.orientation.copy(),
            linear_velocity=state.linear_velocity.copy(),
            angular_velocity=state.angular_velocity.copy(),
        )

    def integrate(self, state: RigidBodyState, force: np.ndarray, torque: np.ndarray, gravity: np.ndarray, dt: float) -> None:
        force_total = np.asarray(force, dtype=np.float64) + np.asarray(gravity, dtype=np.float64) * state.mass
        torque_total = np.asarray(torque, dtype=np.float64)
        state.linear_velocity = state.linear_velocity + dt * force_total * state.inv_mass
        state.position = state.position + dt * state.linear_velocity
        angular_acc = state.inertia_body_inv @ torque_total
        state.angular_velocity = state.angular_velocity + dt * angular_acc
        state.orientation = integrate_quaternion(state.orientation, state.angular_velocity, dt)
        self.resolve_tank_contacts(state)

    def resolve_tank_contacts(self, state: RigidBodyState) -> None:
        shape = self.make_shape(state)
        vertices = shape.world_vertices()
        min_bound, max_bound = self.tank.interior_bounds()
        correction = np.zeros(3, dtype=np.float64)
        correction[0] += max(0.0, min_bound[0] - float(vertices[:, 0].min()))
        correction[0] -= max(0.0, float(vertices[:, 0].max()) - max_bound[0])
        correction[1] += max(0.0, min_bound[1] - float(vertices[:, 1].min()))
        correction[1] -= max(0.0, float(vertices[:, 1].max()) - max_bound[1])
        correction[2] += max(0.0, min_bound[2] - float(vertices[:, 2].min()))
        correction[2] -= max(0.0, float(vertices[:, 2].max()) - max_bound[2])
        if np.linalg.norm(correction) > 0.0:
            state.position = state.position + correction
            shape = self.make_shape(state)
            vertices = shape.world_vertices()
        contacts: list[tuple[np.ndarray, np.ndarray, float]] = []
        axis_normals = (
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        )
        for vertex in vertices:
            if vertex[0] < min_bound[0]:
                contacts.append((vertex, axis_normals[0], min_bound[0] - vertex[0]))
            if vertex[0] > max_bound[0]:
                contacts.append((vertex, axis_normals[1], vertex[0] - max_bound[0]))
            if vertex[1] < min_bound[1]:
                contacts.append((vertex, axis_normals[2], min_bound[1] - vertex[1]))
            if vertex[1] > max_bound[1]:
                contacts.append((vertex, axis_normals[3], vertex[1] - max_bound[1]))
            if vertex[2] < min_bound[2]:
                contacts.append((vertex, axis_normals[4], min_bound[2] - vertex[2]))
            if vertex[2] > max_bound[2]:
                contacts.append((vertex, axis_normals[5], vertex[2] - max_bound[2]))

        if not contacts:
            return

        deepest = max(contacts, key=lambda item: item[2])
        contact_point, normal, depth = deepest
        state.position = state.position + normal * depth
        r = contact_point - state.position
        vel_at_contact = state.linear_velocity + np.cross(state.angular_velocity, r)
        normal_speed = float(np.dot(vel_at_contact, normal))
        if normal_speed < 0.0:
            impulse_mag = -(1.0 + self.restitution) * normal_speed
            denom = state.inv_mass
            cross_term = np.cross(r, normal)
            denom += float(normal @ np.cross(state.inertia_body_inv @ cross_term, r))
            impulse_mag /= max(denom, 1e-6)
            impulse = normal * impulse_mag
            state.linear_velocity = state.linear_velocity + impulse * state.inv_mass
            state.angular_velocity = state.angular_velocity + state.inertia_body_inv @ np.cross(r, impulse)

        tangential = state.linear_velocity - normal * np.dot(state.linear_velocity, normal)
        state.linear_velocity = tangential * self.friction + normal * np.dot(state.linear_velocity, normal)
        state.angular_velocity *= 0.985
