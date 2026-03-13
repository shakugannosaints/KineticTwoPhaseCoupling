from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import taichi as ti

from sim.config import ScenePreset
from sim.rigid import RigidBodyState
from sim.solver.lattices import FLOW_C, FLOW_OPP, FLOW_W, PHASE_C, PHASE_OPP, PHASE_W

Q_FLOW = len(FLOW_C)
Q_PHASE = len(PHASE_C)
MAX_FACE_COUNT = 64


@dataclass
class SolverStats:
    steps: int = 0


@ti.data_oriented
class DualGridLBMSolver:
    def __init__(self, preset: ScenePreset):
        self.preset = preset
        self.nx, self.ny, self.nz = preset.flow_resolution
        self.phase_ratio = preset.phase_ratio
        self.fx = self.nx * self.phase_ratio
        self.fy = self.ny * self.phase_ratio
        self.fz = self.nz * self.phase_ratio
        self.gravity = ti.Vector(list(preset.gravity))
        self.density_ratio = float(preset.density_ratio)
        self.nu_water = float(preset.water_viscosity)
        self.nu_air = float(preset.air_viscosity)
        self.surface_tension = float(preset.surface_tension)
        self.force_scale = float(preset.force_scale)
        self.torque_scale = float(preset.torque_scale)
        self.mobility = 0.12
        self.sharpen_strength = 1.4
        self.stats = SolverStats()

        self.g_src = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz, Q_FLOW))
        self.g_dst = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz, Q_FLOW))
        self.h_src = ti.field(dtype=ti.f32, shape=(self.fx, self.fy, self.fz, Q_PHASE))
        self.h_dst = ti.field(dtype=ti.f32, shape=(self.fx, self.fy, self.fz, Q_PHASE))

        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        self.velocity_prev = ti.Vector.field(3, dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        self.grad_coarse = ti.Vector.field(3, dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        self.grad_fine = ti.Vector.field(3, dtype=ti.f32, shape=(self.fx, self.fy, self.fz))
        self.normal_fine = ti.Vector.field(3, dtype=ti.f32, shape=(self.fx, self.fy, self.fz))
        self.phi_fine = ti.field(dtype=ti.f32, shape=(self.fx, self.fy, self.fz))
        self.phi_coarse = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        self.rho = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        self.nu = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))

        self.coarse_solid = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
        self.fine_solid = ti.field(dtype=ti.i32, shape=(self.fx, self.fy, self.fz))
        self.fluid_force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fluid_torque = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.body_position = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.body_orientation = ti.Vector.field(4, dtype=ti.f32, shape=())
        self.body_linear_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.body_angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.body_radius = ti.field(dtype=ti.f32, shape=())

        self.face_normals = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FACE_COUNT)
        self.face_offsets = ti.field(dtype=ti.f32, shape=MAX_FACE_COUNT)
        self.face_count = ti.field(dtype=ti.i32, shape=())
        self.phase_blend = ti.field(dtype=ti.f32, shape=())
        self.normal_source_flag = ti.field(dtype=ti.i32, shape=())
        self.fill_ratio_field = ti.field(dtype=ti.f32, shape=())
        self.flow_c_field = ti.Vector.field(3, dtype=ti.i32, shape=Q_FLOW)
        self.flow_w_field = ti.field(dtype=ti.f32, shape=Q_FLOW)
        self.flow_opp_field = ti.field(dtype=ti.i32, shape=Q_FLOW)
        self.phase_c_field = ti.Vector.field(3, dtype=ti.i32, shape=Q_PHASE)
        self.phase_w_field = ti.field(dtype=ti.f32, shape=Q_PHASE)
        self.phase_opp_field = ti.field(dtype=ti.i32, shape=Q_PHASE)
        self.flow_c_field.from_numpy(np.asarray(FLOW_C, dtype=np.int32))
        self.flow_w_field.from_numpy(np.asarray(FLOW_W, dtype=np.float32))
        self.flow_opp_field.from_numpy(np.asarray(FLOW_OPP, dtype=np.int32))
        self.phase_c_field.from_numpy(np.asarray(PHASE_C, dtype=np.int32))
        self.phase_w_field.from_numpy(np.asarray(PHASE_W, dtype=np.float32))
        self.phase_opp_field.from_numpy(np.asarray(PHASE_OPP, dtype=np.int32))
        self.target_phase_mass = ti.field(dtype=ti.f32, shape=())
        self.current_phase_mass = ti.field(dtype=ti.f32, shape=())
        self.active_phase_cells = ti.field(dtype=ti.f32, shape=())
        self.phase_mass_correction = ti.field(dtype=ti.f32, shape=())

    def configure_geometry(self, state: RigidBodyState, face_normals: np.ndarray, face_offsets: np.ndarray) -> None:
        self.set_body_state(state)
        self.body_radius[None] = float(state.radius)
        if len(face_normals) > MAX_FACE_COUNT:
            raise ValueError(f"Shape has {len(face_normals)} faces, exceeds max {MAX_FACE_COUNT}.")
        padded_normals = np.zeros((MAX_FACE_COUNT, 3), dtype=np.float32)
        padded_offsets = np.zeros((MAX_FACE_COUNT,), dtype=np.float32)
        padded_normals[: len(face_normals)] = face_normals.astype(np.float32)
        padded_offsets[: len(face_offsets)] = face_offsets.astype(np.float32)
        self.face_normals.from_numpy(padded_normals)
        self.face_offsets.from_numpy(padded_offsets)
        self.face_count[None] = int(len(face_normals))
        self.reset()

    def set_body_state(self, state: RigidBodyState) -> None:
        self.body_position[None] = tuple(float(v) for v in state.position)
        self.body_orientation[None] = tuple(float(v) for v in state.orientation)
        self.body_linear_velocity[None] = tuple(float(v) for v in state.linear_velocity)
        self.body_angular_velocity[None] = tuple(float(v) for v in state.angular_velocity)
        self.body_radius[None] = float(state.radius)

    @ti.func
    def coarse_pos(self, i, j, k):
        return ti.Vector([ti.cast(i, ti.f32) + 0.5, ti.cast(j, ti.f32) + 0.5, ti.cast(k, ti.f32) + 0.5])

    @ti.func
    def fine_pos(self, i, j, k):
        scale = 1.0 / ti.cast(self.phase_ratio, ti.f32)
        return ti.Vector(
            [
                (ti.cast(i, ti.f32) + 0.5) * scale,
                (ti.cast(j, ti.f32) + 0.5) * scale,
                (ti.cast(k, ti.f32) + 0.5) * scale,
            ]
        )

    @ti.func
    def flow_vec(self, q):
        return ti.cast(self.flow_c_field[q], ti.f32)

    @ti.func
    def phase_vec(self, q):
        return ti.cast(self.phase_c_field[q], ti.f32)

    @ti.func
    def qmat(self, q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return ti.Matrix(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )

    @ti.func
    def body_signed_distance(self, pos):
        rot = self.qmat(self.body_orientation[None])
        local = rot.transpose() @ (pos - self.body_position[None])
        dist = -1e6
        for f in range(self.face_count[None]):
            d = self.face_normals[f].dot(local) - self.face_offsets[f]
            dist = ti.max(dist, d)
        return dist

    @ti.func
    def body_velocity(self, pos):
        return self.body_linear_velocity[None] + self.body_angular_velocity[None].cross(pos - self.body_position[None])

    @ti.func
    def clamp_index(self, value, upper):
        return ti.max(0, ti.min(upper, value))

    @ti.func
    def safe_phi_fine(self, i, j, k):
        ii = self.clamp_index(i, self.fx - 1)
        jj = self.clamp_index(j, self.fy - 1)
        kk = self.clamp_index(k, self.fz - 1)
        return self.phi_fine[ii, jj, kk]

    @ti.func
    def safe_phi_coarse(self, i, j, k):
        ii = self.clamp_index(i, self.nx - 1)
        jj = self.clamp_index(j, self.ny - 1)
        kk = self.clamp_index(k, self.nz - 1)
        return self.phi_coarse[ii, jj, kk]

    @ti.func
    def normalize_vec(self, v):
        n = ti.sqrt(v.dot(v))
        out = ti.Vector([0.0, 0.0, 0.0])
        if n > 1e-6:
            out = v / n
        return out

    @ti.func
    def phase_eq(self, q, phi, u):
        return self.phase_w_field[q] * phi * (1.0 + 3.0 * self.phase_vec(q).dot(u))

    @ti.func
    def flow_eq(self, q, rho, u):
        c = self.flow_vec(q)
        cu = c.dot(u)
        uu = u.dot(u)
        return self.flow_w_field[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)

    @ti.func
    def blended_velocity(self, i, j, k, blend):
        x = (ti.cast(i, ti.f32) + 0.5) / ti.cast(self.phase_ratio, ti.f32) - 0.5
        y = (ti.cast(j, ti.f32) + 0.5) / ti.cast(self.phase_ratio, ti.f32) - 0.5
        z = (ti.cast(k, ti.f32) + 0.5) / ti.cast(self.phase_ratio, ti.f32) - 0.5
        bx = self.clamp_index(ti.floor(x, ti.i32), self.nx - 2)
        by = self.clamp_index(ti.floor(y, ti.i32), self.ny - 2)
        bz = self.clamp_index(ti.floor(z, ti.i32), self.nz - 2)
        fx = x - ti.cast(bx, ti.f32)
        fy = y - ti.cast(by, ti.f32)
        fz = z - ti.cast(bz, ti.f32)
        out = ti.Vector([0.0, 0.0, 0.0])
        for dx, dy, dz in ti.static(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))):
            wx = fx if dx == 1 else 1.0 - fx
            wy = fy if dy == 1 else 1.0 - fy
            wz = fz if dz == 1 else 1.0 - fz
            vel = (1.0 - blend) * self.velocity_prev[bx + dx, by + dy, bz + dz] + blend * self.velocity[bx + dx, by + dy, bz + dz]
            out += vel * wx * wy * wz
        return out

    @ti.func
    def sample_phi_to_coarse(self, i, j, k):
        base_i = i * self.phase_ratio
        base_j = j * self.phase_ratio
        base_k = k * self.phase_ratio
        total = 0.0
        for dx, dy, dz in ti.static(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))):
            total += self.phi_fine[base_i + dx, base_j + dy, base_k + dz]
        return total * 0.125

    @ti.func
    def fine_grad(self, i, j, k):
        return 0.5 * ti.Vector(
            [
                self.safe_phi_fine(i + 1, j, k) - self.safe_phi_fine(i - 1, j, k),
                self.safe_phi_fine(i, j + 1, k) - self.safe_phi_fine(i, j - 1, k),
                self.safe_phi_fine(i, j, k + 1) - self.safe_phi_fine(i, j, k - 1),
            ]
        )

    @ti.func
    def fine_laplacian(self, i, j, k):
        return (
            self.safe_phi_fine(i + 1, j, k)
            + self.safe_phi_fine(i - 1, j, k)
            + self.safe_phi_fine(i, j + 1, k)
            + self.safe_phi_fine(i, j - 1, k)
            + self.safe_phi_fine(i, j, k + 1)
            + self.safe_phi_fine(i, j, k - 1)
            - 6.0 * self.safe_phi_fine(i, j, k)
        )

    @ti.func
    def coarse_grad_from_phi(self, i, j, k):
        return 0.5 * ti.Vector(
            [
                self.safe_phi_coarse(i + 1, j, k) - self.safe_phi_coarse(i - 1, j, k),
                self.safe_phi_coarse(i, j + 1, k) - self.safe_phi_coarse(i, j - 1, k),
                self.safe_phi_coarse(i, j, k + 1) - self.safe_phi_coarse(i, j, k - 1),
            ]
        )

    @ti.func
    def coarse_grad_from_fine(self, i, j, k):
        base_i = i * self.phase_ratio
        base_j = j * self.phase_ratio
        base_k = k * self.phase_ratio
        out = ti.Vector([0.0, 0.0, 0.0])
        for dx, dy, dz in ti.static(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))):
            out += self.grad_fine[base_i + dx, base_j + dy, base_k + dz]
        return out * 0.125

    @ti.func
    def near_solid(self, i, j, k):
        flag = 0
        if self.coarse_solid[i, j, k] == 1:
            flag = 1
        for dx, dy, dz in ti.static(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
            ni = i + dx
            nj = j + dy
            nk = k + dz
            if ni < 0 or nj < 0 or nk < 0 or ni >= self.nx or nk >= self.nz:
                flag = 1
            elif nj < self.ny and self.coarse_solid[self.clamp_index(ni, self.nx - 1), self.clamp_index(nj, self.ny - 1), self.clamp_index(nk, self.nz - 1)] == 1:
                flag = 1
        return flag

    @ti.func
    def tau_at(self, i, j, k):
        return 3.0 * self.nu[i, j, k] + 0.5

    @ti.func
    def external_force(self, i, j, k):
        phi = self.phi_coarse[i, j, k]
        force = self.gravity * phi
        force += self.surface_tension * self.grad_coarse[i, j, k] * phi * (1.0 - phi)
        force += -0.01 * self.velocity[i, j, k]
        if j > self.ny - 4:
            force += -0.03 * self.velocity[i, j, k]
        return force

    @ti.func
    def filtered_density(self, i, j, k):
        rho0 = self.rho[i, j, k]
        phi0 = self.phi_coarse[i, j, k]
        total = 0.0
        for q in range(Q_FLOW):
            c = self.flow_vec(q)
            ni = self.clamp_index(i + ti.cast(c[0], ti.i32), self.nx - 1)
            nj = self.clamp_index(j + ti.cast(c[1], ti.i32), self.ny - 1)
            nk = self.clamp_index(k + ti.cast(c[2], ti.i32), self.nz - 1)
            blend = 1.0
            if ti.abs(self.phi_coarse[ni, nj, nk] - phi0) > 0.25:
                blend = 0.0
            total += self.flow_w_field[q] * (blend * self.rho[ni, nj, nk] + (1.0 - blend) * rho0)
        return total

    @ti.func
    def filtered_pressure(self, i, j, k):
        total = 0.0
        for q in range(Q_FLOW):
            c = self.flow_vec(q)
            ni = self.clamp_index(i + ti.cast(c[0], ti.i32), self.nx - 1)
            nj = self.clamp_index(j + ti.cast(c[1], ti.i32), self.ny - 1)
            nk = self.clamp_index(k + ti.cast(c[2], ti.i32), self.nz - 1)
            total += self.flow_w_field[q] * self.rho[ni, nj, nk]
        return total

    @ti.func
    def post_collision_value(self, i, j, k, q):
        rho = self.rho[i, j, k]
        u = self.velocity[i, j, k]
        tau = self.tau_at(i, j, k)
        omega = 1.0 / tau
        dist = self.g_src[i, j, k, q]
        eq = self.flow_eq(q, rho, u)
        force_term = self.flow_force_term(q, u, self.external_force(i, j, k))
        return dist - omega * (dist - eq) + force_term

    @ti.func
    def flow_force_term(self, q, u, force):
        c = self.flow_vec(q)
        cu = c.dot(u)
        return self.flow_w_field[q] * (3.0 * (c - u).dot(force) + 9.0 * cu * c.dot(force))

    @ti.func
    def hmbb(self, i, j, k, q, moving_body):
        opp = self.flow_opp_field[q]
        pos = self.coarse_pos(i, j, k)
        u_s = ti.Vector([0.0, 0.0, 0.0])
        if moving_body == 1:
            u_s = self.body_velocity(pos)
        rho_bar = self.filtered_pressure(i, j, k)
        alpha = 0.1
        incoming = (1.0 - alpha) * self.post_collision_value(i, j, k, opp)
        incoming += alpha * self.flow_eq(opp, rho_bar, u_s)
        incoming += 6.0 * self.flow_w_field[opp] * u_s.dot(self.flow_vec(q))
        return incoming

    @ti.kernel
    def clear_force_accumulators(self):
        self.fluid_force[None] = ti.Vector([0.0, 0.0, 0.0])
        self.fluid_torque[None] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def clear_phase_mass_stats(self):
        self.current_phase_mass[None] = 0.0
        self.active_phase_cells[None] = 0.0

    @ti.kernel
    def copy_velocity_to_prev(self):
        for i, j, k in self.velocity:
            self.velocity_prev[i, j, k] = self.velocity[i, j, k]

    @ti.kernel
    def update_solid_masks(self):
        for i, j, k in self.coarse_solid:
            self.coarse_solid[i, j, k] = 1 if self.body_signed_distance(self.coarse_pos(i, j, k)) <= 0.0 else 0
        for i, j, k in self.fine_solid:
            solid = 1 if self.body_signed_distance(self.fine_pos(i, j, k)) <= 0.0 else 0
            self.fine_solid[i, j, k] = solid
            if solid == 1:
                self.phi_fine[i, j, k] = 0.0
                for q in range(Q_PHASE):
                    self.h_src[i, j, k, q] = 0.0
                    self.h_dst[i, j, k, q] = 0.0

    @ti.kernel
    def initialize_phase(self):
        waterline = self.fill_ratio_field[None] * ti.cast(self.fy, ti.f32)
        for i, j, k in self.phi_fine:
            phi = 1.0 if ti.cast(j, ti.f32) < waterline else 0.0
            if self.fine_solid[i, j, k] == 1:
                phi = 0.0
            self.phi_fine[i, j, k] = phi
            for q in range(Q_PHASE):
                self.h_src[i, j, k, q] = self.phase_eq(q, phi, ti.Vector([0.0, 0.0, 0.0]))
                self.h_dst[i, j, k, q] = self.h_src[i, j, k, q]

    @ti.kernel
    def sample_phase_and_density(self):
        for i, j, k in self.phi_coarse:
            phi = self.sample_phi_to_coarse(i, j, k)
            self.phi_coarse[i, j, k] = phi
            self.rho[i, j, k] = 1.0 + (self.density_ratio - 1.0) * phi
            inv_nu = (1.0 / self.nu_air) + ((1.0 / self.nu_water) - (1.0 / self.nu_air)) * phi
            self.nu[i, j, k] = 1.0 / inv_nu

    @ti.kernel
    def initialize_flow(self):
        zero = ti.Vector([0.0, 0.0, 0.0])
        for i, j, k in self.velocity:
            if self.coarse_solid[i, j, k] == 1:
                self.velocity[i, j, k] = self.body_velocity(self.coarse_pos(i, j, k))
            else:
                self.velocity[i, j, k] = zero
            self.velocity_prev[i, j, k] = self.velocity[i, j, k]
            for q in range(Q_FLOW):
                value = self.flow_eq(q, self.rho[i, j, k], self.velocity[i, j, k])
                self.g_src[i, j, k, q] = value
                self.g_dst[i, j, k, q] = value

    @ti.kernel
    def compute_fine_normals(self):
        for i, j, k in self.grad_fine:
            if self.fine_solid[i, j, k] == 1:
                self.grad_fine[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.normal_fine[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            else:
                grad = self.fine_grad(i, j, k)
                moment = ti.Vector([0.0, 0.0, 0.0])
                for q in range(Q_PHASE):
                    h_val = self.h_dst[i, j, k, q] if self.normal_source_flag[None] == 1 else self.h_src[i, j, k, q]
                    moment += h_val * (-self.phase_vec(q))
                phi = self.phi_fine[i, j, k]
                mixed = (1.0 - phi) * self.normalize_vec(grad) + phi * self.normalize_vec(moment)
                self.grad_fine[i, j, k] = grad
                self.normal_fine[i, j, k] = self.normalize_vec(mixed)

    @ti.kernel
    def compute_coarse_gradients(self):
        for i, j, k in self.grad_coarse:
            grad_h = self.coarse_grad_from_fine(i, j, k)
            grad_g = self.coarse_grad_from_phi(i, j, k)
            phi = self.phi_coarse[i, j, k]
            if self.near_solid(i, j, k) == 1:
                self.grad_coarse[i, j, k] = grad_h
            else:
                self.grad_coarse[i, j, k] = (1.0 - phi) * grad_h + phi * grad_g

    @ti.kernel
    def flow_stream_step(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            pos = self.coarse_pos(i, j, k)
            if self.coarse_solid[i, j, k] == 1:
                vel = self.body_velocity(pos)
                self.velocity[i, j, k] = vel
                for q in range(Q_FLOW):
                    self.g_dst[i, j, k, q] = self.flow_eq(q, self.rho[i, j, k], vel)
            else:
                for q in range(Q_FLOW):
                    c = self.flow_vec(q)
                    sx = i - ti.cast(c[0], ti.i32)
                    sy = j - ti.cast(c[1], ti.i32)
                    sz = k - ti.cast(c[2], ti.i32)
                    incoming = 0.0
                    if 0 <= sx < self.nx and 0 <= sy < self.ny and 0 <= sz < self.nz and self.coarse_solid[sx, sy, sz] == 0:
                        rho_s = self.rho[sx, sy, sz]
                        u_s = self.velocity[sx, sy, sz]
                        tau = self.tau_at(sx, sy, sz)
                        omega = 1.0 / tau
                        dist = self.g_src[sx, sy, sz, q]
                        eq = self.flow_eq(q, rho_s, u_s)
                        incoming = dist - omega * (dist - eq) + self.flow_force_term(q, u_s, self.external_force(sx, sy, sz))
                    else:
                        if sy >= self.ny:
                            incoming = self.g_src[i, j, k, q]
                        else:
                            moving = 1 if (0 <= sx < self.nx and 0 <= sy < self.ny and 0 <= sz < self.nz) else 0
                            incoming = self.hmbb(i, j, k, q, moving)
                            if moving == 1:
                                rho_bar = self.filtered_density(i, j, k)
                                opp = self.flow_opp_field[q]
                                delta = rho_bar * (self.post_collision_value(i, j, k, opp) + incoming) * self.flow_vec(opp)
                                ti.atomic_add(self.fluid_force[None][0], delta[0] * self.force_scale)
                                ti.atomic_add(self.fluid_force[None][1], delta[1] * self.force_scale)
                                ti.atomic_add(self.fluid_force[None][2], delta[2] * self.force_scale)
                                arm = pos - self.body_position[None]
                                torque = arm.cross(delta) * self.torque_scale
                                ti.atomic_add(self.fluid_torque[None][0], torque[0])
                                ti.atomic_add(self.fluid_torque[None][1], torque[1])
                                ti.atomic_add(self.fluid_torque[None][2], torque[2])
                    self.g_dst[i, j, k, q] = incoming

    @ti.kernel
    def update_flow_macros(self):
        for i, j, k in self.velocity:
            if self.coarse_solid[i, j, k] == 1:
                self.velocity[i, j, k] = self.body_velocity(self.coarse_pos(i, j, k))
            else:
                rho = self.rho[i, j, k]
                momentum = ti.Vector([0.0, 0.0, 0.0])
                for q in range(Q_FLOW):
                    momentum += self.g_dst[i, j, k, q] * self.flow_vec(q)
                vel = momentum / ti.max(rho, 1e-5)
                speed = ti.sqrt(vel.dot(vel))
                if speed > 0.18:
                    vel *= 0.18 / speed
                self.velocity[i, j, k] = vel

    @ti.kernel
    def phase_stream_step(self):
        tau = 0.82
        omega = 1.0 / tau
        for i, j, k in ti.ndrange(self.fx, self.fy, self.fz):
            if self.fine_solid[i, j, k] == 1:
                for q in range(Q_PHASE):
                    self.h_dst[i, j, k, q] = 0.0
            else:
                for q in range(Q_PHASE):
                    c = self.phase_vec(q)
                    sx = i - ti.cast(c[0], ti.i32)
                    sy = j - ti.cast(c[1], ti.i32)
                    sz = k - ti.cast(c[2], ti.i32)
                    incoming = 0.0
                    if 0 <= sx < self.fx and 0 <= sy < self.fy and 0 <= sz < self.fz and self.fine_solid[sx, sy, sz] == 0:
                        phi = self.phi_fine[sx, sy, sz]
                        u = self.blended_velocity(sx, sy, sz, self.phase_blend[None])
                        eq = self.phase_eq(q, phi, u)
                        dist = self.h_src[sx, sy, sz, q]
                        incoming = dist - omega * (dist - eq)
                    else:
                        if sy >= self.fy:
                            incoming = self.h_src[i, j, k, q]
                        else:
                            incoming = self.h_src[i, j, k, self.phase_opp_field[q]]
                    self.h_dst[i, j, k, q] = incoming

    @ti.kernel
    def reconstruct_phi_from_h(self):
        for i, j, k in self.phi_fine:
            if self.fine_solid[i, j, k] == 1:
                self.phi_fine[i, j, k] = 0.0
            else:
                total = 0.0
                for q in range(Q_PHASE):
                    total += self.h_dst[i, j, k, q]
                self.phi_fine[i, j, k] = ti.min(1.0, ti.max(0.0, total))

    @ti.kernel
    def sharpen_phase(self):
        for i, j, k in self.phi_fine:
            if self.fine_solid[i, j, k] == 0:
                phi = self.phi_fine[i, j, k]
                lap = self.fine_laplacian(i, j, k)
                correction = self.mobility * (lap - self.sharpen_strength * phi * (1.0 - phi) * (1.0 - 2.0 * phi))
                phi = ti.min(1.0, ti.max(0.0, phi + correction))
                self.phi_fine[i, j, k] = phi
                u = self.blended_velocity(i, j, k, self.phase_blend[None])
                for q in range(Q_PHASE):
                    self.h_dst[i, j, k, q] = self.phase_eq(q, phi, u)

    @ti.kernel
    def swap_phase_buffers(self):
        for i, j, k, q in self.h_src:
            self.h_src[i, j, k, q] = self.h_dst[i, j, k, q]

    @ti.kernel
    def swap_flow_buffers(self):
        for i, j, k, q in self.g_src:
            self.g_src[i, j, k, q] = self.g_dst[i, j, k, q]

    @ti.kernel
    def accumulate_phase_mass(self):
        for i, j, k in self.phi_fine:
            if self.fine_solid[i, j, k] == 0:
                ti.atomic_add(self.current_phase_mass[None], self.phi_fine[i, j, k])
                ti.atomic_add(self.active_phase_cells[None], 1.0)

    @ti.kernel
    def apply_phase_mass_correction(self):
        for i, j, k in self.phi_fine:
            if self.fine_solid[i, j, k] == 0:
                phi = ti.min(1.0, ti.max(0.0, self.phi_fine[i, j, k] + self.phase_mass_correction[None]))
                self.phi_fine[i, j, k] = phi
                u = self.blended_velocity(i, j, k, 0.5)
                for q in range(Q_PHASE):
                    value = self.phase_eq(q, phi, u)
                    self.h_src[i, j, k, q] = value
                    self.h_dst[i, j, k, q] = value

    def reset(self) -> None:
        self.clear_force_accumulators()
        self.update_solid_masks()
        self.fill_ratio_field[None] = self.preset.fill_ratio
        self.initialize_phase()
        self.clear_phase_mass_stats()
        self.accumulate_phase_mass()
        self.target_phase_mass[None] = self.current_phase_mass[None]
        self.sample_phase_and_density()
        self.initialize_flow()
        self.normal_source_flag[None] = 0
        self.compute_fine_normals()
        self.compute_coarse_gradients()
        self.stats.steps = 0

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        self.clear_force_accumulators()
        self.copy_velocity_to_prev()
        self.update_solid_masks()
        self.sample_phase_and_density()
        self.normal_source_flag[None] = 0
        self.compute_fine_normals()
        self.compute_coarse_gradients()
        self.flow_stream_step()
        self.update_flow_macros()
        self.swap_flow_buffers()
        self.phase_blend[None] = 0.0
        self.phase_stream_step()
        self.reconstruct_phi_from_h()
        self.normal_source_flag[None] = 1
        self.compute_fine_normals()
        self.sharpen_phase()
        self.swap_phase_buffers()
        self.phase_blend[None] = 0.5
        self.phase_stream_step()
        self.reconstruct_phi_from_h()
        self.normal_source_flag[None] = 1
        self.compute_fine_normals()
        self.sharpen_phase()
        self.swap_phase_buffers()
        self.clear_phase_mass_stats()
        self.accumulate_phase_mass()
        if self.active_phase_cells[None] > 0.0:
            raw_correction = (self.target_phase_mass[None] - self.current_phase_mass[None]) / self.active_phase_cells[None]
            correction = raw_correction * self.preset.phase_mass_correction_gain
            self.phase_mass_correction[None] = max(-0.05, min(0.05, correction))
            self.apply_phase_mass_correction()
        self.sample_phase_and_density()
        self.compute_coarse_gradients()
        self.stats.steps += 1
        return np.array(self.fluid_force[None], dtype=np.float64), np.array(self.fluid_torque[None], dtype=np.float64)

    def get_phase_field(self) -> np.ndarray:
        return self.phi_fine.to_numpy()

    def get_surface_field(self) -> np.ndarray:
        return self.phi_coarse.to_numpy()

    def get_velocity_field(self) -> np.ndarray:
        return self.velocity.to_numpy()
