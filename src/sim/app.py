from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np
import taichi as ti

from sim.config import BackendName, PresetName, QualityName, ScenePreset, available_preset_names, build_preset
from sim.geometry import TankShape, make_body_shape
from sim.mesh import extract_surface_mesh, open_tank_wireframe
from sim.math3d import quat_from_euler
from sim.rigid import RigidBodySolver, RigidBodyState
from sim.solver import DualGridLBMSolver


def init_taichi(backend: BackendName) -> None:
    arch = ti.cuda if backend == "cuda" else ti.cpu
    try:
        ti.init(arch=arch, default_fp=ti.f32, default_ip=ti.i32, offline_cache=True)
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, offline_cache=True)


def build_window_title(preset: ScenePreset) -> str:
    return (
        f"LD23 Simplified Reproduction | {preset.display_name} | "
        "[Space] Pause  [R] Reset  [Tab] Next Scene  [1] D20  [2] Plane"
    )


@dataclass
class SimulationApp:
    preset: ScenePreset

    def __post_init__(self) -> None:
        self.tank = TankShape(self.preset.tank_extent)
        self.rigid_solver = RigidBodySolver(self.tank)
        self.initial_state = self._create_initial_state()
        self.state = self._create_initial_state()
        body_shape = self.make_body_shape()
        self.solver = DualGridLBMSolver(self.preset)
        self.solver.configure_geometry(self.state, body_shape.face_normals, body_shape.face_offsets)
        self.wireframe = open_tank_wireframe(self.preset.tank_extent)
        self.surface_vertices = np.empty((0, 3), dtype=np.float32)
        self.surface_faces = np.empty((0, 3), dtype=np.int32)
        self.surface_normals = np.empty((0, 3), dtype=np.float32)
        self.frame_index = 0
        self.sim_time = 0.0
        self.force_ema = np.zeros(3, dtype=np.float64)
        self.torque_ema = np.zeros(3, dtype=np.float64)
        self.surface_height_hint = self.preset.fill_ratio * self.preset.flow_resolution[1]
        if self.preset.motion_mode == "scripted":
            self._apply_scripted_body(0.0)
            self.solver.set_body_state(self.state)
            self.solver.reset()
        self.refresh_surface(force=True)

    def _create_initial_state(self) -> RigidBodyState:
        return RigidBodyState.create_body(
            shape_kind=self.preset.body_kind,
            shape_size=self.preset.body_size,
            density=self.preset.body_density,
            position=self.preset.body_initial_pose[0],
            orientation=self.preset.body_initial_pose[1],
        )

    def make_body_shape(self):
        return make_body_shape(
            shape_kind=self.state.shape_kind,
            shape_size=self.state.shape_size,
            position=self.state.position,
            orientation=self.state.orientation,
            linear_velocity=self.state.linear_velocity,
            angular_velocity=self.state.angular_velocity,
        )

    def reset(self) -> None:
        self.state = self._create_initial_state()
        self.frame_index = 0
        self.sim_time = 0.0
        self.force_ema.fill(0.0)
        self.torque_ema.fill(0.0)
        self.surface_height_hint = self.preset.fill_ratio * self.preset.flow_resolution[1]
        if self.preset.motion_mode == "scripted":
            self._apply_scripted_body(0.0)
        self.solver.set_body_state(self.state)
        self.solver.reset()
        self.surface_vertices = np.empty((0, 3), dtype=np.float32)
        self.surface_faces = np.empty((0, 3), dtype=np.int32)
        self.surface_normals = np.empty((0, 3), dtype=np.float32)
        self.refresh_surface(force=True)

    def _apply_scripted_body(self, local_time: float) -> None:
        start = np.asarray(self.preset.body_initial_pose[0], dtype=np.float64)
        velocity = np.asarray(self.preset.scripted_linear_velocity, dtype=np.float64)
        pos = start + velocity * local_time
        vertical_omega = 1.15
        lateral_omega = 0.6
        roll_omega = 1.7
        roll = self.preset.scripted_roll_amplitude * np.sin(local_time * roll_omega)
        pos[1] = start[1] + self.preset.scripted_vertical_amplitude * np.sin(local_time * vertical_omega)
        pos[2] = start[2] + self.preset.scripted_lateral_amplitude * np.sin(local_time * lateral_omega)
        linear_velocity = velocity.copy()
        linear_velocity[1] = self.preset.scripted_vertical_amplitude * vertical_omega * np.cos(local_time * vertical_omega)
        linear_velocity[2] = self.preset.scripted_lateral_amplitude * lateral_omega * np.cos(local_time * lateral_omega)
        angular_velocity = np.array(
            [self.preset.scripted_roll_amplitude * roll_omega * np.cos(local_time * roll_omega), 0.0, 0.0],
            dtype=np.float64,
        )
        self.state.position = pos
        self.state.linear_velocity = linear_velocity
        self.state.angular_velocity = angular_velocity
        self.state.orientation = quat_from_euler(roll, self.preset.scripted_pitch, 0.0)

    def step(self) -> None:
        if self.preset.motion_mode == "scripted":
            if self.preset.scripted_duration > 0.0 and self.sim_time >= self.preset.scripted_duration:
                self.reset()
            self._apply_scripted_body(self.sim_time)
            self.solver.set_body_state(self.state)
            self.solver.step()
            self.sim_time += self.preset.rigid_dt
            self.frame_index += 1
            return

        self.solver.set_body_state(self.state)
        force, torque = self.solver.step()
        self.force_ema = 0.85 * self.force_ema + 0.15 * force
        self.torque_ema = 0.8 * self.torque_ema + 0.2 * torque
        stabilized_force = self.limit_fluid_force(self.force_ema)
        self.rigid_solver.integrate(self.state, stabilized_force, self.torque_ema, np.asarray(self.preset.gravity, dtype=np.float64), dt=self.preset.rigid_dt)
        self.sim_time += self.preset.rigid_dt
        self.frame_index += 1

    def estimate_submerged_fraction(self) -> float:
        surface_y = self.surface_height_hint
        bottom = float(self.state.position[1] - self.state.radius)
        top = float(self.state.position[1] + self.state.radius)
        height = max(top - bottom, 1e-6)
        return float(np.clip((surface_y - bottom) / height, 0.0, 1.0))

    def limit_fluid_force(self, force: np.ndarray) -> np.ndarray:
        limited = np.asarray(force, dtype=np.float64).copy()
        submerged_fraction = self.estimate_submerged_fraction()
        if submerged_fraction > 0.0 and limited[1] > 0.0:
            weight = abs(float(self.preset.gravity[1])) * self.state.mass
            upward_cap = weight * self.preset.max_buoyancy_ratio
            limited[1] = min(limited[1], upward_cap)
        return limited

    def refresh_surface(self, force: bool = False) -> None:
        if not force and self.frame_index % self.preset.render_extract_interval != 0:
            return
        cell_size = 1.0
        self.surface_vertices, self.surface_faces, self.surface_normals = extract_surface_mesh(self.solver.get_surface_field(), cell_size)
        if len(self.surface_vertices) > 0:
            self.surface_height_hint = float(np.quantile(self.surface_vertices[:, 1], 0.65))

    def body_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        mesh = self.make_body_shape().transformed_mesh()
        return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)

    def shutdown(self) -> None:
        self.surface_vertices = np.empty((0, 3), dtype=np.float32)
        self.surface_faces = np.empty((0, 3), dtype=np.int32)
        self.surface_normals = np.empty((0, 3), dtype=np.float32)
        self.wireframe = np.empty((0, 3), dtype=np.float32)
        self.solver = None
        gc.collect()


def run_headless(app: SimulationApp, steps: int) -> int:
    start = time.perf_counter()
    for _ in range(steps):
        app.step()
    elapsed = time.perf_counter() - start
    pos = np.asarray(app.state.position)
    print(f"Headless run completed in {elapsed:.2f}s, final position={pos.tolist()}")
    return 0


def run_interactive(initial_preset_name: PresetName, quality: QualityName, backend: BackendName) -> int:
    from sim.viewer import OpenGLViewer

    scene_names = available_preset_names()
    scene_index = scene_names.index(initial_preset_name)
    app = SimulationApp(build_preset(scene_names[scene_index], quality))
    viewer = OpenGLViewer(app.preset.tank_extent)
    viewer.configure_scene(build_window_title(app.preset), app.preset.tank_extent, app.preset.camera_target_factor, app.preset.camera_distance_factor)

    last_tick = time.perf_counter()
    sim_accumulator = 0.0
    last_space = False
    last_tab = False
    last_digit_1 = False
    last_digit_2 = False

    try:
        while not viewer.should_close():
            viewer.poll()
            if viewer.should_close():
                break

            import glfw

            current_space = glfw.get_key(viewer.window, glfw.KEY_SPACE) == glfw.PRESS
            if current_space and not last_space:
                viewer.toggle_pause()
            last_space = current_space

            requested_scene: PresetName | None = None
            current_tab = glfw.get_key(viewer.window, glfw.KEY_TAB) == glfw.PRESS
            if current_tab and not last_tab:
                scene_index = (scene_index + 1) % len(scene_names)
                requested_scene = scene_names[scene_index]
            last_tab = current_tab

            current_digit_1 = (glfw.get_key(viewer.window, glfw.KEY_1) == glfw.PRESS) or (glfw.get_key(viewer.window, glfw.KEY_KP_1) == glfw.PRESS)
            if current_digit_1 and not last_digit_1:
                scene_index = 0
                requested_scene = scene_names[scene_index]
            last_digit_1 = current_digit_1

            current_digit_2 = (glfw.get_key(viewer.window, glfw.KEY_2) == glfw.PRESS) or (glfw.get_key(viewer.window, glfw.KEY_KP_2) == glfw.PRESS)
            if current_digit_2 and not last_digit_2:
                scene_index = min(1, len(scene_names) - 1)
                requested_scene = scene_names[scene_index]
            last_digit_2 = current_digit_2

            if requested_scene is not None and requested_scene != app.preset.name:
                app.shutdown()
                ti.reset()
                init_taichi(backend)
                app = SimulationApp(build_preset(requested_scene, quality))
                viewer.configure_scene(build_window_title(app.preset), app.preset.tank_extent, app.preset.camera_target_factor, app.preset.camera_distance_factor)
                viewer.paused = False
                last_tick = time.perf_counter()
                sim_accumulator = 0.0
                continue

            if viewer.consume_reset():
                app.reset()
                viewer.configure_scene(build_window_title(app.preset), app.preset.tank_extent, app.preset.camera_target_factor, app.preset.camera_distance_factor)
                last_tick = time.perf_counter()
                sim_accumulator = 0.0

            if not viewer.paused:
                now = time.perf_counter()
                sim_accumulator += min(now - last_tick, 0.25) * app.preset.target_sim_hz
                last_tick = now
                simulated = False
                steps_taken = 0
                while sim_accumulator >= 1.0 and steps_taken < app.preset.fluid_steps_per_frame:
                    if viewer.should_close():
                        break
                    app.step()
                    sim_accumulator -= 1.0
                    steps_taken += 1
                    simulated = True
                if simulated:
                    app.refresh_surface()
            else:
                last_tick = time.perf_counter()

            body_vertices, body_faces = app.body_mesh()
            viewer.set_meshes(app.wireframe, app.surface_vertices, app.surface_faces, app.surface_normals, body_vertices, body_faces)
            viewer.render()
        return 0
    finally:
        app.shutdown()
        viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LD23 simplified two-phase fluid-solid coupling prototype.")
    parser.add_argument("--preset", choices=available_preset_names(), default="d20_drop")
    parser.add_argument("--quality", choices=("low", "default", "high"), default="default")
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    init_taichi(args.backend)
    try:
        if args.headless:
            app = SimulationApp(build_preset(args.preset, args.quality))
            try:
                return run_headless(app, args.steps)
            finally:
                app.shutdown()
        return run_interactive(args.preset, args.quality, args.backend)
    finally:
        ti.reset()


if __name__ == "__main__":
    raise SystemExit(main())
