from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sim.math3d import quat_from_euler


BackendName = Literal["cuda", "cpu"]
QualityName = Literal["low", "default", "high"]
PresetName = Literal["d20_drop", "plane_skim"]
BodyKind = Literal["d20", "plane"]
MotionMode = Literal["dynamic", "scripted"]


@dataclass(frozen=True)
class ScenePreset:
    name: PresetName
    display_name: str
    tank_extent: tuple[float, float, float]
    fill_ratio: float
    body_kind: BodyKind
    body_density: float
    body_size: tuple[float, float, float]
    body_initial_pose: tuple[tuple[float, float, float], tuple[float, float, float, float]]
    motion_mode: MotionMode
    scripted_linear_velocity: tuple[float, float, float]
    scripted_vertical_amplitude: float
    scripted_lateral_amplitude: float
    scripted_roll_amplitude: float
    scripted_pitch: float
    scripted_duration: float
    gravity: tuple[float, float, float]
    flow_resolution: tuple[int, int, int]
    phase_ratio: int
    render_extract_interval: int
    fluid_steps_per_frame: int
    fluid_substeps: int
    rigid_dt: float
    target_sim_hz: float
    max_buoyancy_ratio: float
    phase_mass_correction_gain: float
    density_ratio: float
    water_viscosity: float
    air_viscosity: float
    surface_tension: float
    force_scale: float
    torque_scale: float
    camera_target_factor: tuple[float, float, float]
    camera_distance_factor: float


QUALITY_PARAMS: dict[QualityName, dict[str, float | int]] = {
    "low": {"render_extract_interval": 3, "fluid_steps_per_frame": 1, "rigid_dt": 0.16, "target_sim_hz": 8.0},
    "default": {"render_extract_interval": 3, "fluid_steps_per_frame": 1, "rigid_dt": 0.14, "target_sim_hz": 10.0},
    "high": {"render_extract_interval": 4, "fluid_steps_per_frame": 1, "rigid_dt": 0.12, "target_sim_hz": 12.0},
}

SCENE_FLOW_RESOLUTIONS: dict[PresetName, dict[QualityName, tuple[int, int, int]]] = {
    "d20_drop": {
        "low": (28, 40, 28),
        "default": (36, 54, 36),
        "high": (48, 72, 48),
    },
    "plane_skim": {
        "low": (52, 24, 22),
        "default": (64, 28, 24),
        "high": (80, 34, 28),
    },
}


def available_preset_names() -> tuple[PresetName, ...]:
    return ("d20_drop", "plane_skim")


def build_preset(name: PresetName, quality: QualityName) -> ScenePreset:
    params = QUALITY_PARAMS[quality]
    nx, ny, nz = SCENE_FLOW_RESOLUTIONS[name][quality]
    tank_extent = (float(nx), float(ny), float(nz))

    if name == "d20_drop":
        radius = min(nx, ny, nz) * 0.105
        return ScenePreset(
            name=name,
            display_name="D20 Drop",
            tank_extent=tank_extent,
            fill_ratio=0.42,
            body_kind="d20",
            body_density=2.2,
            body_size=(radius, radius, radius),
            body_initial_pose=((nx * 0.5, ny * 0.76, nz * 0.5), (1.0, 0.0, 0.0, 0.0)),
            motion_mode="dynamic",
            scripted_linear_velocity=(0.0, 0.0, 0.0),
            scripted_vertical_amplitude=0.0,
            scripted_lateral_amplitude=0.0,
            scripted_roll_amplitude=0.0,
            scripted_pitch=0.0,
            scripted_duration=0.0,
            gravity=(0.0, -0.035, 0.0),
            flow_resolution=(nx, ny, nz),
            phase_ratio=2,
            render_extract_interval=int(params["render_extract_interval"]),
            fluid_steps_per_frame=int(params["fluid_steps_per_frame"]),
            fluid_substeps=1,
            rigid_dt=float(params["rigid_dt"]),
            target_sim_hz=float(params["target_sim_hz"]),
            max_buoyancy_ratio=0.45,
            phase_mass_correction_gain=1.0,
            density_ratio=8.0,
            water_viscosity=0.11,
            air_viscosity=0.028,
            surface_tension=0.0025,
            force_scale=0.0025,
            torque_scale=0.0008,
            camera_target_factor=(0.5, 0.35, 0.5),
            camera_distance_factor=1.65,
        )

    if name == "plane_skim":
        length = nx * 0.18
        height = max(ny * 0.07, 1.5)
        span = nz * 0.48
        waterline = ny * 0.38
        initial_pos = (-length * 0.72, waterline + height * 0.82, nz * 0.5)
        base_pitch = -0.04
        speed = 5.2 if quality == "low" else 5.8 if quality == "default" else 6.4
        duration = (nx + length * 1.6) / speed
        return ScenePreset(
            name=name,
            display_name="Plane Skim",
            tank_extent=tank_extent,
            fill_ratio=0.38,
            body_kind="plane",
            body_density=0.9,
            body_size=(length, height, span),
            body_initial_pose=(initial_pos, tuple(quat_from_euler(0.0, base_pitch, 0.0))),
            motion_mode="scripted",
            scripted_linear_velocity=(speed, 0.0, 0.0),
            scripted_vertical_amplitude=0.1,
            scripted_lateral_amplitude=0.12,
            scripted_roll_amplitude=0.035,
            scripted_pitch=base_pitch,
            scripted_duration=duration,
            gravity=(0.0, -0.035, 0.0),
            flow_resolution=(nx, ny, nz),
            phase_ratio=2,
            render_extract_interval=max(2, int(params["render_extract_interval"])),
            fluid_steps_per_frame=int(params["fluid_steps_per_frame"]),
            fluid_substeps=1,
            rigid_dt=float(params["rigid_dt"]),
            target_sim_hz=float(params["target_sim_hz"]),
            max_buoyancy_ratio=0.0,
            phase_mass_correction_gain=3.5,
            density_ratio=8.0,
            water_viscosity=0.10,
            air_viscosity=0.026,
            surface_tension=0.0023,
            force_scale=0.0018,
            torque_scale=0.0005,
            camera_target_factor=(0.46, 0.28, 0.5),
            camera_distance_factor=1.2,
        )

    raise ValueError(f"Unknown preset: {name}")
