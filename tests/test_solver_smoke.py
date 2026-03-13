import unittest

import numpy as np
import taichi as ti

from sim.app import SimulationApp
from sim.config import build_preset
from sim.geometry import make_body_shape
from sim.rigid import RigidBodyState
from sim.solver import DualGridLBMSolver


class SolverSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, offline_cache=False)

    def test_small_cpu_run_stays_finite(self) -> None:
        preset = build_preset("d20_drop", "low")
        preset = preset.__class__(
            **{
                **preset.__dict__,
                "flow_resolution": (12, 18, 12),
                "tank_extent": (12.0, 18.0, 12.0),
                "body_size": (1.6, 1.6, 1.6),
                "body_initial_pose": ((6.0, 13.0, 6.0), (1.0, 0.0, 0.0, 0.0)),
            }
        )
        state = RigidBodyState.create_body(
            shape_kind=preset.body_kind,
            shape_size=preset.body_size,
            density=preset.body_density,
            position=preset.body_initial_pose[0],
            orientation=preset.body_initial_pose[1],
        )
        shape = make_body_shape(
            shape_kind=state.shape_kind,
            shape_size=state.shape_size,
            position=state.position,
            orientation=state.orientation,
            linear_velocity=state.linear_velocity,
            angular_velocity=state.angular_velocity,
        )
        solver = DualGridLBMSolver(preset)
        solver.configure_geometry(state, shape.face_normals, shape.face_offsets)
        for _ in range(3):
            force, torque = solver.step()
            self.assertTrue(np.isfinite(force).all())
            self.assertTrue(np.isfinite(torque).all())
        phi = solver.get_phase_field()
        self.assertTrue(np.isfinite(phi).all())
        self.assertGreaterEqual(float(phi.min()), 0.0)
        self.assertLessEqual(float(phi.max()), 1.0)

    def test_plane_scene_advances_scripted_body(self) -> None:
        preset = build_preset("plane_skim", "low")
        app = SimulationApp(preset)
        start_x = float(app.state.position[0])
        for _ in range(30):
            app.step()
            app.refresh_surface(force=True)
        self.assertGreater(float(app.state.position[0]), start_x)
        self.assertTrue(np.isfinite(app.solver.get_surface_field()).all())
        plane_bottom = float(app.body_mesh()[0][:, 1].min())
        self.assertGreater(plane_bottom - float(app.surface_height_hint), 0.1)
        app.shutdown()


if __name__ == "__main__":
    unittest.main()
