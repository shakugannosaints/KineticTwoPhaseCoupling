import unittest

import numpy as np

from sim.geometry import TankShape
from sim.rigid import RigidBodySolver, RigidBodyState


class RigidBodyTests(unittest.TestCase):
    def test_contact_keeps_body_inside_box(self) -> None:
        tank = TankShape((20.0, 20.0, 20.0))
        solver = RigidBodySolver(tank)
        state = RigidBodyState.create_d20(radius=2.5, density=1.2, position=(2.0, 1.0, 2.0), orientation=(1.0, 0.0, 0.0, 0.0))
        state.linear_velocity[:] = np.array([-2.0, -3.0, -1.0])
        solver.integrate(state, np.zeros(3), np.zeros(3), np.array([0.0, -0.1, 0.0]), dt=1.0)
        shape = solver.make_shape(state)
        vertices = shape.world_vertices()
        self.assertGreaterEqual(vertices[:, 0].min(), -1e-4)
        self.assertGreaterEqual(vertices[:, 1].min(), -1e-4)
        self.assertGreaterEqual(vertices[:, 2].min(), -1e-4)


if __name__ == "__main__":
    unittest.main()

