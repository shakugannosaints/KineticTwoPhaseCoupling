import unittest

import numpy as np

from sim.geometry import D20Shape


class GeometryTests(unittest.TestCase):
    def test_d20_signed_distance_signs(self) -> None:
        shape = D20Shape(
            radius=3.0,
            position=np.array([5.0, 5.0, 5.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        self.assertLess(shape.signed_distance(np.array([5.0, 5.0, 5.0])), 0.0)
        self.assertGreater(shape.signed_distance(np.array([12.0, 5.0, 5.0])), 0.0)


if __name__ == "__main__":
    unittest.main()

