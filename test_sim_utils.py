"""sim_utils.quaternion_yaw must match the template's quaternion2bearing (run: python -m unittest test_sim_utils)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "robotnik_arm_task"))

from sim_utils import quaternion_yaw  # noqa: E402


class YawMatchesTemplate(unittest.TestCase):
    def test_random_quaternions(self):
        from robotnik_arm_control import quaternion2bearing  # reference only, in the test
        rng = np.random.default_rng(0)
        for q in rng.normal(size=(2000, 4)):
            self.assertAlmostEqual(quaternion_yaw(*q), quaternion2bearing(*q), places=9)

    def test_planar_yaw(self):
        for yaw in np.linspace(-3.1, 3.1, 50):
            self.assertAlmostEqual(quaternion_yaw(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)), yaw, places=12)


if __name__ == "__main__":
    unittest.main()
