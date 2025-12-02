import unittest
from src.ParkingEnv import ParkingEnv
import numpy as np
import math

class TestEnv(unittest.TestCase):


    def test_calculate_angle_diff(self):
        env = ParkingEnv()
        env.reset()
        action = env.action_space.sample()
        env.step(action)
        
        angle_diff = env._calculate_angle_diff(np.deg2rad(90.0), np.deg2rad(90.0))
        self.assertAlmostEqual(abs(angle_diff), np.deg2rad(0.0), delta=1e-6)

        angle_diff = env._calculate_angle_diff(np.deg2rad(90.0), np.deg2rad(180.0))
        self.assertAlmostEqual(abs(angle_diff), np.deg2rad(90.0), delta=1e-6)

        angle_diff = env._calculate_angle_diff(np.deg2rad(90.0), np.deg2rad(0.0))
        self.assertAlmostEqual(abs(angle_diff), np.deg2rad(90.0), delta=1e-6)

        angle_diff = env._calculate_angle_diff(np.deg2rad(90.0), np.deg2rad(-90.0))
        self.assertAlmostEqual(abs(angle_diff), np.deg2rad(180.0), delta=1e-6)

        angle_diff = env._calculate_angle_diff(np.deg2rad(90.0), np.deg2rad(180.0))
        self.assertAlmostEqual(abs(angle_diff), np.deg2rad(90.0), delta=1e-6)

        goal_direction = env._calculate_goal_direction((0.0, 0.0), (1.0, 1.0))
        self.assertAlmostEqual(goal_direction, np.deg2rad(45.0), delta=1e-6)

        goal_direction = env._calculate_goal_direction((0.0, 0.0), (-1.0, 1.0))
        self.assertAlmostEqual(goal_direction, np.deg2rad(135.0), delta=1e-6)

        goal_direction = env._calculate_goal_direction((0.0, 0.0), (1.0, -1.0))
        self.assertAlmostEqual(goal_direction, np.deg2rad(-45.0), delta=1e-6)

        goal_direction = env._calculate_goal_direction((0.0, 0.0), (-1.0, -1.0))
        self.assertAlmostEqual(goal_direction, np.deg2rad(-135.0), delta=1e-6)



