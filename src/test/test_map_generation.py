import unittest
from src.SimulationConfigLoader import SimulationLoader
from src.Visualization import to_rgb_array
from src.ParkingEnv import ParkingEnv
import matplotlib.pyplot as plt

class TestMapGeneration(unittest.TestCase):


    def test_map_generation(self):
        simulation_loader = SimulationLoader()
        simulation_1 = simulation_loader.load_simulation()
        rgb_array_1 = to_rgb_array(simulation_1, img_size=(2000, 2000))
        env1 = ParkingEnv()
        rgb_array_2 = env1.render()
        #subplots
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_array_1)
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_array_2)
        plt.show()
