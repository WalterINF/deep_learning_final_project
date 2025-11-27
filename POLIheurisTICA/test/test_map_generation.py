import unittest
from src.SimulationConfigLoader import SimulationLoader
from src.Visualization import to_rgb_array
import matplotlib.pyplot as plt

class TestMapGeneration(unittest.TestCase):


    def test_map_generation(self):
        simulation_loader = SimulationLoader()
        simulation_1 = simulation_loader.load_simulation()
        rgb_array_1 = to_rgb_array(simulation_1, img_size=(2000, 2000))
        simulation_2 = simulation_loader.load_simulation()
        rgb_array_2 = to_rgb_array(simulation_2, img_size=(2000, 2000))
        simulation_3 = simulation_loader.load_simulation()
        rgb_array_3 = to_rgb_array(simulation_3, img_size=(2000, 2000))
        #subplots
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_array_1)
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_array_2)
        plt.subplot(1, 3, 3)
        plt.imshow(rgb_array_3)
        plt.show()
