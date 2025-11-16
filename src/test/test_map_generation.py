import unittest
from SimulationConfigLoader import MapConfigLoader
from Simulation import Map
from Visualization import to_rgb_array
from SimulationConfigLoader import VehicleConfigLoader
import matplotlib.pyplot as plt

class TestMapGeneration(unittest.TestCase):


    def test_map_generation(self):
        map_loader = MapConfigLoader("config/lista_mapas.json")
        vehicle_loader = VehicleConfigLoader("config/lista_veiculos.json")
        map1 = map_loader.load_map("MAPA_1")

        vehicle = vehicle_loader.load_vehicle("BUG1")
        map1.place_vehicle(vehicle)
        self.assertIsNotNone(map1)
        self.assertIsNotNone(map1.get_start_position())
        self.assertIsNotNone(map1.get_parking_goal())

        rgb_array1 = to_rgb_array(map1, vehicle, img_size=(576, 576))
    
        map2 = self._generate_random_map()
        vehicle2 = vehicle_loader.load_vehicle("BUG1")
        map2.place_vehicle(vehicle2)
        self.assertIsNotNone(map2)
        self.assertIsNotNone(map2.get_start_position())
        self.assertIsNotNone(map2.get_parking_goal())

        rgb_array2 = to_rgb_array(map2, vehicle2, img_size=(576, 576))

        map3 = self._generate_random_map()
        vehicle3 = vehicle_loader.load_vehicle("BUG1")
        map3.place_vehicle(vehicle3)
        self.assertIsNotNone(map3)
        self.assertIsNotNone(map3.get_start_position())
        self.assertIsNotNone(map3.get_parking_goal())

        rgb_array3 = to_rgb_array(map3, vehicle3, img_size=(576, 576))

        plt.figure(figsize=(12, 4))
        plt.suptitle("Mapas gerados aleatoriamente")
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_array1)
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_array2)
        plt.subplot(1, 3, 3)
        plt.imshow(rgb_array3)
        plt.show()

    def _generate_random_map(self):
        map = MapConfigLoader("config/lista_mapas.json").load_map("MAPA_1")
        vehicle = VehicleConfigLoader("config/lista_veiculos.json").load_vehicle("BUG1")
        map.place_vehicle(vehicle)
        return map

        
