import numpy as np
import gymnasium as gym
from VehicleConfigLoader import VehicleConfigLoader
from Simulation import ArticulatedVehicle
from Simulation import Map
from Simulation import MapEntity
import random
import Visualization



class ParkingEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}

    def __init__(self, vehicle_name: str, map_size: tuple[float, float], dt: float = 0.1):

        loader = VehicleConfigLoader("codigos/lista_veiculos.json")
        vehicle_geometry = loader.get_geometry(vehicle_name)
        self.vehicle = ArticulatedVehicle(geometry=vehicle_geometry)
        self.map_size = map_size
        self.map = Map(map_size, self.vehicle)
        self.map.add_entity(MapEntity(position_x=map_size[0]/2, position_y=map_size[1]/2, width=10.0, height=10.0, theta=0.0, type=MapEntity.ENTITY_WALL))
        self.dt = dt

        # Gymnasium spaces based on README specifications
        sensor_range_m = 150.0  # meters
        speed_limit_ms = 5.0    # m/s
        steering_limit_rad = float(np.deg2rad(28.0))
        jackknife_limit_rad = float(np.deg2rad(65.0))

        map_width, map_height = float(self.map_size[0]), float(self.map_size[1])

        # Observation: [x, y, theta, beta, alpha, r1..r14]
        obs_low = np.array(
            [
                0.0,                         # x
                0.0,                         # y
                -np.pi,                      # theta
                -jackknife_limit_rad,        # beta
                -steering_limit_rad,         # alpha
            ]
            + [0.0] * 14,                    # raycasts
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                map_width,                   # x
                map_height,                  # y
                np.pi,                       # theta
                jackknife_limit_rad,         # beta
                steering_limit_rad,          # alpha
            ]
            + [sensor_range_m] * 14,         # raycasts
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: [v, alpha]
        act_low = np.array(
            [
                -speed_limit_ms,             # v (allow reverse)
                -steering_limit_rad,         # alpha
            ],
            dtype=np.float32,
        )
        act_high = np.array(
            [
                speed_limit_ms,              # v
                steering_limit_rad,          # alpha
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

    def reset(self):
        pass

    def step(self, action) :
        pass

    def render(self):
        pass

    def close(self):
        pass

class Simulation:

    map: Map # mapa do simulador, contendo o veículo e as entidades
    dt: float # passo de integração

    def __init__(self, map_size: tuple[float, float], vehicle: ArticulatedVehicle, entities: list[MapEntity] = None, dt: float = 0.03):
        self.map = Map(map_size, vehicle, entities)
        self.map.add_entity(MapEntity(position_x=map_size[0], position_y=map_size[1]/2, width=10.0, height=10.0, theta=0.0, type=MapEntity.ENTITY_WALL))

        # parede esquerda
        self.map.add_entity(MapEntity(position_x=0.0, position_y=map_size[1]/2, width=2.0, height=map_size[1], theta=0.0, type=MapEntity.ENTITY_WALL))
        # parede direita
        self.map.add_entity(MapEntity(position_x=map_size[0], position_y=map_size[1]/2, width=2.0, height=map_size[1], theta=0.0, type=MapEntity.ENTITY_WALL))
        # parede superior
        self.map.add_entity(MapEntity(position_x=map_size[0]/2, position_y=0.0, width=map_size[0], height=2.0, theta=0.0, type=MapEntity.ENTITY_WALL))
        # parede inferior
        self.map.add_entity(MapEntity(position_x=map_size[0]/2, position_y=map_size[1], width=map_size[0], height=2.0, theta=0.0, type=MapEntity.ENTITY_WALL))

        self.dt = dt

    def step(self, velocity: float, alpha: float):
        self.map.move_vehicle(velocity, alpha, self.dt)

    def run_random_simulation_and_save_video(self, time_steps: int, output_path: str):
        print(f"Running random simulation and saving video to {output_path}")
        frames = []
        for i in range(time_steps):
            print(f"Step {i} of {time_steps}")
            velocity = random.uniform(0.0, 10.0)
            alpha = random.uniform(-0.7, 0.7)
            self.step(velocity, alpha)
            frames.append(Visualization.to_rgb_array(self.map, img_size=(360, 360)))
        Visualization.save_frames_as_mp4(frames, output_path)


def main():
    loader = VehicleConfigLoader("codigos/lista_veiculos.json")
    vehicle_geometry = loader.get_geometry("BUG1")
    articulated_vehicle = ArticulatedVehicle(geometry=vehicle_geometry)
    simulation = Simulation(map_size=(60, 60), vehicle=articulated_vehicle, dt=0.03)
    simulation.run_random_simulation_and_save_video(time_steps=200, output_path="simulation.mp4")

if __name__ == "__main__":
    main()
