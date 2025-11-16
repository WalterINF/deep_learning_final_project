import numpy as np
import gymnasium as gym
from src.SimulationConfigLoader import VehicleConfigLoader, MapConfigLoader
from src.Simulation import ArticulatedVehicle
from src.Simulation import Map
from src.Simulation import MapEntity
import random
import src.Visualization as Visualization
from casadi import cos, sin, tan
from typing import Any, SupportsFloat, Optional
from stable_baselines3 import PPO
import stable_baselines3.common.monitor
import torch
import os
import tensorboard

max_steps = 500

class ParkingEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}

    def __init__(self, vehicle_name: str, map: Map, dt: float = 0.5, max_steps: int = 1000):

        loader = VehicleConfigLoader("codigos/lista_veiculos.json")
        vehicle_geometry = loader.get_geometry(vehicle_name)
        self.vehicle = ArticulatedVehicle(vehicle_geometry)
        self.map = map
        self.dt = dt
        self.max_steps = max_steps
        self.steps = 0
        self.best_distance_to_goal = float('inf')
        # Gymnasium spaces based on README specifications
        sensor_range_m = 150.0  # meters
        speed_limit_ms = 5.0    # m/s
        steering_limit_rad = float(np.deg2rad(28.0))
        jackknife_limit_rad = float(np.deg2rad(65.0))

        map_width, map_height = self.map.get_size()

        # Observation: [x, y, theta, beta, alpha, r1..r14, goal_x, goal_y, goal_theta]
        obs_low = np.array(
            [
                0.0,                         # x
                0.0,                         # y
                -np.pi,                      # theta
                -jackknife_limit_rad,        # beta
                -steering_limit_rad,         # alpha
            ]
            + [0.0] * 14                     # raycasts
            + [0.0, 0.0, -np.pi],            # goal_x, goal_y, goal_theta
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
            + [sensor_range_m] * 14          # raycasts
            + [map_width, map_height, np.pi],# goal_x, goal_y, goal_theta
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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.steps = 0
        start_pos = self.map.get_random_start_position(self.vehicle, max_attempts=100)
        if start_pos is None:
            start_x = self.map.get_size()[0] / 2
            start_y = self.map.get_size()[1] / 2
        else:
            start_x, start_y = start_pos
        start_theta = 0.0
        start_beta = 0.0
        start_alpha = 0.0
        self.best_distance_to_goal = float('inf')
        self.vehicle.update_physical_properties(start_x, start_y, start_theta, start_beta, start_alpha)
        self.vehicle.initialize_raycasts()
        self.vehicle.update_raycasts(self.map.get_entities())

        # Build observation
        x, y = self.vehicle.get_position()
        theta = self.vehicle.get_theta()
        beta = self.vehicle.get_beta()
        alpha_current = self.vehicle.get_alpha()
        raycast_lengths = [self.vehicle.raycasts[f"r{i}"].length for i in range(1, 15)]
        goal_x, goal_y = self.map.get_parking_goal_position()
        goal_theta = self.map.get_parking_goal_theta()
        observation = np.array([x, y, theta, beta, alpha_current] + raycast_lengths + [goal_x, goal_y, goal_theta], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        velocity, alpha = action
        self.steps += 1
        reward = -0.01 # recompensa base por passo de tempo (reduzida)
        terminated = False
        truncated = False
        info = {}

        ## reward for getting closer to goal
        ##calculate distance to goal
        x, y = self.vehicle.get_position()
        goal_x, goal_y = self.map.get_parking_goal_position()
        distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        if distance_to_goal < self.best_distance_to_goal:
            self.best_distance_to_goal = distance_to_goal
            reward += 0.2  # Aumentado para encorajar movimento em direção ao objetivo

        ## Reward for maintaining velocity (encourages movement)
        # Linear reward for velocity magnitude, scaled appropriately
        velocity_reward = 0.05 * abs(velocity) / 5.0  # Max reward of 0.05 at max speed (5.0 m/s)
        reward += velocity_reward

        ## Strongly punish zero/low speed
        if abs(velocity) < 0.1:
            reward -= 0.15  # Penalidade muito maior para ficar parado
        elif abs(velocity) < 0.5:
            reward -= 0.05  # Penalidade menor para velocidade muito baixa

        if self.steps >= self.max_steps:
            truncated = True
            reward = -100.0
        self._move_vehicle(velocity, alpha, self.dt)
        if self._check_vehicle_collision():
            terminated = True
            reward = -100.0
        elif self._check_vehicle_parking():
            terminated = True
            reward = 100.0
        elif self._check_trailer_jackknife():
            terminated = True
            reward = -100.0
        self.vehicle.update_raycasts(self.map.get_entities())


        # Observação: [x, y, theta, beta, alpha, r1..r14]
        x, y = self.vehicle.get_position()
        theta = self.vehicle.get_theta()
        beta = self.vehicle.get_beta()
        alpha_current = self.vehicle.get_alpha()
        raycast_lengths = [self.vehicle.raycasts[f"r{i}"].length for i in range(1, 15)]
        goal_x, goal_y = self.map.get_parking_goal_position()
        goal_theta = self.map.get_parking_goal_theta()
        observation = np.array([x, y, theta, beta, alpha_current] + raycast_lengths + [goal_x, goal_y, goal_theta], dtype=np.float32)

        return observation, reward, terminated, truncated, info

    def render(self):
        rgb_array = Visualization.to_rgb_array(self.map, self.vehicle, img_size=(288, 288))
        return rgb_array

    def close(self):
        pass


    def _move_vehicle(self, velocity: float, alpha: float, dt: float):
        # Current state
        x, y = self.vehicle.get_position()
        theta = self.vehicle.get_theta()
        beta = self.vehicle.get_beta()

        # Geometry
        D = self.vehicle.get_distancia_eixo_dianteiro_quinta_roda() - self.vehicle.get_distancia_eixo_traseiro_quinta_roda()
        L = self.vehicle.get_distancia_eixo_traseiro_trailer_quinta_roda()
        a = self.vehicle.get_distancia_eixo_traseiro_quinta_roda()

        angular_velocity_tractor = (velocity / D) * tan(alpha)
        beta_dot = angular_velocity_tractor * (1 - (alpha * cos(beta)) / L) - (velocity * sin(beta)) / L

        # Kinematics
        x_dot = velocity * cos(theta)
        y_dot = velocity * sin(theta)
        theta_dot = (velocity / D) * tan(alpha)
        beta_dot = beta_dot = angular_velocity_tractor * (1 - (alpha * cos(beta)) / L) - (velocity * sin(beta)) / L

        # Euler step
        new_x = x + x_dot * dt
        new_y = y + y_dot * dt
        new_theta = theta + theta_dot * dt
        new_beta = beta + beta_dot * dt


        self.vehicle.update_physical_properties(new_x, new_y, new_theta, new_beta, alpha)
        self.vehicle.update_raycasts(self.map.get_entities())

    def _check_vehicle_collision(self) -> bool:
        """Verifica se o veículo colidiu com alguma parede ou passou por cima de uma vaga de estacionamento."""
        for entity in self.map.get_entities():
            if entity.type == MapEntity.ENTITY_WALL:
                if self.vehicle.check_collision(entity):
                    return True
        return False

    def _check_vehicle_parking(self) -> bool:
        """Verifica se o trailer do veículo está dentro de uma vaga de estacionamento."""
        goal = self.map.get_parking_goal()
        if goal.get_bounding_box().contains_bounding_box(self.vehicle.get_bounding_box_trailer()):
            return True
        return False

    def _check_trailer_jackknife(self) -> bool:
        """Verifica se o trailer do veículo está em jackknife."""
        return self.vehicle.get_beta() > np.deg2rad(65.0) or self.vehicle.get_beta() < np.deg2rad(-65.0)




def main():
    model = None
    if os.path.exists("ppo_model.zip"):
        #load and train existing model
        model = PPO.load("ppo_model.zip")
        model = train_ppo(model)
    else:
        #train new model
        model = train_ppo(model)
        model.save("ppo_model.zip")
    reward = run_episode_and_save_video(model)
    mean_reward = evaluate_model(model)
    print(f"Mean reward: {mean_reward}")
    print(f"Reward: {reward}")

def evaluate_model(model: PPO, iterations: int = 10):
    total_reward = 0.0
    for _ in range(iterations):
        total_reward += run_episode(model)
    return total_reward / iterations
    
def run_episode_and_save_video(model):
    video_recorder = Visualization.VideoRecorder("simulation.mp4", fps=10)
    map = MapConfigLoader("config/lista_mapas.json").load_map("MAPA_1")
    env = ParkingEnv(vehicle_name="BUG1", map=map, dt=0.5, max_steps=max_steps)
    observation, info = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        video_recorder.append(env.render())
        if terminated or truncated:
            break

    video_recorder.close()
    env.close()
    return total_reward

def run_episode(model):
    map = MapConfigLoader("config/lista_mapas.json").load_map("MAPA_1")
    env = ParkingEnv(vehicle_name="BUG1", map=map, dt=0.5, max_steps=max_steps)
    observation, info = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()
    return total_reward

def run_random_episode():
    map = MapConfigLoader("config/lista_mapas.json").load_map("MAPA_1")
    env = ParkingEnv(vehicle_name="BUG1", map=map, dt=0.03, max_steps=max_steps)
    observation, info = env.reset()
    total_reward = 0.0
    frames = []
    for t in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        frames.append(env.render())
        if terminated or truncated:
            break
    Visualization.save_frames_as_mp4(frames, "simulation.mp4")
    env.close()
    return total_reward

def train_ppo(model: PPO | None = None):
    # Create environment

    log_dir = "logs"

    map = MapConfigLoader("config/lista_mapas.json").load_map("MAPA_1")

    env = ParkingEnv(vehicle_name="BUG1", map=map, dt=0.5, max_steps=max_steps)

    # Wrap the environment with a Monitor to log training progress
    # so we don't need to manually record statistics
    env = stable_baselines3.common.monitor.Monitor(env, log_dir)

    # neural network hyperparameters
    # net_arch is a list of number of neurons per hidden layer, e.g. [16,20] means
    # two hidden layers with 16 and 20 neurons, respectively
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=[16,16,8])

    # instantiates the model using the defined hyperparameters
    if model is None:
        model = PPO(
            policy="MlpPolicy",           # neural network policy architecture (MLP for vector observations)
            env=env,                      # gymnasium-compatible environment to train on
            policy_kwargs=policy_kwargs,  # custom network architecture and activation
            verbose=0,                    # logging verbosity: 0(silent),1(info),2(debug)
            tensorboard_log=log_dir,      # directory for TensorBoard logs
            learning_rate=3e-4,           # optimizer learning rate
            n_steps=2048,                 # rollout steps per environment update
            batch_size=64,                # minibatch size for optimization
            gamma=0.99,                   # discount factor
            gae_lambda=0.95,              # GAE lambda for bias-variance tradeoff
            ent_coef=0.01,                 # entropy coefficient (encourages exploration)
            clip_range=0.2,               # PPO clipping parameter
            n_epochs=10,                  # number of optimization epochs per update
            device="auto"                 # use GPU if available, else CPU
        )
    else:
        model.set_env(env)

    # You can also experiment with other RL algorithms like A2C, PPO, DDPG etc.
    # Refer to  https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
    # for documentation. For example, if you would like to run DDPG, just replace "DQN" above with "DDPG".

    model.learn(total_timesteps=50000, progress_bar=True)

    model.save("ppo_model")

    return model


if __name__ == "__main__":
    main()
