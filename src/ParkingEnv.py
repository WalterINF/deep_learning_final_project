import numpy as np
import gymnasium as gym
from SimulationConfigLoader import VehicleConfigLoader, MapConfigLoader
from Simulation import ArticulatedVehicle
from Simulation import Map
from Simulation import MapEntity
import random
import Visualization as Visualization
from casadi import cos, sin, tan
from typing import Any, SupportsFloat, Optional
import stable_baselines3.common.monitor
import torch
import os
import tensorboard
import numerize
import math


class ParkingEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}


    ## ambiente
    VEHICLE_NAME = "BUG1" #nome do veículo
    MAP_NAME = "MAPA_COMPLEXO" #nome do mapa
    SENSOR_RANGE_M = 20.0 # raio do sensor
    SPEED_LIMIT_MS = 5.0 # velocidade maxima
    STEERING_LIMIT_RAD = float(np.deg2rad(28.0)) # angulo maximo de esterçamento
    JACKKNIFE_LIMIT_RAD = float(np.deg2rad(65.0)) # angulo maximo de jackknife
    DT = 0.2 # tempo de simulação
    MAX_SECONDS = 120.0
    MAX_STEPS = int(MAX_SECONDS / DT)
    VEHICLE_PARKED_THRESHOLD_M = 3.0 # distancia minima entre centro do trailer e centro da vaga para considerar o veículo estacionado


    ## recompensas
    REWARD_GOAL = 100.0 # recompensa por chegar ao objetivo
    REWARD_ALIGNMENT = 50.0 # recompensa adicional por alinhar o veículo na vaga corretamente
    REWARD_PROGRESS = 50.0 # recompensa por progresso, calculado como (porcentagem de distancia ganha no passo * REWARD_PROGRESS)
    MAX_PUNISHMENT_TIME_PER_EPISODE = -20.0 # penalidade maxima por tempo acumulada durante o episódio
    PUNISHMENT_TIME = MAX_PUNISHMENT_TIME_PER_EPISODE / MAX_STEPS # penalidade por tempo a cada passo
    PUNISHMENT_ZERO_SPEED = 5 * PUNISHMENT_TIME # penalidade por velocidade zero - 5 vezes maior que a penalidade por tempo
    PUNISHMENT_COLLISION = -150.0 # penalidade por colisão com paredes
    PUNISHMENT_OVERLAP = 20 * PUNISHMENT_TIME # penalidade por invadir uma vaga a cada passo, 20 vezes maior que a penalidade por tempo
    PUNISHMENT_JACKKNIFE = -150.0 # penalidade por jackknife


    def __init__(self, seed = 0):

        self.render_mode = "rgb_array"

        self.vehicle_loader = VehicleConfigLoader("config/lista_veiculos.json")
        self.map_loader = MapConfigLoader("config/lista_mapas.json")

        self.vehicle = self.vehicle_loader.load_vehicle(self.VEHICLE_NAME)
        self.map = self.map_loader.load_map(self.MAP_NAME)

        self.map.place_vehicle(self.vehicle)

        self.steps = 0
        self.total_reward = 0.0

        self.initial_distance_to_goal = self._calculate_distance_manhattan(self.vehicle.get_position(), self.map.get_parking_goal_position())
        self.last_distance_to_goal = self._calculate_distance_manhattan(self.vehicle.get_position(), self.map.get_parking_goal_position())

        
        # Observation: [theta,           # ângulo de orientação do veículo
        #              beta,            # ângulo de articulação do trator-trailer
        #              alpha,           # ângulo de esterçamento do trator
        #              r1..r14,         # distâncias dos raycasts do veículo
        #              e1..e14,         # classes dos objetos detectados pelos raycasts (0: nada, 1: parede, 2: vaga de estacionamento)
        #              goal_distance,   # distância relativa do veículo ao objetivo em metros (disância atual/disância inicial)
        #              goal_direction,  # ângulo global em relação ao objetivo em radianos
        #              angle_diff]      # diferença de orientação entre a vaga de estacionamento e o veículo
        obs_low = np.array(
            [
                -np.pi,                      # theta
                -self.JACKKNIFE_LIMIT_RAD,        # beta
                -self.STEERING_LIMIT_RAD,         # alpha
            ]
            + [0.0] * self.vehicle.get_raycast_count()                     # raycast lengths
            + [MapEntity.MIN_COLLIDABLE_ENTITY_TYPE] * self.vehicle.get_raycast_count() # raycast object classes
            + [0.0, -np.pi, -np.pi],            # goal distance, goal direction, angle_diff
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                np.pi,                       # theta
                self.JACKKNIFE_LIMIT_RAD,         # beta
                self.STEERING_LIMIT_RAD,          # alpha
            ]
            + [self.SENSOR_RANGE_M] * self.vehicle.get_raycast_count()          # raycast lengths
            + [MapEntity.MAX_COLLIDABLE_ENTITY_TYPE] * self.vehicle.get_raycast_count() # raycast object classes
            + [200, np.pi, np.pi], # goal distance, goal direction, angle_diff
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: [v, alpha]
        act_low = np.array(
            [
                -self.SPEED_LIMIT_MS,             # v (allow reverse)
                -self.STEERING_LIMIT_RAD,         # alpha
            ],
            dtype=np.float32,
        )
        act_high = np.array(
            [
                self.SPEED_LIMIT_MS,             # v (allow reverse)
                self.STEERING_LIMIT_RAD,          # alpha
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.vehicle = self.vehicle_loader.load_vehicle(self.VEHICLE_NAME)
        self.map = self.map_loader.load_map(self.MAP_NAME)

        self.map.place_vehicle(self.vehicle)

        self.steps = 0
        self.total_reward = 0.0

        self.initial_distance_to_goal = self._calculate_distance_manhattan(self.vehicle.get_position(), self.map.get_parking_goal_position())
        self.last_distance_to_goal = self._calculate_distance_manhattan(self.vehicle.get_position(), self.map.get_parking_goal_position())

        # Build observation
        observation = self._build_observation()
        info = {}
        return observation, info

    def _build_observation(self) -> np.ndarray:
        # Observação: [theta, beta, alpha, r1..r14, e1..e14, goal_dist, goal_direction, angle_diff]
        theta = self.vehicle.get_theta()
        beta = self.vehicle.get_beta()
        alpha_current = self.vehicle.get_alpha()
        raycast_obs = self.vehicle.get_raycast_lengths_and_object_classes()
        goal_distance = self._calculate_distance_euclidean(self.vehicle.get_position(), self.map.get_parking_goal_position())
        goal_direction = self._calculate_goal_direction(self.vehicle.get_position(), self.map.get_parking_goal_position())
        angle_diff = self._calculate_angle_diff(self.vehicle.get_theta(), self.map.get_parking_goal_theta())
        observation = np.array(
            [theta, beta, alpha_current]
            + raycast_obs
            + [goal_distance, goal_direction, angle_diff],
            dtype=np.float32,
        )
        return observation

    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        velocity, alpha = action
        self.steps += 1
        terminated = False
        truncated = False
        info = {}
        reward = self.PUNISHMENT_TIME # recompensa base por passo de tempo (reduzida)

        self._move_vehicle(velocity, alpha, self.DT)

        ## Strongly punish zero/low speed
        if abs(velocity) < 0.1:
            reward = self.PUNISHMENT_ZERO_SPEED  # Penalidade muito maior para ficar parado

        if self.steps >= self.MAX_STEPS:
            truncated = True
        elif self._check_vehicle_parking():
            terminated = True
            reward = self._calculate_parking_reward(self.vehicle.get_theta(), self.map.get_parking_goal_theta())
        elif self._check_trailer_jackknife(self.vehicle.get_beta()):
            terminated = True
            reward = self.PUNISHMENT_JACKKNIFE
        else:
            collided, overlapped = self._check_vehicle_collision_or_overlap()
            if collided:
                terminated = True
                reward = self.PUNISHMENT_COLLISION
            elif overlapped:
                reward += self.PUNISHMENT_OVERLAP
 
        new_distance_to_goal = self._calculate_distance_manhattan(self.vehicle.get_position(), self.map.get_parking_goal_position())
        #recompensa baseada no progresso real em relação à distancia inicial para o objetivo
        den = max(self.initial_distance_to_goal, 1e-6)
        progress_percentage = (self.last_distance_to_goal - new_distance_to_goal) / den
        reward += progress_percentage * self.REWARD_PROGRESS

        self.last_distance_to_goal = new_distance_to_goal

        observation = self._build_observation()
        self.total_reward += reward

        return observation, reward, terminated, truncated, info

    def render(self):
        rgb_array = Visualization.to_rgb_array(self.map, self.vehicle, (288, 288), self._calculate_distance_euclidean(self.vehicle.get_position(), self.map.get_parking_goal_position()), self._calculate_goal_direction(self.vehicle.get_position(), self.map.get_parking_goal_position()), total_reward=self.total_reward)
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

        # Kinematics
        x_dot = velocity * cos(theta)
        y_dot = velocity * sin(theta)
        theta_dot = (velocity / D) * tan(alpha)
        beta_dot = angular_velocity_tractor * (1 - (a * cos(beta)) / L) - (velocity * sin(beta)) / L

        # Euler step
        new_x = x + x_dot * dt
        new_y = y + y_dot * dt
        new_theta = theta + theta_dot * dt
        new_beta = beta + beta_dot * dt

        self.vehicle.update_physical_properties(new_x, new_y, velocity, new_theta, new_beta, alpha)
        self.vehicle.update_raycasts(self.map.get_entities())

    def _check_vehicle_collision(self) -> bool:
        """Verifica se o veículo colidiu com alguma parede ou passou por cima de uma vaga de estacionamento."""
        for entity in self.map.get_entities():
            if entity.type == MapEntity.ENTITY_WALL or entity.type == MapEntity.ENTITY_PARKING_SLOT:
                if self.vehicle.check_collision(entity):
                    return True
        return False

    def _check_vehicle_collision_or_overlap(self) -> tuple[bool,bool]:
        """Verifica se o veículo colidiu com alguma parede ou passou por cima de uma vaga de estacionamento.
        Retorna uma tupla [bool,bool]"""
        collided = False
        overlapped = False
        for entity in self.map.get_entities():
            if entity.type == MapEntity.ENTITY_PARKING_SLOT:
                if self.vehicle.check_collision(entity):
                    overlapped = True
            elif entity.type == MapEntity.ENTITY_WALL:
                if self.vehicle.check_collision(entity):
                    collided = True
        
        return (collided, overlapped)
                    


    def _calculate_distance_euclidean(self, position: tuple[float, float], goal_position: tuple[float, float]):
        distance_to_goal = np.sqrt((position[0] - goal_position[0])**2 + (position[1] - goal_position[1])**2)
        return distance_to_goal

    def _calculate_distance_manhattan(self, position: tuple[float, float], goal_position: tuple[float, float]):
        distance_to_goal = abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1])
        return distance_to_goal

    def _calculate_goal_direction(self, position: tuple[float, float], goal_position: tuple[float, float]):
        direction_to_goal = np.arctan2(goal_position[1] - position[1], goal_position[0] - position[0])
        return direction_to_goal

    def _calculate_angle_diff(self, vehicle_theta: float, parking_goal_theta: float):
        """
        Calcula a diferença de orientação entre a vaga de estacionamento (parking_goal_theta)
        e o veículo (vehicle_theta), normalizada para o intervalo [-pi, pi].
        """
        diff = parking_goal_theta - vehicle_theta
        # normaliza para [-pi, pi]
        angle_diff = np.arctan2(np.sin(diff), np.cos(diff))
        return angle_diff

    def _check_vehicle_parking(self) -> bool:
        """Verifica se o trailer do veículo está dentro de uma vaga de estacionamento."""
        return self._calculate_distance_euclidean(self.vehicle.get_position(), self.map.get_parking_goal_position()) < self.VEHICLE_PARKED_THRESHOLD_M

    def _calculate_parking_reward(self, vehicle_theta: float, parking_goal_theta: float) -> float:
        """Calcula a recompensa por estacionar o veículo. valr máximo quando a diferença de orientação é 0
        e valor mínimo quando a diferença de orientação é pi/2 ou maior"""
        angle_diff = self._calculate_angle_diff(vehicle_theta, parking_goal_theta)
        #se é maior que pi/2, considera 0
        if abs(angle_diff) > math.pi/2:
            return self.REWARD_GOAL
        return self.REWARD_GOAL + (1 - (abs(angle_diff) / (math.pi/2))) * self.REWARD_ALIGNMENT

    def _check_trailer_jackknife(self, beta: float) -> bool:
        """Verifica se o trailer do veículo está em jackknife."""
        return abs(beta) > self.JACKKNIFE_LIMIT_RAD