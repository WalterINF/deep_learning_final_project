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


class ParkingEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}


    ## ambiente
    VEHICLE_NAME = "BUG1" #nome do veículo
    MAP_NAME = "MAPA_1" #nome do mapa
    SENSOR_RANGE_M = 20.0 # raio do sensor
    SPEED_LIMIT_MS = 5.0 # velocidade maxima
    STEERING_LIMIT_RAD = float(np.deg2rad(28.0)) # angulo maximo de esterçamento
    JACKKNIFE_LIMIT_RAD = float(np.deg2rad(65.0)) # angulo maximo de jackknife
    DT = 0.5 # tempo de simulação
    MAX_SECONDS = 120.0
    MAX_STEPS = int(MAX_SECONDS / DT)


    ## recompensas
    MAX_PUNISHMENT_TIME_PER_EPISODE = -20.0 # penalidade maxima por tempo acumulada por episodio
    #deve ser menor que a punição por colisão e jackknife, senão o agente vai colidir propositalmente

    REWARD_GOAL = 100.0 # recompensa por chegar ao objetivo
    PUNISHMENT_TIME = MAX_PUNISHMENT_TIME_PER_EPISODE / MAX_STEPS # penalidade por tempo a cada passo
    PUNISHMENT_ZERO_SPEED = 5 * PUNISHMENT_TIME # penalidade por velocidade zero - 5 vezes maior que a penalidade por tempo
    PUNISHMENT_COLLISION = -150.0 # penalidade por colisão com paredes
    PUNISHMENT_OVERLAP = -1 # penalidade por invadir uma vaga dada a cada passo
    PUNISHMENT_JACKKNIFE = -150.0 # penalidade por jackknife
    PROGRESS_REWARD_MULTIPLIER = 0.5 # multiplicador da recompensa por progresso (recompensa = metros ganhos * multiplicador)

    def __init__(self, seed = 0):

        self.render_mode = "rgb_array"

        self.vehicle_loader = VehicleConfigLoader("config/lista_veiculos.json")
        self.map_loader = MapConfigLoader("config/lista_mapas.json")

        self.vehicle = self.vehicle_loader.load_vehicle(self.VEHICLE_NAME)
        self.map = self.map_loader.load_map(self.MAP_NAME)

        self.map.place_vehicle(self.vehicle)

        self.steps = 0
        self.total_reward = 0.0

        self.last_distance_to_goal = self._calculate_goal_distance_manhattan()

        


        # Observation: [x, y, theta, beta, alpha, r1..r14, goal_x, goal_y, goal_theta]
        obs_low = np.array(
            [
                -np.pi,                      # theta
                -self.JACKKNIFE_LIMIT_RAD,        # beta
                -self.STEERING_LIMIT_RAD,         # alpha
            ]
            + [0.0] * self.vehicle.get_raycast_count()                     # raycasts
            + [0.0, -np.pi],            # goal distance, goal direction
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                np.pi,                       # theta
                self.JACKKNIFE_LIMIT_RAD,         # beta
                self.STEERING_LIMIT_RAD,          # alpha
            ]
            + [self.SENSOR_RANGE_M] * self.vehicle.get_raycast_count()          # raycasts
            + [100, np.pi], # goal distance, goal direction
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

        self.last_distance_to_goal = self._calculate_goal_distance_manhattan()

        # Build observation
        theta = self.vehicle.get_theta()
        beta = self.vehicle.get_beta()
        alpha_current = self.vehicle.get_alpha()
        raycast_lengths = self.vehicle.get_raycast_lengths()
        goal_distance = self._calculate_goal_distance()
        goal_direction = self._calculate_goal_direction()
        observation = np.array([theta, beta, alpha_current] + raycast_lengths + [goal_distance, goal_direction], dtype=np.float32)
        info = {}
        return observation, info

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
            reward = self.REWARD_GOAL
        elif self._check_trailer_jackknife():
            terminated = True
            reward = self.PUNISHMENT_JACKKNIFE
        else:
            collided, overlapped = self._check_vehicle_collision_or_overlap()
            if collided:
                terminated = True
                reward = self.PUNISHMENT_COLLISION
            elif overlapped:
                reward += self.PUNISHMENT_OVERLAP
 
        new_distance_to_goal = self._calculate_goal_distance_manhattan()

        # Recompensa baseada no progresso real
        progress_reward = (self.last_distance_to_goal - new_distance_to_goal)
        reward += progress_reward * self.PROGRESS_REWARD_MULTIPLIER # Ajuste o multiplicador (0.1) conforme necessário

        self.last_distance_to_goal = new_distance_to_goal

        # Observação: [x, y, theta, beta, alpha, r1..r14, goal_dist, goal_direction]
        theta = self.vehicle.get_theta()
        beta = self.vehicle.get_beta()
        alpha_current = self.vehicle.get_alpha()
        raycast_lengths = self.vehicle.get_raycast_lengths()
        observation = np.array([theta, beta, alpha_current] + raycast_lengths + [self._calculate_goal_distance(), self._calculate_goal_direction()], dtype=np.float32)

        self.total_reward += reward

        return observation, reward, terminated, truncated, info

    def render(self):
        rgb_array = Visualization.to_rgb_array(self.map, self.vehicle, (288, 288), self._calculate_goal_distance(), self._calculate_goal_direction(), total_reward=self.total_reward)
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
                    


    def _calculate_goal_distance(self):
        x, y = self.vehicle.get_trailer_position()
        goal_x, goal_y = self.map.get_parking_goal_position()
        distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        return distance_to_goal

    def _calculate_goal_distance_manhattan(self):
        x, y = self.vehicle.get_trailer_position()
        goal_x, goal_y = self.map.get_parking_goal_position()
        distance_to_goal = abs(x - goal_x) + abs(y - goal_y)
        return distance_to_goal

    def _calculate_goal_direction(self):
        x, y = self.vehicle.get_position()
        goal_x, goal_y = self.map.get_parking_goal_position()
        direction_to_goal = np.arctan2(goal_y - y, goal_x - x)
        return direction_to_goal

    def _check_vehicle_parking(self) -> bool:
        """Verifica se o trailer do veículo está dentro de uma vaga de estacionamento."""
        goal = self.map.get_parking_goal()
        if goal.get_bounding_box().contains_bounding_box(self.vehicle.get_bounding_box_trailer()):
            return True
        return False

    def _check_trailer_jackknife(self) -> bool:
        """Verifica se o trailer do veículo está em jackknife."""
        return self.vehicle.get_beta() > np.deg2rad(65.0) or self.vehicle.get_beta() < np.deg2rad(-65.0)