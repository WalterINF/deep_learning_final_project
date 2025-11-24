import numpy as np
import gymnasium as gym
from SimulationConfigLoader import SimulationLoader
from Simulation import MapEntity
import Visualization as Visualization
from typing import Any, SupportsFloat
import math


class ParkingEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}


    ## ambiente
    SENSOR_RANGE_M = 10.0 # raio do sensor
    SPEED_LIMIT_MS = 5.0 # velocidade maxima
    STEERING_LIMIT_RAD = float(np.deg2rad(28.0)) # angulo maximo de esterçamento
    JACKKNIFE_LIMIT_RAD = float(np.deg2rad(65.0)) # angulo maximo de jackknife
    DT = 0.2 # tempo do passo de simulação
    MAX_SECONDS = 90.0
    MAX_STEPS = int(MAX_SECONDS / DT)
    VEHICLE_PARKED_THRESHOLD_M = 5.0 # distancia minima entre centro do trailer e centro da vaga para considerar o veículo estacionado


    ## recompensas
    REWARD_GOAL = 50.0 # recompensa por chegar ao objetivo
    REWARD_ALIGNMENT = 50.0 # recompensa adicional por alinhar o veículo na vaga corretamente
    REWARD_PROGRESS = 2.0 #recompensa por metro de progresso
    MAX_PUNISHMENT_TIME_PER_EPISODE = -20.0 # penalidade maxima por tempo acumulada durante o episódio
    PUNISHMENT_TIME = MAX_PUNISHMENT_TIME_PER_EPISODE / MAX_STEPS # penalidade por tempo a cada passo
    PUNISHMENT_ZERO_SPEED = -0.5 # penalidade por velocidade zero 
    PUNISHMENT_COLLISION = -25.0 # penalidade por colisão com paredes
    PUNISHMENT_OVERLAP = 0.0 * PUNISHMENT_TIME # penalidade por invadir uma vaga a cada passo,
    PUNISHMENT_JACKKNIFE = -25.0 # penalidade por jackknife

    REWARD_VELOCITY = -1.0 * PUNISHMENT_TIME #recompensa por velocidade


    def __init__(self, seed = 0):

        self.render_mode = "rgb_array"

        self.simulation_loader = SimulationLoader()

        self.simulation = self.simulation_loader.load_simulation()

        self.steps = 0
        self.total_reward = 0.0

        self.initial_distance_to_goal = self._calculate_distance_euclidean(self.simulation.vehicle.get_position(), self.simulation.map.get_parking_goal_position())
        self.last_distance_to_goal = self._calculate_distance_euclidean(self.simulation.vehicle.get_position(), self.simulation.map.get_parking_goal_position())

        
        # Observation: [theta,           # ângulo de orientação do veículo
        #              beta,            # ângulo de articulação do trator-trailer
        #              alpha,           # ângulo de esterçamento do trator
        #              r1..r14,         # distâncias dos raycasts do veículo
        #              e1..e14,         # classes dos objetos detectados pelos raycasts (0: nada, 1: parede, 2: vaga de estacionamento)
        #              goal_proximity,  # proximidade do veículo ao objetivo (1 quando o veículo está no objetivo, 0 quando está longe)
        #              goal_direction,  # ângulo global em relação ao objetivo em radianos
        #              angle_diff]      # diferença de orientação entre a vaga de estacionamento e o veículo
        obs_low = np.array(
            [
                -np.pi,                           # theta
                -self.JACKKNIFE_LIMIT_RAD,        # beta
                -self.STEERING_LIMIT_RAD,         # alpha
            ]
            + [0.0] * self.simulation.vehicle.get_raycast_count()                     # raycast lengths (Normalized 0 to 1)
            #+ [MapEntity.MIN_COLLIDABLE_ENTITY_TYPE] * self.simulation.vehicle.get_raycast_count() # raycast object classes
            + [0.0, -np.pi, -np.pi],            # goal proximity, goal direction, angle_diff
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                np.pi,                            # theta
                self.JACKKNIFE_LIMIT_RAD,         # beta
                self.STEERING_LIMIT_RAD,          # alpha
            ]

            + [1.0] * self.simulation.vehicle.get_raycast_count()          
            #+ [MapEntity.MAX_COLLIDABLE_ENTITY_TYPE] * self.simulation.vehicle.get_raycast_count() # raycast object classes
            + [1.0, np.pi, np.pi], # goal proximity, goal direction, angle_diff
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

        self.simulation = self.simulation_loader.load_simulation()

        self.steps = 0
        self.total_reward = 0.0

        self.initial_distance_to_goal = self._calculate_distance_euclidean(self.simulation.vehicle.get_trailer_position(), self.simulation.map.get_parking_goal_position())
        self.last_distance_to_goal = self._calculate_distance_euclidean(self.simulation.vehicle.get_trailer_position(), self.simulation.map.get_parking_goal_position())

        # Build observation
        observation = self._build_observation()
        info = {}
        return observation, info

    def _build_observation(self) -> np.ndarray:
        # Get kinematic states
        theta = self.simulation.vehicle.get_theta()
        beta = self.simulation.vehicle.get_beta()
        alpha_current = self.simulation.vehicle.get_alpha()
        
        # Get raycast data (Unpacking the new tuple)
        raw_lengths, object_classes = self.simulation.vehicle.get_raycast_lengths_and_object_classes()

        # NORMALIZE: Divide lengths by the maximum sensor range to get values between [0, 1]
        # This helps the neural network learn faster and more stably.
        normalized_lengths = [length / self.SENSOR_RANGE_M for length in raw_lengths]

        # Calculate goal metrics
        goal_proximity = self._calculate_goal_proximity(self.simulation.vehicle.get_trailer_position(), self.simulation.map.get_parking_goal_position())
        goal_direction = self._calculate_goal_direction(self.simulation.vehicle.get_trailer_position(), self.simulation.map.get_parking_goal_position())
        angle_diff = self._calculate_angle_diff(self.simulation.vehicle.get_theta(), self.simulation.map.get_parking_goal_theta())

        # Construct the final array
        # Note: object_classes are integers, but will be cast to float32 to fit in the Box space
        observation = np.array(
            [theta, beta, alpha_current]
            + normalized_lengths
            #+ object_classes
            + [goal_proximity, goal_direction, angle_diff],
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

        self.simulation.move_vehicle(velocity, alpha, self.DT)

        ## Strongly punish zero/low speed
        if abs(velocity) < 0.1:
            reward += self.PUNISHMENT_ZERO_SPEED  # Penalidade muito maior para ficar parado
        else:
            reward += self.REWARD_VELOCITY

        if self.steps >= self.MAX_STEPS:
            truncated = True
        elif self._check_vehicle_parking():
            terminated = True
            reward = self._calculate_parking_reward(self.simulation.vehicle.get_theta(), self.simulation.map.get_parking_goal_theta())
        elif self._check_trailer_jackknife(self.simulation.vehicle.get_beta()):
            terminated = True
            reward = self.PUNISHMENT_JACKKNIFE
        else:
            collided = self._check_vehicle_collision()
            if collided:
                terminated = True
                reward = self.PUNISHMENT_COLLISION
            elif self._check_vehicle_overlap():
                reward += self.PUNISHMENT_OVERLAP
 
        new_distance_to_goal = self._calculate_distance_euclidean(self.simulation.vehicle.get_trailer_position(), self.simulation.map.get_parking_goal_position())
        #recompensa baseada no progresso em metros
        reward += (self.last_distance_to_goal - new_distance_to_goal) * self.REWARD_PROGRESS
        self.last_distance_to_goal = new_distance_to_goal

        observation = self._build_observation()
        self.total_reward += reward

        return observation, reward, terminated, truncated, info

    def render(self):
        rgb_array = Visualization.to_rgb_array(self.simulation, img_size=(320, 320))
        return rgb_array

    def close(self):
        pass

    def _check_vehicle_collision(self) -> bool:
        """Verifica se o veículo colidiu com alguma parede"""
        for entity in self.simulation.map.get_entities():
            if entity.type == MapEntity.ENTITY_WALL:
                if self.simulation.vehicle.check_collision(entity):
                    return True
        return False
                    
    def _check_vehicle_overlap(self) -> bool:
        """Verifica se o veículo passou por cima de uma vaga de estacionamento"""
        for entity in self.simulation.map.get_entities():
            if entity.type == MapEntity.ENTITY_PARKING_SLOT:
                if self.simulation.vehicle.check_collision(entity):
                    return True
        return False

    def _calculate_distance_euclidean(self, position: tuple[float, float], goal_position: tuple[float, float]):
        distance_to_goal = np.sqrt((position[0] - goal_position[0])**2 + (position[1] - goal_position[1])**2)
        return distance_to_goal

    def _calculate_goal_proximity(self, position: tuple[float, float], goal_position: tuple[float, float]):
        """Retorna a transformação inversa da distância euclidiana,  
        calculada como 1 / (1 + distância), que retorna 1 quando a distância é 0 e tende a 0 quando a distância tende
        ao infinito."""
        raw_distance = np.sqrt((position[0] - goal_position[0])**2 + (position[1] - goal_position[1])**2)
        return 1.0 / (1.0 + raw_distance)
        

    def _calculate_goal_direction(self, position: tuple[float, float], goal_position: tuple[float, float]):
        """Retorna a direção até o objetivo em radianos, normalizada para o intervalo [-pi, pi]"""
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
        return self._calculate_distance_euclidean(self.simulation.vehicle.get_trailer_position(), self.simulation.map.get_parking_goal_position()) < self.VEHICLE_PARKED_THRESHOLD_M

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