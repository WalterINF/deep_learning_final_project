import numpy as np
import gymnasium as gym
from src import Simulation
from src.SimulationConfigLoader import SimulationLoader
from src.Simulation import MapEntity, Vehicle
import src.Visualization as Visualization
from typing import Any, SupportsFloat
import math
from collections import deque
from src.heuristics import calcular_distancia_nao_holonomica_carro


class ParkingEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}

    ## ambiente
    GRID_RESOLUTION = 1.0  # Resolução do grid em metros para o pathfinding
    SENSOR_RANGE_M = 50.0 # raio do sensor
    SPEED_LIMIT_MS = 5.0 # velocidade maxima
    STEERING_LIMIT_RAD = float(np.deg2rad(28.0)) # angulo maximo de esterçamento
    DT = 0.2 # tempo do passo de simulação
    MAX_SECONDS = 90.0
    MAX_STEPS = int(MAX_SECONDS / DT)
    VEHICLE_PARKED_THRESHOLD_M = 1.0 # distancia minima entre centro do veículo e centro da vaga para considerar o veículo estacionado
    VEHICLE_PARKED_THRESHOLD_ANGLE = float(np.deg2rad(5.0)) # angulo maximo de desalinhamento para considerar o veículo estacionado

    ## recompensas
    REWARD_GOAL = 100.0 # recompensa por chegar ao objetivo 
    REWARD_ALIGNMENT = 0.0 # recompensa adicional por alinhar o veículo na vaga corretamente
    REWARD_PROGRESS = 1.0 # multiplicador da recompensa da heurística 
    REWARD_HEADING = 0.0 # recompensa por apontar em direção ao objetivo (desabilitar por enquanto)
    MAX_PUNISHMENT_TIME_PER_EPISODE = -10.0 # penalidade maxima por tempo 
    PUNISHMENT_TIME = MAX_PUNISHMENT_TIME_PER_EPISODE / MAX_STEPS # penalidade por tempo a cada passo
    PUNISHMENT_ZERO_SPEED = -0.1 # penalidade por velocidade zero 
    PUNISHMENT_COLLISION = -100.0 # penalidade por colisão com paredes ou outras vagas de estacionamento
    PUNISHMENT_JACKKNIFE = -100.0 # penalidade por jackknife

    ## heuristicas
    HEURSITICAS_DISPONIVEIS = {"nao_holonomica", "euclidiana", "nenhuma", "manhattan"}

    def __init__(self, seed = 0, heuristica = "nao_holonomica"):

        self.render_mode = "rgb_array"


        self.simulation_loader = SimulationLoader()

        self.simulation = self.simulation_loader.load_simulation()

        self.kinematic_car_params = KinematicCarParams(
            wheelbase=self.simulation.vehicle.get_wheelbase()
        )

        self.steps = 0
        self.total_reward = 0.0

        self.initial_distance_to_goal = self._calculate_holonomic_distance()
        self.last_distance_to_goal = self.initial_distance_to_goal

        # o estado  bruto é x = [x1, y1, θ1, θ2]
        # L é o comprimento da barra de reboque
        # z(x) = [
        # (x_1-x_g)*cos(theta_1g) + (y_1-y_g)*sin(theta_1g),
        # (theta_1 - theta_1g),
        # −(x_1 − x_g ) sin theta_1g + (y_1 − y_g ) cos theta_1g
        # z3 − L · sin(z2 − (theta_2 − theta_2g ))]
        # z = [z1, z2, z3, z4] 
        # Espaço de observação k = [z/z_normalizado], onde k é o vetor de erro normalizado
        # + O_local: percepção local dos obstáculos - 14 sensores raycast posicionados ao redor do veículo
        
        obs_low = np.array(
            [
                -150.0,         
                -self.STEERING_LIMIT_RAD,                         
                -math.pi,          
                -500.0,  
            ]
            + [0.0] * self.simulation.vehicle.get_raycast_count(),
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                150.0,         
                self.STEERING_LIMIT_RAD,       
                math.pi,         
                500.0,  
            ]
            + [0.0] * self.simulation.vehicle.get_raycast_count(),
            dtype=np.float32,
        )



        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: [v, alpha]
        act_low = np.array(
            [
                -self.SPEED_LIMIT_MS,             # v 
                -self.STEERING_LIMIT_RAD,         # alpha
            ],
            dtype=np.float32,
        )
        act_high = np.array(
            [
                self.SPEED_LIMIT_MS,             # v 
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
        
        self.initial_distance_to_goal = self._calculate_holonomic_distance()
        self.last_distance_to_goal = self.initial_distance_to_goal

        # Build observação
        observation = self._build_observation()
        info = {}
        return observation, info

    def _build_observation(self) -> np.ndarray:

        vehicle_state = self.simulation.vehicle.get_state()
        desired_state = [
            self.simulation.map.get_parking_goal_position()[0], 
            self.simulation.map.get_parking_goal_position()[1], 
            self.simulation.map.get_parking_goal_theta(),
            0.0] # phi_d = 0

        privileged_coordinates = compute_privileged_coordinates(
            state=vehicle_state,
            goal=desired_state,
            params=self.kinematic_car_params
        )
        k_vector = privileged_coordinates.z

        raw_lengths, _ = self.simulation.vehicle.get_raycast_lengths_and_object_classes()
        normalized_lengths = [length / self.SENSOR_RANGE_M for length in raw_lengths]

        return np.concatenate((k_vector, normalized_lengths))
    
    def _normalize_angle(self, angle: float) -> float:
        """Normaliza um ângulo para o intervalo [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        velocity, alpha = action
        self.steps += 1
        terminated = False
        truncated = False
        info = {}
        reward = self.PUNISHMENT_TIME  # penalidade base por passo de tempo

        self.simulation.move_vehicle(velocity, alpha, self.DT)

        goal_pos = self.simulation.map.get_parking_goal_position()
        goal_theta = self.simulation.map.get_parking_goal_theta()

        if abs(velocity) < 0.1:
            reward += self.PUNISHMENT_ZERO_SPEED
        
        new_distance_to_goal = self._calculate_holonomic_distance()
        
        # verificação de terminação
        if self.steps >= self.MAX_STEPS:
            truncated = True
        elif self._check_vehicle_parking():
            terminated = True
            reward = self._calculate_parking_reward(self.simulation.vehicle.get_theta(), goal_theta)
            info["is_success"] = True
        else:
            collided = self._check_vehicle_collision()
            if collided:
                terminated = True 
                reward = self.PUNISHMENT_COLLISION
        # recompensa por progresso (redução de distância) <-- aqui é definida a heurística
        reward += (self.last_distance_to_goal - new_distance_to_goal) * self.REWARD_PROGRESS
        self.last_distance_to_goal = new_distance_to_goal

        observation = self._build_observation()
        self.total_reward += reward

        if terminated or truncated:
            if "is_success" not in info:
                info["is_success"] = False

        return observation, reward, terminated, truncated, info

    def render(self):
        rgb_array = Visualization.to_rgb_array(
            self.simulation,
            img_size=(320, 320),
            distance_map=None,
            grid_resolution=self.GRID_RESOLUTION,
            observation=self._build_observation(),
            heuristic_value=self._calculate_holonomic_distance()
        )
        return rgb_array

    def close(self):
        pass

    def _check_vehicle_collision(self) -> bool:
        """Verifica se o veículo colidiu com alguma parede"""
        for entity in self.simulation.map.get_entities():
            if entity.is_collidable():
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

    def _calculate_holonomic_distance(self):
        position_car_x, position_car_y = self.simulation.vehicle.get_position()
        car_theta = self.simulation.vehicle.get_theta()
        position_goal_x, position_goal_y = self.simulation.map.get_parking_goal_position()
        goal_theta = self.simulation.map.get_parking_goal_theta()

        return calcular_distancia_nao_holonomica_carro(
            estado_atual=[position_car_x,position_car_y,car_theta],
            estado_alvo=[position_goal_x, position_goal_y, goal_theta],
            l_carro=self.simulation.vehicle.get_wheelbase()
        )

    def _check_vehicle_parking(self) -> bool:
        """Verifica se o veículo está dentro de uma vaga de estacionamento no ângulo correto."""
        return self._calculate_distance_euclidean(self.simulation.vehicle.get_position(), self.simulation.map.get_parking_goal_position()) < self.VEHICLE_PARKED_THRESHOLD_M and self.menor_diferenca_entre_angulos(self.simulation.vehicle.get_theta(), self.simulation.map.get_parking_goal_theta(), usarDirecao=False) < self.VEHICLE_PARKED_THRESHOLD_ANGLE

    def menor_diferenca_entre_angulos(self, angulo1: float, angulo2: float, usarDirecao: bool = False) -> float:
        """
        Calcula o menor ângulo em radianos entre dois ângulos distintos.    
        """
        # Normalizao angulo para o intervalo [-pi, pi] caso ele seja maior que pi ou menor que -pi
        angulo1 = self.normalize_if_needed(angulo1)
        angulo2 = self.normalize_if_needed(angulo2)
        
        # Calcula a diferença entre os ângulos normalizados
        diff = self.normalize_if_needed(angulo1 - angulo2)

        if usarDirecao:
            return diff
        else:
            return abs(diff)

    def normalize_if_needed(self, angle: float) -> float:
        """
        Normaliza um ângulo para o intervalo [-pi, pi] se necessário.
        
        Parâmetros:
            angle (float): O ângulo a ser normalizado.

        Retorna:
            float: O ângulo normalizado no intervalo [-pi, pi].
        """
        if angle > math.pi or angle < -math.pi:
            angle = self.normalize_angle(angle)
        
        return angle

    def normalize_angle(self, angle: float) -> float:
        """
        Normaliza um ângulo para o intervalo [-pi, pi].
        
        Parâmetros:
            angle (float): O ângulo a ser normalizado.

        Retorna:
            float: O ângulo normalizado no intervalo [-pi, pi].
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def soma_angulo_normalizado(self, angulo1: float, angulo2: float) -> float:
        """
        Soma dois ângulos e normaliza o resultado para o intervalo [-pi, pi].
        
        Parâmetros:
            angulo1 (float): O primeiro ângulo.
            angulo2 (float): O segundo ângulo.

        Retorna:
            float: A soma dos ângulos normalizada no intervalo [-pi, pi].
        """
        angulo1 = self.normalize_if_needed(angulo1)
        angulo2 = self.normalize_if_needed(angulo2)
        return self.normalize_if_needed(angulo1 + angulo2)

    def _calculate_parking_reward(self, vehicle_theta: float, parking_goal_theta: float) -> float:
            """Calcula a recompensa por estacionar o veículo baseada na orientação do veículo.
            Valor máximo quando a diferença de orientação é 0, valor mínimo quando é pi/2 ou maior.
            """
            angle_diff = self._calculate_angle_diff(vehicle_theta, parking_goal_theta)
            # se é maior que pi/2 (90 graus), apenas recompensa base (entrou de lado/ré errado)
            if abs(angle_diff) > math.pi/2:
                return self.REWARD_GOAL
            # Normaliza o erro para 0.0 a 1.0
            error_factor = abs(angle_diff) / (math.pi/2)
            # inverte para 1.0 (perfeito) a 0.0 (péssimo)
            alignment_quality = 1.0 - error_factor
            alignment_quality = alignment_quality ** 2
            alignment_bonus = alignment_quality * self.REWARD_ALIGNMENT

            return self.REWARD_GOAL + alignment_bonus

    # o estado  bruto é x = [x1, y1, θ1, θ2]
    # L é o comprimento da barra de reboque
    # z(x) = [
    # (x_1-x_g)*cos(theta_1g) + (y_1-y_g)*sin(theta_1g),
    # (theta_1 - theta_1g),
    # −(x_1 − x_g ) sin theta_1g + (y_1 − y_g ) cos theta_1g
    # z3 − L · sin(z2 − (theta_2 − theta_2g ))]
    def _compute_z1(self, x1: float, y1: float, xg: float, yg: float, theta1g: float) -> float:
        return (x1 - xg) * math.cos(theta1g) + (y1 - yg) * math.sin(theta1g)

    def _compute_z2(self, theta1: float, theta1g: float) -> float:
        return theta1 - theta1g

    def _compute_z3(self, x1: float, y1: float, xg: float, yg: float, theta1g: float) -> float:
        return -(x1 - xg) * math.sin(theta1g) + (y1 - yg) * math.cos(theta1g)

    def _compute_z4(self, z2, z3, L, theta2, theta2g):
        return z3 - L * math.sin(z2 - (theta2 - theta2g))

    def _compute_z_vector(
        self, 
        x1: float, # posição do trator x
        y1: float, # posição do trator y
        xg: float, #posiçaõ do alvo
        yg: float, #posição do alvo
        theta1: float, # ângulo de orientação do trator
        theta1g: float, # ângulo de orientação do alvo
        theta2: float, # ângulo de orientação do trailer
        theta2g: float, # ângulo de orientação do alvo
        L: float) -> np.ndarray:

        z1 = self._compute_z1(x1, y1, xg, yg, theta1g)
        z2 = self._compute_z2(theta1, theta1g)
        z3 = self._compute_z3(x1, y1, xg, yg, theta1g)
        z4 = self._compute_z4(z2, z3, L, theta2, theta2g)
        return np.array([z1, z2, z3, z4])

    def _normalize_z_vector(self, z_vector: np.ndarray) -> np.ndarray:
        return z_vector / np.linalg.norm(z_vector)



# Implementação das coordenadas privilegiadas e custo para o carro cinemático
# Baseado em: "MPC for Non-Holonomic Vehicles Beyond Differential-Drive"

import numpy as np
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class KinematicCarParams:
    """
    Parâmetros do carro cinemático.
    
    :param wheelbase: Distância entre eixos (l) em metros
    :type wheelbase: float
    
    :param state_weights: Pesos para estados (q1, q2, q3, q4)
    :type state_weights: Tuple[float, float, float, float]
    
    :param input_weights: Pesos para entradas (r1, r2)
    :type input_weights: Tuple[float, float]
    """
    wheelbase: float = 0.2
    state_weights: Tuple[float, float, float, float] = (1.0, 1.0, 2.0, 3.0)
    input_weights: Tuple[float, float] = (1.0, 1.0)


class PrivilegedCoordinates(NamedTuple):
    """
    Resultado do cálculo das coordenadas privilegiadas.
    
    :param z: Vetor de coordenadas privilegiadas [z1, z2, z3, z4]
    :type z: np.ndarray
    
    :param weights: Pesos homogêneos w = (w1, w2, w3, w4)
    :type weights: Tuple[int, int, int, int]
    
    :param r: Parâmetros de homogeneidade r = (r1, r2, r3, r4)
    :type r: Tuple[int, int, int, int]
    """
    z: np.ndarray
    weights: Tuple[int, int, int, int]
    r: Tuple[int, int, int, int]


def compute_privileged_coordinates(
    state: np.ndarray,
    goal: np.ndarray,
    params: KinematicCarParams
) -> PrivilegedCoordinates:
    """
    Calcula as coordenadas privilegiadas z do estado atual em relação ao goal.
    
    Para o carro cinemático, as coordenadas privilegiadas são obtidas através
    do algoritmo de Bellaïche, que transforma o espaço de estados original
    em um sistema de coordenadas que preserva a controlabilidade do sistema
    não holonômico.
    
    **Transformação (Passo 1 - Bellaïche):**
    
    .. math::
    
        y = A^{-T}(x - d)
    
    onde :math:`A` é a matriz formada pelos campos vetoriais avaliados no setpoint.
    
    Para o carro cinemático na origem (:math:`d = 0`):
    
    .. math::
    
        y = \\begin{bmatrix} x_1 \\\\ x_4 \\\\ -l \\cdot x_3 \\\\ l \\cdot x_2 \\end{bmatrix}
    
    **Parâmetros de Homogeneidade:**
    
    - Pesos: :math:`w = (1, 1, 2, 3)`
    - Parâmetros r: :math:`r = (1, 1, 2, 3)`
    - Grau de não holonomia: :math:`\\rho = 3`
    
    :param state: Estado atual do veículo [x, y, θ, φ]
    :type state: np.ndarray
    
    :param goal: Estado desejado (setpoint) [x_d, y_d, θ_d, φ_d]
    :type goal: np.ndarray
    
    :param params: Parâmetros do carro cinemático
    :type params: KinematicCarParams
    
    :returns: Coordenadas privilegiadas e parâmetros de homogeneidade
    :rtype: PrivilegedCoordinates
    
    :raises ValueError: Se state ou goal não tiverem dimensão 4
    
    .. note::
    
        Para setpoints com :math:`\\phi_d = 0`, vale :math:`z = y`.
        Para setpoints gerais, o segundo passo do algoritmo de Bellaïche
        é necessário.
    
    **Exemplo de uso:**
    
    .. code-block:: python
    
        params = KinematicCarParams(wheelbase=0.2)
        state = np.array([0.5, 0.3, 0.1, 0.05])
        goal = np.array([0.0, 0.0, 0.0, 0.0])
        
        result = compute_privileged_coordinates(state, goal, params)
        print(f"z = {result.z}")
        print(f"Pesos w = {result.weights}")
    
    **Referências:**
    
    .. [1] Rosenfelder et al. "MPC for Non-Holonomic Vehicles Beyond 
           Differential-Drive", arXiv:2205.11400, 2022.
    .. [2] Jean, F. "Control of Nonholonomic Systems: From Sub-Riemannian
           Geometry to Motion Planning", Springer, 2014.
    """
    if len(state) != 4 or len(goal) != 4:
        raise ValueError("State e goal devem ter dimensão 4: [x, y, θ, φ]")
    
    state = np.asarray(state, dtype=float)
    goal = np.asarray(goal, dtype=float)
    l = params.wheelbase
    
    # Diferença do estado em relação ao goal
    dx = state - goal
    x1, x2, x3, x4 = dx[0], dx[1], dx[2], dx[3]
    
    # =========================================================================
    # Passo 1: Transformação de Bellaïche (Eq. 18 do paper)
    # =========================================================================
    # Matriz do referencial adaptado avaliado na origem:
    # [X1, X2, X3, X4]_0 forma a base do espaço tangente
    #
    # Para o carro cinemático com setpoint na origem:
    # y = [x1, x4, -l*x3, l*x2]^T
    # =========================================================================
    
    y = np.array([
        x1,           # y1 = x1 (direção facilmente controlável)
        x4,           # y2 = φ (ângulo de esterçamento)
        -l * x3,      # y3 = -l*θ (orientação escalada)
        l * x2        # y4 = l*y (direção mais difícil de controlar)
    ])
    
    # =========================================================================
    # Passo 2: Correções polinomiais (Eq. 8 do paper)
    # =========================================================================
    # Para setpoints com φ_d = 0 (goal[3] = 0), as derivadas não holonômicas
    # relevantes desaparecem, resultando em z = y.
    #
    # Para setpoints gerais, seria necessário calcular h_{4,2}(y1, y2, y3)
    # =========================================================================
    
    if np.abs(goal[3]) < 1e-10:
        # Caso simplificado: z = y
        z = y.copy()
    else:
        # Caso geral: aplicar segundo passo de Bellaïche
        # z_j = y_j - sum_{k=2}^{w_j-1} h_{j,k}(y)
        # Para o carro, apenas z4 requer correção
        z = y.copy()
        # h_{4,2} envolve derivadas não holonômicas de ordem superior
        # Implementação completa requer cálculo simbólico adicional
        # Por simplicidade, mantemos z = y (aproximação válida próximo à origem)
    
    # Parâmetros de homogeneidade do carro cinemático
    weights = (1, 1, 2, 3)  # w = (w1, w2, w3, w4)
    r = (1, 1, 2, 3)        # r = (r1, r2, r3, r4)
    
    return PrivilegedCoordinates(z=z, weights=weights, r=r)