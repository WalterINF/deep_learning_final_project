import numpy as np
import gymnasium as gym
from src.SimulationConfigLoader import SimulationLoader
import src.Visualization as Visualization
from typing import Any, SupportsFloat, Tuple, NamedTuple
import math
from src.heuristics import calcular_distancia_nao_holonomica_carro
from dataclasses import dataclass


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
    VEHICLE_PARKED_THRESHOLD_M = 2.0 # distancia minima entre centro do TRAILER e centro da vaga para considerar o veículo estacionado
    VEHICLE_PARKED_THRESHOLD_ANGLE = float(np.deg2rad(5.0)) # angulo maximo de desalinhamento do TRAILER para considerar o veículo estacionado
    JACKKNIFE_LIMIT_RAD = float(np.deg2rad(45.0)) # angulo maximo de desalinhamento do TRAILER para considerar o veículo em jackknife (45 graus)

    ## recompensas
    REWARD_GOAL = 100.0 # recompensa por chegar ao objetivo 
    REWARD_PROGRESS_MULTIPLIER = 1.0 # multiplicador da recompensa da heurística 
    MAX_PUNISHMENT_TIME_PER_EPISODE = -10.0 # penalidade maxima por tempo 
    PUNISHMENT_TIME = MAX_PUNISHMENT_TIME_PER_EPISODE / MAX_STEPS # penalidade por tempo a cada passo
    PUNISHMENT_ZERO_SPEED = -0.1 # penalidade por velocidade zero 
    PUNISHMENT_COLLISION = -100.0 # penalidade por colisão com paredes ou outras vagas de estacionamento
    PUNISHMENT_JACKKNIFE = -100.0 # penalidade por jackknife

    ## heuristicas
    HEURSITICAS_DISPONIVEIS = {"nao_holonomica", "euclidiana", "manhattan", "nenhuma"}

    def __init__(self, seed: int = 0, heuristica: str = "nao_holonomica"):
        """
        Inicializa o ambiente de estacionamento.

        :param seed: Semente para o ambiente.
        :param heuristica: Heurística de distância a ser usada (nao_holonomica, euclidiana, manhattan, nenhuma)
        """
        if heuristica not in self.HEURSITICAS_DISPONIVEIS:
            raise ValueError(f"Heurística inválida: {heuristica}")
        self.heuristica = heuristica

        self.render_mode = "rgb_array"

        self.simulation_loader = SimulationLoader()

        self.simulation = self.simulation_loader.load_simulation()

        self.tractor_trailer_params = TractorTrailerParams()
        self.tractor_trailer_geometry = TractorTrailerGeometry(params=self.tractor_trailer_params)

        self.steps = 0
        self.total_reward = 0.0

        self.initial_distance_to_goal = self._calculate_heuristic_value(self.heuristica)
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
                -150.0,                         
                -150.0,          
                -150.0,  
            ]
            + [0.0] * self.simulation.vehicle.get_raycast_count(),
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                150.0,         
                150.0,       
                150.0,         
                150.0,  
            ]
            + [50.0] * self.simulation.vehicle.get_raycast_count(),
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
        
        self.initial_distance_to_goal = self._calculate_heuristic_value(self.heuristica)
        self.last_distance_to_goal = self.initial_distance_to_goal

        # Build observação
        observation = self._build_observation()
        info = {}
        return observation, info

    def _build_observation(self) -> np.ndarray:

        tractor_state = self.simulation.vehicle.get_tractor_state()
        desired_state = [
            self.simulation.map.get_parking_goal_position()[0], 
            self.simulation.map.get_parking_goal_position()[1], 
            self.simulation.map.get_parking_goal_theta(),
            0.0] # phi_d = 0

        privileged_coordinates = self.tractor_trailer_geometry.privileged_coords(
            state=tractor_state,
            goal=desired_state,
        )

        raw_lengths = self.simulation.vehicle.get_raycast_lengths()
        normalized_lengths = [length / self.SENSOR_RANGE_M for length in raw_lengths]

        return np.concatenate((privileged_coordinates, normalized_lengths))
    
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

        if abs(velocity) < 0.1:
            reward += self.PUNISHMENT_ZERO_SPEED
        

        # verificação de terminação
        if self.steps >= self.MAX_STEPS:
            truncated = True
        elif self._check_vehicle_jackknife():
            terminated = True
            reward = self.PUNISHMENT_JACKKNIFE
        elif self._check_vehicle_parking():
            terminated = True
            reward = self.REWARD_GOAL
            info["is_success"] = True
        elif self._check_vehicle_collision():
            terminated = True 
            reward = self.PUNISHMENT_COLLISION

        # recompensa por progresso baseada na heurística
        new_distance_to_goal = self._calculate_heuristic_value(self.heuristica)
        reward += (self.last_distance_to_goal - new_distance_to_goal) * self.REWARD_PROGRESS_MULTIPLIER
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
            heuristic_value=self._calculate_heuristic_value(self.heuristica)
        )
        return rgb_array

    def close(self):
        pass

    def _calculate_heuristic_value(self, heuristic: str) -> float:
        if heuristic == "nao_holonomica":
            return self._calculate_nao_holonomic_distance()
        elif heuristic == "euclidiana":
            return self._calculate_distance_euclidean(self.simulation.vehicle.get_tractor_position(), self.simulation.map.get_parking_goal_position())
        elif heuristic == "manhattan":
            return self._calculate_distance_manhattan(self.simulation.vehicle.get_tractor_position(), self.simulation.map.get_parking_goal_position())
        elif heuristic == "nenhuma":
            return 0.0
        else:
            raise ValueError(f"Heurística inválida: {heuristic}")

    def _check_vehicle_jackknife(self) -> bool:
        """Verifica se o veículo está em jackknife."""
        return abs(self.simulation.vehicle.get_tractor_beta()) > self.JACKKNIFE_LIMIT_RAD


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

    def _calculate_distance_manhattan(self, position: tuple[float, float], goal_position: tuple[float, float]):
        distance_to_goal = abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1])
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

    def _calculate_nao_holonomic_distance(self):
        goal_state = [
            self.simulation.map.get_parking_goal_position()[0], 
            self.simulation.map.get_parking_goal_position()[1], 
            self.simulation.map.get_parking_goal_theta(), 
            0.0] # phi_d = 0
        
        tractor_state = self.simulation.vehicle.get_tractor_state()
        return self.tractor_trailer_geometry.homogeneous_distance(
            state=tractor_state,
            goal=goal_state,
        )
        


    def _check_vehicle_parking(self) -> bool:
        """Verifica se o veículo está dentro de uma vaga de estacionamento no ângulo correto."""
        return (
            self._calculate_distance_euclidean(self.simulation.vehicle.get_trailer_position(), self.simulation.map.get_parking_goal_position()) < self.VEHICLE_PARKED_THRESHOLD_M 
            and self.menor_diferenca_entre_angulos(self.simulation.vehicle.get_trailer_theta() - math.pi, self.simulation.map.get_parking_goal_theta(), usarDirecao=False) < self.VEHICLE_PARKED_THRESHOLD_ANGLE
        )

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


# Implementação das coordenadas privilegiadas e custo para o carro cinemático
# Baseado em: "MPC for Non-Holonomic Vehicles Beyond Differential-Drive"

@dataclass
class TractorTrailerParams:
    """
    Parâmetros geométricos do modelo trator-reboque.

    :param D: Distância do eixo traseiro ao ponto de articulação. (wheelbase do trator)
    :param L: Comprimento do reboque até a quinta roda
    :param epsilon: Parâmetro geométrico relacionado à posição do eixo de articulação.
    """
    D: float = 4.7
    L: float = 6.53
    epsilon: float = 0.736 #nosso caso - negativo


class TractorTrailerGeometry:
    """
    Geometria e métricas homogêneas para um modelo cinemático trator-reboque.

    Esta classe implementa:

    * Cálculo do erro entre ``state`` e ``goal`` no referencial do alvo.
    * Transformação desse erro em coordenadas privilegiadas.
    * Cálculo da distância homogênea (custo de estado) usando pesos não-holonômicos.

    Convenção de estado
    --------------------

    O estado é dado por:

    .. math::

        x = [x, y, \\theta, \\beta]^T

    onde:

    * ``x``: posição longitudinal no mundo
    * ``y``: posição lateral no mundo
    * ``theta``: orientação do trator
    * ``beta``: ângulo de articulação (reboque/trator)

    As entradas (não usadas diretamente aqui) seriam tipicamente:

    .. math::

        u = [v, \\alpha]^T

    com ``v`` velocidade e ``alpha`` ângulo de direção.
    """

    def __init__(self, params: TractorTrailerParams | None = None):
        self.params = params or TractorTrailerParams()
        # Pesos não-holonômicos dos estados (ordem de crescimento)
        # r1 = 1, r2 = 1, r3 = 2, r4 = 3 (exemplo típico)
        self.r: Tuple[int, int, int, int] = (1, 1, 2, 3)
        # Grau de homogeneidade do pseudo-norma (escolha padrão: d = 2 * sum(r))
        self.d: int = 2 * sum(self.r)

    # ------------------------------------------------------------------
    # 1. Erro no referencial do alvo
    # ------------------------------------------------------------------
    def _relative_error(self, state: np.ndarray, goal: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calcula o erro entre ``state`` e ``goal`` no frame do alvo.

        Parameters
        ----------
        state : np.ndarray
            Estado atual, ``[x, y, theta, beta]``.
        goal : np.ndarray
            Estado alvo, ``[x_g, y_g, theta_g, beta_g]``.

        Returns
        -------
        (x_rel, y_rel, theta_rel, beta_rel) : tuple of float
            Erros nas coordenadas do frame do alvo.

        Notas
        -----
        1. Transladamos o estado atual para o sistema com origem no goal:

           .. math::

               \\Delta x = x - x_g, \\quad \\Delta y = y - y_g

        2. Rotacionamos (\\Delta x, \\Delta y) pelo ângulo do alvo
           para obter o erro no frame do alvo:

           .. math::

               \\begin{bmatrix}
                   x_\\text{rel} \\\\
                   y_\\text{rel}
               \\end{bmatrix}
               =
               R(-\\theta_g)
               \\begin{bmatrix}
                   \\Delta x \\\\
                   \\Delta y
               \\end{bmatrix}

        3. Os erros de orientação e articulação são:

           .. math::

               \\theta_\\text{rel} = \\theta - \\theta_g, \\quad
               \\beta_\\text{rel} = \\beta - \\beta_g
        """
        x, y, th, be = state
        xg, yg, thg, beg = goal

        dx = x - xg
        dy = y - yg
        c = float(np.cos(thg))
        s = float(np.sin(thg))

        # Rotação para o frame do alvo
        x_rel = c * dx + s * dy
        y_rel = -s * dx + c * dy

        th_rel = th - thg
        be_rel = be - beg

        return float(x_rel), float(y_rel), float(th_rel), float(be_rel)

    # ------------------------------------------------------------------
    # 2. Coordenadas privilegiadas z(state, goal)
    # ------------------------------------------------------------------
    def privileged_coords(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Calcula as coordenadas privilegiadas do erro entre dois estados.

        Parameters
        ----------
        state : np.ndarray
            Estado atual, ``[x, y, theta, beta]``.
        goal : np.ndarray
            Estado alvo, ``[x_g, y_g, theta_g, beta_g]``.

        Returns
        -------
        z : np.ndarray
            Vetor de coordenadas privilegiadas ``[z1, z2, z3, z4]``.

        Conceito
        --------
        As coordenadas privilegiadas são um sistema de coordenadas
        adaptado à estrutura não-holonômica do sistema. Para o modelo
        trator-reboque simples, usamos uma forma inspirada na literatura
        e em exemplos de carros não-holonômicos:

        1. Primeiro calculamos o erro relativo:

           .. math::

               (x_\\text{rel}, y_\\text{rel}, \\theta_\\text{rel}, \\beta_\\text{rel})

        2. Em seguida, definimos:

           .. math::

               z_1 &= x_\\text{rel} \\\\
               z_2 &= L \\beta_\\text{rel} - \\theta_\\text{rel} \\\\
               z_3 &= y_\\text{rel} - L^2 \\beta_\\text{rel} \\\\
               z_4 &= \\frac{L^2 \\beta_\\text{rel}}{\\varepsilon - 1}

        onde ``L`` e ``epsilon`` vêm dos parâmetros geométricos.
        """
        L = self.params.L
        eps = self.params.epsilon

        x_rel, y_rel, th_rel, be_rel = self._relative_error(state, goal)

        z1 = x_rel
        z2 = L * be_rel - th_rel
        z3 = y_rel - (L ** 2) * be_rel
        # Evitar divisão por zero se epsilon for 1 (caso degenerado)
        denom = eps - 1.0
        if abs(denom) < 1e-9:
            raise ValueError("epsilon muito próximo de 1: coordenada z4 torna-se singular.")
        z4 = (L ** 2) * be_rel / denom

        return np.array([z1, z2, z3, z4], dtype=float)

    # ------------------------------------------------------------------
    # 3. Distância homogênea (custo de estado)
    # ------------------------------------------------------------------
    def homogeneous_distance(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        q: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> float:
        """
        Calcula a distância homogênea (custo) entre dois estados.

        A métrica utiliza coordenadas privilegiadas e pesos não-holonômicos,
        penalizando erros em direções mais difíceis com potências adequadas.

        Parameters
        ----------
        state : np.ndarray
            Estado atual, ``[x, y, theta, beta]``.
        goal : np.ndarray
            Estado alvo, ``[x_g, y_g, theta_g, beta_g]``.
        q : tuple of float, optional
            Pesos dos estados no custo, ``(q1, q2, q3, q4)``.

        Returns
        -------
        cost : float
            Valor escalar da distância homogênea.

        Fórmula
        -------
        Seja :math:`z = (z_1, z_2, z_3, z_4)` o vetor de coordenadas
        privilegiadas e :math:`r = (r_1, r_2, r_3, r_4)` os pesos
        não-holonômicos (ordens). Definimos o grau de homogeneidade ``d`` e:

        .. math::

            \\ell(z) = \\sum_{i=1}^4 q_i \\; |z_i|^{d / r_i}

        onde tipicamente escolhemos :math:`d = 2 \\sum_i r_i`.

        Intuição
        --------
        * Direções mais fáceis (menor :math:`r_i`) recebem expoentes maiores.
        * Direções mais difíceis (maior :math:`r_i`) recebem expoentes menores,
          o que as torna relativamente mais caras perto da origem.
        """
        z = self.privileged_coords(state, goal)
        r = self.r
        d = float(self.d)

        cost = 0.0
        for i in range(4):
            exp = d / r[i]
            cost += q[i] * (abs(z[i]) ** exp)
        return float(math.log10(1.0 + cost))


# ----------------------------------------------------------------------
# Exemplo rápido de uso
# ----------------------------------------------------------------------
if __name__ == "__main__":
    params = TractorTrailerParams(D=4.7, L=6.53, epsilon=0.736)
    geom = TractorTrailerGeometry(params)

    state = np.array([0.0, 0.0, 0.0, 5.00])
    goal = np.array([0.0, 0.0, 0.0, 0.0])

    z = geom.privileged_coords(state, goal)
    dist = geom.homogeneous_distance(state, goal)

    print("z =", z)
    print("homogeneous distance =", dist)