import numpy as np
import gymnasium as gym
from src.SimulationConfigLoader import SimulationLoader
from src.Simulation import MapEntity
import src.Visualization as Visualization
from typing import Any, SupportsFloat
import math
from collections import deque


class ParkingEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}


    ## ambiente
    GRID_RESOLUTION = 1.0  # Resolução do grid em metros para o pathfinding
    SENSOR_RANGE_M = 50.0 # raio do sensor
    SPEED_LIMIT_MS = 5.0 # velocidade maxima
    STEERING_LIMIT_RAD = float(np.deg2rad(28.0)) # angulo maximo de esterçamento
    JACKKNIFE_LIMIT_RAD = float(np.deg2rad(65.0)) # angulo maximo de jackknife
    DT = 0.2 # tempo do passo de simulação
    MAX_SECONDS = 90.0
    MAX_STEPS = int(MAX_SECONDS / DT)
    VEHICLE_PARKED_THRESHOLD_M = 3.0 # distancia minima entre centro do trailer e centro da vaga para considerar o veículo estacionado


    ## recompensas
    REWARD_GOAL = 100.0 # recompensa por chegar ao objetivo 
    REWARD_ALIGNMENT = 100.0 # recompensa adicional por alinhar o veículo na vaga corretamente
    REWARD_PROGRESS = 1.0 # multiplicador da recompensa da heurística 
    REWARD_HEADING = 0.0 # recompensa por apontar em direção ao objetivo (desabilitar por enquanto)
    MAX_PUNISHMENT_TIME_PER_EPISODE = -10.0 # penalidade maxima por tempo 
    PUNISHMENT_TIME = MAX_PUNISHMENT_TIME_PER_EPISODE / MAX_STEPS # penalidade por tempo a cada passo
    PUNISHMENT_ZERO_SPEED = -0.1 # penalidade por velocidade zero 
    PUNISHMENT_COLLISION = -100.0 # penalidade por colisão com paredes ou outras vagas de estacionamento
    PUNISHMENT_JACKKNIFE = -100.0 # penalidade por jackknife



    def __init__(self, seed = 0):

        self.render_mode = "rgb_array"

        self.simulation_loader = SimulationLoader()

        self.simulation = self.simulation_loader.load_simulation()

        self._compute_navigation_map()

        self.steps = 0
        self.total_reward = 0.0
    
        self.initial_distance_to_goal = self._get_geodesic_distance(self.simulation.vehicle.get_trailer_position())
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
                -1.0,         # z1: componente x rotacionada
                -1.0,                           # z2: diferença de ângulo theta1
                -1.0,          # z3: componente y rotacionada
                -1.0,  # z4: z3 - L*sin(...)
            ]
            + [0.0] * self.simulation.vehicle.get_raycast_count(),
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                1.0,           # z1
                1.0,                           # z2
                1.0,          # z3
                1.0,  # z4
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

        self._compute_navigation_map()

        self.steps = 0
        self.total_reward = 0.0
    
        self.initial_distance_to_goal = self._get_geodesic_distance(self.simulation.vehicle.get_trailer_position())
        self.last_distance_to_goal = self.initial_distance_to_goal

        # Build observação
        observation = self._build_observation()
        info = {}
        return observation, info

    def _build_observation(self) -> np.ndarray:
        x1 = self.simulation.vehicle.get_position()[0]
        y1 = self.simulation.vehicle.get_position()[1]
        xg = self.simulation.map.get_parking_goal_position()[0]
        yg = self.simulation.map.get_parking_goal_position()[1]
        theta1 = self.simulation.vehicle.get_theta()
        theta1g = self.simulation.map.get_parking_goal_theta()
        theta2 = self.simulation.vehicle.get_trailer_theta()
        theta2g = self.simulation.map.get_parking_goal_theta()
        L = self.simulation.vehicle.get_comprimento_trailer()

        z_vector = self._compute_z_vector(
            x1, y1, xg, yg, theta1, theta1g, theta2, theta2g, L
        )

        k_vector = self._normalize_z_vector(z_vector)

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

        trailer_pos = self.simulation.vehicle.get_trailer_position()
        goal_pos = self.simulation.map.get_parking_goal_position()
        goal_theta = self.simulation.map.get_parking_goal_theta()
        tractor_theta = self.simulation.vehicle.get_theta()
        trailer_theta = self.simulation.vehicle.get_trailer_theta()

        if abs(velocity) < 0.1:
            reward += self.PUNISHMENT_ZERO_SPEED
        
        new_distance_to_goal = self._get_geodesic_distance(trailer_pos)
        
        # verificação de terminação
        if self.steps >= self.MAX_STEPS:
            truncated = True
        elif self._check_vehicle_parking():
            terminated = True
            reward = self._calculate_parking_reward(trailer_theta, goal_theta)
            info["is_success"] = True
        elif self._check_trailer_jackknife(self.simulation.vehicle.get_beta()):
            terminated = True
            reward = self.PUNISHMENT_JACKKNIFE
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
            observation=self._build_observation()
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

    def _check_vehicle_parking(self) -> bool:
        """Verifica se o trailer do veículo está dentro de uma vaga de estacionamento."""
        return self._calculate_distance_euclidean(self.simulation.vehicle.get_trailer_position(), self.simulation.map.get_parking_goal_position()) < self.VEHICLE_PARKED_THRESHOLD_M

    def _calculate_parking_reward(self, trailer_theta: float, parking_goal_theta: float) -> float:
            """Calcula a recompensa por estacionar o veículo baseada na orientação do TRAILER.
            Valor máximo quando a diferença de orientação é 0, valor mínimo quando é pi/2 ou maior.
            """
            angle_diff = self._calculate_angle_diff(trailer_theta, parking_goal_theta)
            
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

    def _check_trailer_jackknife(self, beta: float) -> bool:
        """Verifica se o trailer do veículo está em jackknife."""
        return abs(beta) > self.JACKKNIFE_LIMIT_RAD

    def _compute_navigation_map(self):
        """
        Calcula um mapa de distâncias usando BFS (Breadth-First Search) a partir do objetivo.
        Isso cria um campo potencial que guia o agente contornando obstáculos (paredes).
        """
        map_w, map_h = self.simulation.map.get_size()
        self.map_width = int(math.ceil(map_w / self.GRID_RESOLUTION))
        self.map_height = int(math.ceil(map_h / self.GRID_RESOLUTION))
        
        # Inicializa grid: -1 = não visitado, -2 = obstáculo
        self.distance_map = np.full((self.map_width, self.map_height), -1.0, dtype=np.float32)
        
        # 1. Rasteriza obstáculos (entidades colidíveis)
        for entity in self.simulation.map.get_entities():
            if entity.is_collidable():
                self._rasterize_entity(entity, -2.0)

        # 2. Inicializa BFS a partir do objetivo
        queue = deque()
        goal = self.simulation.map.get_parking_goal()
        
        # Marca células do objetivo com distância 0 e adiciona à fila
        self._rasterize_entity(goal, 0.0, queue)
        
        # 3. Executa BFS (8 vizinhos para melhor aproximação)
        neighbors = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0), 
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        
        while queue:
            cx, cy = queue.popleft()
            current_dist = self.distance_map[cx, cy]
            
            for dx, dy, cost in neighbors:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if self.distance_map[nx, ny] == -1.0: # Não visitado e livre
                        new_dist = current_dist + cost * self.GRID_RESOLUTION
                        self.distance_map[nx, ny] = new_dist
                        queue.append((nx, ny))

    def _rasterize_entity(self, entity, value, queue=None):
        """Marca as células do grid ocupadas por uma entidade."""
        bbox = entity.get_bounding_box()
        corners = bbox.get_corners()
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        
        # Limites no grid
        min_x = max(0, int(min(xs) / self.GRID_RESOLUTION))
        max_x = min(self.map_width - 1, int(max(xs) / self.GRID_RESOLUTION))
        min_y = max(0, int(min(ys) / self.GRID_RESOLUTION))
        max_y = min(self.map_height - 1, int(max(ys) / self.GRID_RESOLUTION))
        
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                # Verifica colisão com o centro da célula
                cx = (i + 0.5) * self.GRID_RESOLUTION
                cy = (j + 0.5) * self.GRID_RESOLUTION
                if bbox.contains_point((cx, cy)):
                    # Se for obstáculo (-2) ou se estamos inicializando o objetivo (e não é parede)
                    if value == -2.0 or self.distance_map[i, j] != -2.0:
                        self.distance_map[i, j] = value
                        if queue is not None:
                            queue.append((i, j))

    def _get_geodesic_distance(self, position: tuple[float, float]) -> float:
        """Retorna a distância de navegação da posição até o objetivo."""
        x, y = position
        ix = int(x / self.GRID_RESOLUTION)
        iy = int(y / self.GRID_RESOLUTION)
        
        if 0 <= ix < self.map_width and 0 <= iy < self.map_height:
             dist = self.distance_map[ix, iy]
             if dist >= 0:
                 return dist
        
        # Fallback para distância Euclidiana se fora do mapa ou dentro de obstáculo
        return self._calculate_distance_euclidean(position, self.simulation.map.get_parking_goal_position())

    
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