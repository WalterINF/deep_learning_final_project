import math
from casadi import cos, sin
import random
from typing import Dict


class RaycastResult:
    origin_x: float
    origin_y: float
    theta: float
    length: float
    entity: 'MapEntity'

    def __init__(self, origin_x: float, origin_y: float, theta: float, length: float, entity: 'MapEntity' = None):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.theta = theta
        self.length = length
        self.entity = entity

class BoundingBox:

    position_x: float
    position_y: float
    width: float
    height: float
    theta: float # ângulo em relação ao eixo x do mapa

    def __init__(self, position_x: float, position_y: float, width: float, height: float, theta: float):
        self.position_x = position_x
        self.position_y = position_y
        self.width = width
        self.height = height
        self.theta = theta

    def check_collision(self, other: 'BoundingBox') -> bool:
        """ Verifica se há colisão entre duas bounding boxes 
        Args:
            other: BoundingBox a ser comparada
        Returns:
            bool: True se houver colisão, False caso contrário
        """
        # Oriented Bounding Box (OBB) intersection via Separating Axis Theorem (SAT)
        # Local axes for each box (unit vectors)
        cos1 = math.cos(self.theta)
        sin1 = math.sin(self.theta)
        cos2 = math.cos(other.theta)
        sin2 = math.sin(other.theta)

        # Axes for self
        u1 = (cos1, sin1)          # axis along width
        v1 = (-sin1, cos1)         # axis along height
        # Axes for other
        u2 = (cos2, sin2)
        v2 = (-sin2, cos2)

        half_w1 = self.width / 2.0
        half_h1 = self.height / 2.0
        half_w2 = other.width / 2.0
        half_h2 = other.height / 2.0

        # Vector between centers
        center_dx = other.position_x - self.position_x
        center_dy = other.position_y - self.position_y

        def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
            return a[0] * b[0] + a[1] * b[1]

        axes = [u1, v1, u2, v2]
        eps = 1e-9

        for axis in axes:
            # Projection radii of each box onto current axis
            r1 = half_w1 * abs(dot(u1, axis)) + half_h1 * abs(dot(v1, axis))
            r2 = half_w2 * abs(dot(u2, axis)) + half_h2 * abs(dot(v2, axis))

            # Distance between centers projected onto axis
            dist = abs(center_dx * axis[0] + center_dy * axis[1])

            # If there is a separating axis, boxes do not intersect
            if dist > (r1 + r2) + eps:
                return False

        # No separating axis found -> boxes intersect
        return True

    def contains_point(self, point: tuple[float, float]) -> bool:
        """verifica se um ponto está dentro da bounding box"""
        # Transform point to this box's local coordinates (translate, then rotate by -theta)
        dx = point[0] - self.position_x
        dy = point[1] - self.position_y

        cos_theta = math.cos(-self.theta)
        sin_theta = math.sin(-self.theta)

        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta

        half_w = self.width / 2.0
        half_h = self.height / 2.0
        eps = 1e-9

        return (-half_w - eps) <= local_x <= (half_w + eps) and (-half_h - eps) <= local_y <= (half_h + eps)

    def contains_bounding_box(self, other: 'BoundingBox') -> bool:
        """verifica se uma bounding box está dentro da bounding box"""
        # All 4 corners of the other OBB must be inside this OBB
        for corner in other.get_corners():
            if not self.contains_point(corner):
                return False
        return True
    
    def get_corners(self) -> list[tuple[float, float]]:
        """Calcula os quatro cantos da bounding box em coordenadas globais"""
        # Half dimensions
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        
        # Local corners (relative to center, before rotation)
        local_corners = [
            (-half_w, -half_h),  # bottom-left
            (half_w, -half_h),   # bottom-right
            (half_w, half_h),    # top-right
            (-half_w, half_h)    # top-left
        ]
        
        # Rotate and translate to global coordinates
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        corners = []
        for local_x, local_y in local_corners:
            # Rotate
            rotated_x = local_x * cos_theta - local_y * sin_theta
            rotated_y = local_x * sin_theta + local_y * cos_theta
            # Translate
            global_x = rotated_x + self.position_x
            global_y = rotated_y + self.position_y
            corners.append((global_x, global_y))
        
        return corners

class MapEntity:

    ENTITY_NOTHING = 0
    ENTITY_WALL = 1
    ENTITY_PARKING_SLOT = 2
    ENTITY_PARKING_GOAL = 3
    ENTITY_START = 4

    ENTITY_COLLIDABLE_FLAGS = {
        ENTITY_WALL: True,
        ENTITY_PARKING_SLOT: True,
        ENTITY_PARKING_GOAL: False,
        ENTITY_START: False,
    }

    ## valores mínimo e máximo para os tipos de entidades colidíveis
    ## isso é usado para inicializar o observation_space
    MIN_COLLIDABLE_ENTITY_TYPE = min(key for key, value in ENTITY_COLLIDABLE_FLAGS.items() if value is True)
    MAX_COLLIDABLE_ENTITY_TYPE = max(key for key, value in ENTITY_COLLIDABLE_FLAGS.items() if value is True)

    position_x: float 
    position_y: float
    width: float
    height: float
    theta: float # ângulo em relação ao eixo x do mapa
    type: int

    def __init__(self, position_x: float, position_y: float, width: float, height: float, theta: float, type: int = ENTITY_WALL):
        self.position_x = position_x
        self.position_y = position_y
        self.width = width
        self.height = height
        self.theta = theta
        self.type = type

    def get_bounding_box(self) -> BoundingBox:
        return BoundingBox(self.position_x, self.position_y, self.width, self.height, self.theta)

    def to_value(self) -> int:
        return self.type

class ArticulatedVehicle():

    ## ============ propriedades do veículo ============

    ## propriedades do trator
    comprimento_trator: float
    distancia_eixo_traseiro_quinta_roda: float
    distancia_eixo_dianteiro_quinta_roda: float
    distancia_frente_quinta_roda: float
    largura_trator: float

    ## propriedades do trailer
    comprimento_trailer: float
    distancia_eixo_traseiro_trailer_quinta_roda: float
    distancia_frente_trailer_quinta_roda: float
    largura_trailer: float
    largura_roda: float
    comprimento_roda: float
    angulo_maximo_articulacao: float

    ## ============ propriedades físicas do veículo ============

    ## posição do trator
    position_x_trator: float # posição x do trator
    position_y_trator: float # posição y do trator
    theta_trator: float # ângulo de orientação do trator
    beta_trator: float # ângulo de articulação do trator-trailer
    raycast_results: Dict[str, RaycastResult] # raycasts do trator

    raycast_positions_and_angle_offsets = [
       # (nome,   origem,    ângulo de offset)
        ("r1", "tractor", 0.0), # 0 graus
        ("r2", "tractor", math.pi/4), # 45 graus
        ("r3", "tractor", math.pi/2), # 90 graus
        ("r4", "tractor", 3*math.pi/4), # 135 graus
        ("r5", "tractor", -3*math.pi/4), # -135 graus
        ("r6", "tractor", -math.pi/2), # -90 graus
        ("r7", "tractor", -math.pi/4), # -45 graus

        ("r8", "trailer", math.pi/4), # 45 graus
        ("r9", "trailer", math.pi/2), # 90 graus
        ("r10", "trailer", 3*math.pi/4), # 135 graus
        ("r11", "trailer", -3*math.pi/4), # -135 graus
        ("r12", "trailer", -math.pi/2), # -90 graus
        ("r13", "trailer", -math.pi/4), # -45 graus
        ("r14", "trailer", math.pi), # 180 graus
    ]


    MAX_RAYCAST_LENGTH = 20.0 # 20 metros

    def __init__(self, geometry: dict):
        # Inicializa como uma entidade neutra; dimensões/posição podem ser ajustadas posteriormente

        # Parâmetros físicos do trator
        self.comprimento_trator = float(geometry.get("comprimento_trator", 0.0))
        self.distancia_eixo_traseiro_quinta_roda = float(geometry.get("distancia_eixo_traseiro_quinta_roda", 0.0))
        self.distancia_eixo_dianteiro_quinta_roda = float(geometry.get("distancia_eixo_dianteiro_quinta_roda", 0.0))
        self.distancia_frente_quinta_roda = float(geometry.get("distancia_frente_quinta_roda", 0.0))
        self.largura_trator = float(geometry.get("largura_trator", 0.0))

        # Parâmetros físicos do trailer
        self.comprimento_trailer = float(geometry.get("comprimento_trailer", 0.0))
        self.distancia_eixo_traseiro_trailer_quinta_roda = float(geometry.get("distancia_eixo_traseiro_trailer_quinta_roda", 0.0))
        self.distancia_frente_trailer_quinta_roda = float(geometry.get("distancia_frente_trailer_quinta_roda", 0.0))
        self.largura_trailer = float(geometry.get("largura_trailer", 0.0))
        self.largura_roda = float(geometry.get("largura_roda", 0.0))
        self.comprimento_roda = float(geometry.get("comprimento_roda", 0.0))
        self.angulo_maximo_articulacao = float(geometry.get("angulo_maximo_articulacao", 0.0))

        # Define uma largura/altura aproximadas para o bounding box do veículo no mapa
        # Aqui usamos as maiores dimensões laterais/longitudinais disponíveis como aproximação simples
        approx_width = max(self.largura_trator, self.largura_trailer, 0.0)
        approx_length = self.comprimento_trator + self.comprimento_trailer
        self.width = approx_width
        self.height = approx_length

        # Variáveis da simulação (mudam a cada passo)
        self.position_x_trator = 10.0
        self.position_y_trator = 10.0
        self.velocity_trator = 0.0
        self.theta_trator = 0.0
        self.beta_trator = 0.0
        self.alpha_trator = 0.0 ## ângulo de esterçamento 
        self.raycast_results = {}
        
        self.initialize_raycasts()

    def initialize_raycasts(self):
        for raycast_name, origin_point, angle_offset in self.raycast_positions_and_angle_offsets:
            self.raycast_results[raycast_name] = RaycastResult(0.0, 0.0, 0.0, 0.0, None)

    def update_raycasts(self, entities: list[MapEntity]):

        rear_axle_position = self.get_trailer_rear_axle_position()
        trailer_position_x = rear_axle_position[0]
        trailer_position_y = rear_axle_position[1]

        for raycast_name, origin_point, angle_offset in self.raycast_positions_and_angle_offsets:
            #dispara os raycasts e armazena os resultados
            if origin_point == "tractor":
                self.raycast_results[raycast_name] = fire_raycast(
                    self.position_x_trator, 
                    self.position_y_trator, 
                    self.theta_trator + angle_offset, 
                    self.MAX_RAYCAST_LENGTH, 
                    entities)

            elif origin_point == "trailer":
                self.raycast_results[raycast_name] = fire_raycast(
                    trailer_position_x, 
                    trailer_position_y, 
                    self.get_trailer_theta() + angle_offset, 
                    self.MAX_RAYCAST_LENGTH, 
                    entities)

    def get_raycast_result(self, raycast_name: str) -> RaycastResult:
        return self.raycast_results[raycast_name]

    def get_raycast_results(self) -> Dict[str, RaycastResult]:
        return self.raycast_results

    def get_raycast_count(self) -> int:
        return len(self.raycast_positions_and_angle_offsets)

    def get_raycast_lengths_and_object_classes(self) -> list[float]:
        """
        Retorna um único array contendo, em ordem fixa e consistente:
        [r1_len, r2_len, ..., rN_len, r1_class, r2_class, ..., rN_class]

        Onde:
        - r*_len é o comprimento do raycast correspondente.
        - r*_class é a classe do objeto detectado (0: nada, 1: parede, 2: vaga, ...).

        A ordem é sempre a mesma, definida pela lista `raycast_positions_and_angle_offsets`.
        """
        raycast_lengths: list[float] = []
        raycast_object_classes: list[int] = []

        # Garante ordem consistente percorrendo a lista de configuração,
        # e não o dicionário (que poderia ter outra ordem).
        for raycast_name, _origin_point, _angle_offset in self.raycast_positions_and_angle_offsets:
            result = self.raycast_results[raycast_name]

            # Comprimento do raycast
            raycast_lengths.append(result.length)

            # Classe do objeto atingido pelo raycast
            if result.entity is None:
                obj_class = MapEntity.ENTITY_NOTHING
            else:
                # Usa o tipo da entidade (já é um inteiro compatível com o espaço de observação)
                obj_class = result.entity.type

            raycast_object_classes.append(obj_class)

        # Concatena em um único array (lista Python) conforme especificação
        return raycast_lengths + raycast_object_classes

    def get_position(self):
        return self.position_x_trator, self.position_y_trator

    def get_velocity(self):
        return self.velocity_trator

    def get_theta(self):
        return self.theta_trator

    def get_beta(self):
        return self.beta_trator

    def get_alpha(self):
        return self.alpha_trator

    def get_comprimento_trator(self) -> float:
        return self.comprimento_trator

    def get_distancia_eixo_traseiro_quinta_roda(self) -> float:
        return self.distancia_eixo_traseiro_quinta_roda

    def get_distancia_eixo_dianteiro_quinta_roda(self) -> float:
        return self.distancia_eixo_dianteiro_quinta_roda

    def get_distancia_frente_quinta_roda(self) -> float:
        return self.distancia_frente_quinta_roda

    def get_largura_trator(self) -> float:
        return self.largura_trator

    def get_rear_axle_position(self) -> tuple[float, float]:
        pass
    
    def get_trailer_rear_axle_position(self) -> tuple[float, float]:
        """
        Retorna o ponto médio entre as rodas traseiras do trailer (centro do eixo traseiro).
        Calculado a partir da posição da quinta roda e da orientação do trailer.
        """
        # Posição do ponto de articulação (quinta roda) no sistema global
        joint_x = self.position_x_trator + self.distancia_eixo_traseiro_quinta_roda * math.cos(self.theta_trator)
        joint_y = self.position_y_trator + self.distancia_eixo_traseiro_quinta_roda * math.sin(self.theta_trator)

        # Orientação do trailer
        theta_trailer = self.theta_trator - self.beta_trator

        # Centro do eixo traseiro do trailer fica a uma distância
        # 'distancia_eixo_traseiro_trailer_quinta_roda' atrás da quinta roda ao longo do trailer
        rear_axle_x = joint_x - self.distancia_eixo_traseiro_trailer_quinta_roda * math.cos(theta_trailer)
        rear_axle_y = joint_y - self.distancia_eixo_traseiro_trailer_quinta_roda * math.sin(theta_trailer)

        return rear_axle_x, rear_axle_y

    def get_trailer_theta(self) -> float:
        """
        Retorna o ângulo global do trailer (θ_trailer) em relação ao eixo x.
        Pela definição de β (ângulo entre trator e trailer): β = θ_trator - θ_trailer
        Logo: θ_trailer = θ_trator - β
        """
        return self.theta_trator - self.beta_trator

    def get_comprimento_trailer(self) -> float:
        return self.comprimento_trailer

    def get_trailer_position(self) -> tuple[float, float]:
        """Retorna a posição absoluta do ponto central do trailer"""
        # Posição do ponto de articulação (quinta roda) no global
        joint_x = self.position_x_trator + self.distancia_eixo_traseiro_quinta_roda * math.cos(self.theta_trator)
        joint_y = self.position_y_trator + self.distancia_eixo_traseiro_quinta_roda * math.sin(self.theta_trator)
        
        # Orientação do trailer
        theta_trailer = self.theta_trator - self.beta_trator
        
        # Deslocamento do centro do trailer a partir da quinta roda
        # O centro fica deslocado por (comprimento/2 - distancia_frente_trailer_quinta_roda)
        # ao longo do eixo do trailer em direção à traseira
        offset_from_joint = (self.comprimento_trailer / 2.0) - self.distancia_frente_trailer_quinta_roda
        
        # Centro do trailer (deslocado do joint em direção à traseira)
        center_x = joint_x - offset_from_joint * math.cos(theta_trailer)
        center_y = joint_y - offset_from_joint * math.sin(theta_trailer)
        
        return center_x, center_y

    def get_distancia_eixo_traseiro_trailer_quinta_roda(self) -> float:
        return self.distancia_eixo_traseiro_trailer_quinta_roda

    def get_distancia_frente_trailer_quinta_roda(self) -> float:
        return self.distancia_frente_trailer_quinta_roda

    def get_largura_trailer(self) -> float:
        return self.largura_trailer

    def get_largura_roda(self) -> float:
        return self.largura_roda

    def get_comprimento_roda(self) -> float:
        return self.comprimento_roda

    def get_angulo_maximo_articulacao(self) -> float:
        return self.angulo_maximo_articulacao

    def update_physical_properties(self, 
                                    position_x: float, 
                                    position_y: float, 
                                    velocity: float,
                                    theta: float,
                                    beta: float,
                                    alpha: float):
        self.position_x_trator = position_x
        self.position_y_trator = position_y
        self.velocity_trator = velocity
        self.set_theta(theta)
        self.beta_trator = beta
        self.alpha_trator = alpha

    def set_theta(self, theta: float):
        """atualiza o ângulo de orientação do trator fazendo clipping para o intervalo [-pi, pi]"""
        new_theta = 0.0
        if theta > math.pi:
            new_theta = theta - 2 * math.pi
        elif theta < -math.pi:
            new_theta = theta + 2 * math.pi
        else:
            new_theta = theta
        self.theta_trator = new_theta

    def get_bounding_box_tractor(self) -> BoundingBox:
        """
        Retorna a BoundingBox do trator. A referência (x1, y1) é o eixo traseiro do trator;
        assumimos comprimento do trator medido de traseira (próximo ao eixo traseiro) até a frente.
        O centro geométrico do retângulo está a meio comprimento à frente do eixo traseiro.
        """
        cx = self.position_x_trator + (self.comprimento_trator / 2.0) * math.cos(self.theta_trator)
        cy = self.position_y_trator + (self.comprimento_trator / 2.0) * math.sin(self.theta_trator)
        # width = comprimento (longitudinal, local x), height = largura (lateral, local y)
        return BoundingBox(cx, cy, self.comprimento_trator, self.largura_trator, self.theta_trator)


    def get_bounding_box_trailer(self) -> BoundingBox:
        """
        Retorna a BoundingBox do trailer. A posição do trailer é derivada do trator:
        - Ponto de articulação (quinta roda) fica a 'distancia_eixo_traseiro_quinta_roda' à frente do eixo traseiro do trator.
        - Orientação do trailer θ2 = θ1 - β (β = θ1 - θ2).
        - Centro do trailer fica deslocado do pino de engate por (comprimento/2 - distancia_frente_trailer_quinta_roda)
        ao longo do eixo do trailer em direção à traseira.
        """
        trailer_center_x, trailer_center_y = self.get_trailer_position()

        # width = comprimento (longitudinal, local x), height = largura (lateral, local y)
        return BoundingBox(trailer_center_x, trailer_center_y, self.comprimento_trailer, self.largura_trailer, self.get_trailer_theta())


    def get_perpendicular_to_theta(self) -> float:
        """
        Retorna o ângulo perpendicular ao ângulo de orientação do veículo.
        """
        return self.theta_trator + math.pi / 2
    
    def get_front_axle_position(self) -> tuple[float, float]:
        front_axle_center_x = self.position_x_trator + self.distancia_eixo_dianteiro_quinta_roda * math.cos(self.theta_trator)
        front_axle_center_y = self.position_y_trator + self.distancia_eixo_dianteiro_quinta_roda * math.sin(self.theta_trator)
        return front_axle_center_x, front_axle_center_y
    
    def get_rear_axle_position(self) -> tuple[float, float]:
        rear_axle_center_x = self.position_x_trator + self.distancia_eixo_traseiro_quinta_roda * math.cos(self.theta_trator)
        rear_axle_center_y = self.position_y_trator + self.distancia_eixo_traseiro_quinta_roda * math.sin(self.theta_trator)
        return rear_axle_center_x, rear_axle_center_y
    
    def get_wheels_bounding_boxes(self) -> list[BoundingBox]:
        # Center of the front axle (from rear axle by 'distancia_eixo_dianteiro_quinta_roda')
        front_axle_center_x, front_axle_center_y = self.get_front_axle_position()

        rear_axle_center_x, rear_axle_center_y = self.get_rear_axle_position()

        # Lateral offsets (half the tractor width) perpendicular to the heading
        half_width = self.largura_trator / 2.0
        perpendicular_angle_tractor = self.theta_trator + math.pi / 2.0
        perpendicular_angle_trailer = self.get_trailer_theta() + math.pi / 2.0

        front_left_wheel_position_x = front_axle_center_x + half_width * math.cos(perpendicular_angle_tractor)
        front_left_wheel_position_y = front_axle_center_y + half_width * math.sin(perpendicular_angle_tractor)
        front_right_wheel_position_x = front_axle_center_x - half_width * math.cos(perpendicular_angle_tractor)
        front_right_wheel_position_y = front_axle_center_y - half_width * math.sin(perpendicular_angle_tractor)

        rear_left_wheel_position_x = rear_axle_center_x + half_width * math.cos(perpendicular_angle_tractor)
        rear_left_wheel_position_y = rear_axle_center_y + half_width * math.sin(perpendicular_angle_tractor)
        rear_right_wheel_position_x = rear_axle_center_x - half_width * math.cos(perpendicular_angle_tractor)
        rear_right_wheel_position_y = rear_axle_center_y - half_width * math.sin(perpendicular_angle_tractor)

        rear_axle_trailer_center_x, rear_axle_trailer_center_y = self.get_trailer_rear_axle_position()

        rear_left_trailer_wheel_position_x = rear_axle_trailer_center_x + half_width * math.cos(perpendicular_angle_trailer)
        rear_left_trailer_wheel_position_y = rear_axle_trailer_center_y + half_width * math.sin(perpendicular_angle_trailer)
        rear_right_trailer_wheel_position_x = rear_axle_trailer_center_x - half_width * math.cos(perpendicular_angle_trailer)
        rear_right_trailer_wheel_position_y = rear_axle_trailer_center_y - half_width * math.sin(perpendicular_angle_trailer)

        return [
            BoundingBox(front_left_wheel_position_x, front_left_wheel_position_y, self.comprimento_roda, self.largura_roda, self.theta_trator + self.alpha_trator),
            BoundingBox(front_right_wheel_position_x, front_right_wheel_position_y, self.comprimento_roda, self.largura_roda, self.theta_trator + self.alpha_trator),
            BoundingBox(rear_left_wheel_position_x, rear_left_wheel_position_y, self.comprimento_roda, self.largura_roda, self.theta_trator),
            BoundingBox(rear_right_wheel_position_x, rear_right_wheel_position_y, self.comprimento_roda, self.largura_roda, self.theta_trator),
            BoundingBox(rear_left_trailer_wheel_position_x, rear_left_trailer_wheel_position_y, self.comprimento_roda, self.largura_roda, self.get_trailer_theta()),
            BoundingBox(rear_right_trailer_wheel_position_x, rear_right_trailer_wheel_position_y, self.comprimento_roda, self.largura_roda, self.get_trailer_theta()),
        ]

    def check_collision(self, entity: MapEntity) -> bool:
        """Verifica se o veículo colidiu com uma entidade."""
        target_bbox = entity.get_bounding_box()
        return self.get_bounding_box_tractor().check_collision(target_bbox) or self.get_bounding_box_trailer().check_collision(target_bbox)
 

class Map:

    entities: list[MapEntity]
    parking_goal: MapEntity
    start_position: MapEntity
    size_x: float
    size_y: float
    spawn_margin: float

    def __init__(self, size: tuple[float, float], entities: list[MapEntity] = None):
        self.size_x = size[0]
        self.size_y = size[1]
        self.entities = []
        self.parking_goal = None
        self.spawn_margin = min(10.0, self.size_x/2, self.size_y/2)
        if entities is not None:
            for entity in entities:
                self.add_entity(entity)

    def get_parking_goal(self) -> MapEntity:
        return self.parking_goal

    def get_parking_goal_position(self) -> tuple[float, float]:
        return self.parking_goal.position_x, self.parking_goal.position_y

    def get_parking_goal_theta(self) -> float:
        return self.parking_goal.theta

    def add_entity(self, entity: MapEntity):
        if entity.type == MapEntity.ENTITY_PARKING_GOAL:
            self.parking_goal = entity
            self.entities.append(entity)
        else:
            self.entities.append(entity)

    def get_entities(self) -> list[MapEntity]:
        return self.entities

    def get_parking_slots(self) -> list[MapEntity]:
        return [entity for entity in self.entities if entity.type == MapEntity.ENTITY_PARKING_SLOT]

    def get_random_parking_slot(self) -> MapEntity:
        return random.choice(self.get_parking_slots())

    def get_size(self) -> tuple[float, float]:
        return (self.size_x, self.size_y)


    def get_start_position(self) -> MapEntity:
        if self.start_position is None:
            raise Exception("Must add a start position to the map before getting it")
        return self.start_position

    def place_vehicle(self, vehicle: ArticulatedVehicle):
        if self.start_position is None:
            raise Exception("Must add a start position to the map before placing a vehicle")
        start_pos = self.get_start_position()
        start_theta = start_pos.theta + math.pi / 2
        vehicle.update_physical_properties(start_pos.position_x, start_pos.position_y, 0.0, start_theta, 0.0, 0.0)
        vehicle.initialize_raycasts()
        vehicle.update_raycasts(self.entities)

    def check_collision_with_entities(self, entity: MapEntity) -> bool:
        for other_entity in self.entities:
            if entity.get_bounding_box().check_collision(other_entity.get_bounding_box()):
                return True
        return False




def fire_raycast(origin_x: float, origin_y: float, theta: float, range: float, entities: list[MapEntity]) -> tuple[float, int]:

    min_collision_distance = range
    min_collision_entity = None
    for entity in entities:
        if entity.type == MapEntity.ENTITY_WALL or entity.type == MapEntity.ENTITY_PARKING_SLOT:
            collision_distance = check_raycast_collision(origin_x, origin_y, theta, range, entity.get_bounding_box())
            if collision_distance is not None:
                if collision_distance < min_collision_distance:
                    min_collision_distance = collision_distance
                    min_collision_entity = entity

    return RaycastResult(origin_x, origin_y, theta, min_collision_distance, min_collision_entity)

def check_raycast_collision(origin_x: float, origin_y: float, theta: float, ray_length: float, other: 'BoundingBox') -> float | None:
    """ Verifica se há colisão entre esta raycast e uma bounding box
    Args:
        origin_x: Posição x da origem do raycast
        origin_y: Posição y da origem do raycast
        theta: Ângulo do raycast
        ray_length: Comprimento máximo do raycast
        other: BoundingBox a ser comparada
    Returns:
        float | None: Distância da origem até o ponto de colisão, ou None se não houver colisão
    """
    # Transform ray origin to box's local coordinate system
    # First translate to box center, then rotate by -theta
    dx = origin_x - other.position_x
    dy = origin_y - other.position_y
    
    cos_theta = math.cos(-other.theta)
    sin_theta = math.sin(-other.theta)
    
    # Rotate ray origin to box's local space
    local_origin_x = dx * cos_theta - dy * sin_theta
    local_origin_y = dx * sin_theta + dy * cos_theta
    
    # Transform ray direction to box's local space
    ray_dir_x = math.cos(theta)
    ray_dir_y = math.sin(theta)
    
    local_dir_x = ray_dir_x * cos_theta - ray_dir_y * sin_theta
    local_dir_y = ray_dir_x * sin_theta + ray_dir_y * cos_theta
    
    # Box bounds in local space (axis-aligned)
    half_w = other.width / 2.0
    half_h = other.height / 2.0
    
    box_min_x = -half_w
    box_max_x = half_w
    box_min_y = -half_h
    box_max_y = half_h
    
    # Ray-AABB intersection using slab method
    # Calculate intersection distances for each axis
    if abs(local_dir_x) < 1e-9:  # Ray parallel to y-axis
        if local_origin_x < box_min_x or local_origin_x > box_max_x:
            return None
        # Ray is inside box bounds on x-axis, check y-axis
        if abs(local_dir_y) < 1e-9:
            # Ray is a point, treat as no valid direction
            return None
        
        t1 = (box_min_y - local_origin_y) / local_dir_y if local_dir_y != 0 else float('inf')
        t2 = (box_max_y - local_origin_y) / local_dir_y if local_dir_y != 0 else float('inf')
        t_min = min(t1, t2)
        t_max = max(t1, t2)
    elif abs(local_dir_y) < 1e-9:  # Ray parallel to x-axis
        if local_origin_y < box_min_y or local_origin_y > box_max_y:
            return None
        # Ray is inside box bounds on y-axis, check x-axis
        t1 = (box_min_x - local_origin_x) / local_dir_x if local_dir_x != 0 else float('inf')
        t2 = (box_max_x - local_origin_x) / local_dir_x if local_dir_x != 0 else float('inf')
        t_min = min(t1, t2)
        t_max = max(t1, t2)
    else:
        # Calculate intersection distances for x-axis
        t_x1 = (box_min_x - local_origin_x) / local_dir_x
        t_x2 = (box_max_x - local_origin_x) / local_dir_x
        t_x_min = min(t_x1, t_x2)
        t_x_max = max(t_x1, t_x2)
        
        # Calculate intersection distances for y-axis
        t_y1 = (box_min_y - local_origin_y) / local_dir_y
        t_y2 = (box_max_y - local_origin_y) / local_dir_y
        t_y_min = min(t_y1, t_y2)
        t_y_max = max(t_y1, t_y2)
        
        # Find the intersection range
        t_min = max(t_x_min, t_y_min)
        t_max = min(t_x_max, t_y_max)
    
    # Check if there's no intersection
    if t_min > t_max or t_max < 0:
        return None

    # If outside, first hit forward is t_min (>= 0).
    # If inside (t_min < 0 <= t_max), first forward hit is the exit t_max.
    t_intersection = t_min if t_min >= 0 else t_max

    # Check if intersection is within ray length and forward
    if t_intersection < 0 or t_intersection > ray_length:
        return None

    # Return the distance from origin to intersection point
    return t_intersection












