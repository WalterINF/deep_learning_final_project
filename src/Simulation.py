import math
import random
from typing import Dict
from math import tan, cos, sin
import numpy as np

class Simulation:
    vehicle: "Vehicle"
    map: "Map"

    def __init__(self, vehicle: "Vehicle", map: "Map"):
        self.vehicle = vehicle
        self.map = map
        self.map.place_vehicle(self.vehicle)

    def move_vehicle(self, velocity: float, alpha: float, dt: float):
        # Current state
        x, y = self.vehicle.get_position()
        theta = self.vehicle.get_theta()

        # Geometry - wheelbase (distance between front and rear axles)
        L = self.vehicle.get_wheelbase()

        # Kinematics (bicycle model)
        x_dot = velocity * cos(theta)
        y_dot = velocity * sin(theta)
        theta_dot = (velocity / L) * tan(alpha)

        # Euler step
        new_x = x + x_dot * dt
        new_y = y + y_dot * dt
        new_theta = theta + theta_dot * dt

        self.vehicle.update_physical_properties(new_x, new_y, velocity, new_theta, alpha)
        self.vehicle.update_raycasts(self.map.get_entities())


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
        """Calcula os quatro cantos da bounding box em coordenadas globais,
        na ordem: bottom-left, bottom-right, top-right, top-left"""
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

    ## define quais entidades são colidíveis
    ENTITY_COLLIDABLE_FLAGS = {
        ENTITY_WALL: True,
        ENTITY_PARKING_SLOT: False,
        ENTITY_PARKING_GOAL: False,
        ENTITY_START: False,
    }

    position_x: float 
    position_y: float
    width: float
    length: float
    theta: float # ângulo em relação ao eixo x do mapa
    type: int

    def __init__(self, position_x: float, position_y: float, width: float, length: float, theta: float, type: int = ENTITY_WALL):
        self.position_x = position_x
        self.position_y = position_y
        self.width = width
        self.length = length
        self.theta = theta
        self.type = type

    def get_bounding_box(self) -> BoundingBox:
        # BoundingBox expects: (dimension_along_theta, dimension_perpendicular_to_theta)
        # MapEntity: length is along theta, width is perpendicular to theta
        return BoundingBox(self.position_x, self.position_y, self.length, self.width, self.theta)

    def is_collidable(self) -> bool:
        return self.ENTITY_COLLIDABLE_FLAGS[self.type]

    def to_value(self) -> int:
        return self.type

class Vehicle():

    ## ============ propriedades do veículo ============

    comprimento: float  # vehicle length
    wheelbase: float    # distância entre-eixos
    largura: float      # vehicle width

    ## ============ estado do veículo ============

    position_x: float  # posição x do veículo
    position_y: float  # posição y do veículo
    theta: float       # ângulo de orientação do veículo
    raycast_results: Dict[str, RaycastResult]

    ## lista de raycasts do veículo
    raycast_positions_and_angle_offsets = [
       # (nome,   origem,    ângulo de offset)
        ("r1", "vehicle", 0.0), # 0 graus
        ("r2", "vehicle", math.pi/4), # 45 graus
        ("r3", "vehicle", math.pi/2), # 90 graus
        ("r4", "vehicle", 3*math.pi/4), # 135 graus
        ("r5", "vehicle", -3*math.pi/4), # -135 graus
        ("r6", "vehicle", -math.pi/2), # -90 graus
        ("r7", "vehicle", -math.pi/4), # -45 graus
    ]


    MAX_RAYCAST_LENGTH = 50.0 # 20 metros

    def __init__(self, geometry: dict):
        # Inicializa como uma entidade neutra; dimensões/posição podem ser ajustadas posteriormente

        # Parâmetros físicos do veículo
        self.comprimento = float(geometry.get("comprimento", geometry.get("comprimento_trator", 0.0)))
        self.largura = float(geometry.get("largura", geometry.get("largura_trator", 0.0)))
        self.wheelbase = float(geometry.get("wheelbase", geometry.get("distancia_eixo_dianteiro_quinta_roda", 0.0)))
        self.largura_roda = float(geometry.get("largura_roda", 0.0))
        self.comprimento_roda = float(geometry.get("comprimento_roda", 0.0))
        self.angulo_maximo_esterco = float(geometry.get("angulo_maximo_esterco", geometry.get("angulo_maximo_articulacao", 0.0)))

        # Define largura/altura para o bounding box do veículo no mapa
        self.width = self.largura
        self.length = self.comprimento

        # Variáveis da simulação (mudam a cada passo)
        self.position_x = 10.0
        self.position_y = 10.0
        self.velocity = 0.0
        self.theta = 0.0
        self.alpha = 0.0  # ângulo de esterçamento
        self.raycast_results = {}
        
        self.initialize_raycasts()

    def initialize_raycasts(self):
        for raycast_name, origin_point, angle_offset in self.raycast_positions_and_angle_offsets:
            self.raycast_results[raycast_name] = RaycastResult(0.0, 0.0, 0.0, 0.0, None)

    def update_raycasts(self, entities: list[MapEntity]):
        center_position = self.get_center_position()
        vehicle_theta = self.get_theta()
        
        for raycast_name, origin_point, angle_offset in self.raycast_positions_and_angle_offsets:
            # dispara os raycasts e armazena os resultados
            if origin_point == "vehicle":
                self.raycast_results[raycast_name] = fire_raycast(
                    center_position[0], 
                    center_position[1], 
                    vehicle_theta + angle_offset, 
                    self.MAX_RAYCAST_LENGTH, 
                    entities)

    def get_raycast_results(self) -> Dict[str, RaycastResult]:
        return self.raycast_results

    def get_center_position(self) -> tuple[float, float]:
        return self.position_x + cos(self.theta) * self.get_comprimento()/2, self.position_y + sin(self.theta) * self.get_comprimento()/2

    def get_raycast_count(self) -> int:
        return len(self.raycast_positions_and_angle_offsets)

    def get_raycast_lengths_and_object_classes(self) -> tuple[list[float], list[int]]:
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

        return raycast_lengths, raycast_object_classes

    def get_position(self) -> tuple[float, float]:
        return self.position_x, self.position_y

    def get_velocity(self) -> float:
        return self.velocity

    def get_theta(self) -> float:
        return self.theta

    def get_alpha(self) -> float:
        return self.alpha

    def get_comprimento(self) -> float:
        return self.comprimento

    def get_largura(self) -> float:
        return self.largura

    def get_state(self) -> np.ndarray:
        """retorna o estado do veículo como um array numpy"""
        return np.array([self.position_x, self.position_y, self.theta, self.alpha])

    def get_wheelbase(self) -> float:
        return self.wheelbase

    def get_largura_roda(self) -> float:
        return self.largura_roda

    def get_comprimento_roda(self) -> float:
        return self.comprimento_roda

    def get_angulo_maximo_esterco(self) -> float:
        return self.angulo_maximo_esterco

    def update_physical_properties(self, 
                                    position_x: float, 
                                    position_y: float, 
                                    velocity: float,
                                    theta: float,
                                    alpha: float):
        self.position_x = position_x
        self.position_y = position_y
        self.velocity = velocity
        self.set_theta(theta)
        self.alpha = alpha

    def set_theta(self, new_theta: float):
        """atualiza o ângulo de orientação do veículo fazendo clipping para o intervalo [-pi, pi]"""
        if new_theta > math.pi:
            self.theta = new_theta - 2 * math.pi
        elif new_theta < -math.pi:
            self.theta = new_theta + 2 * math.pi
        else:
            self.theta = new_theta

    def get_bounding_box(self) -> BoundingBox:
        """
        Retorna a BoundingBox do veículo. A referência (x, y) é o eixo traseiro do veículo;
        assumimos comprimento do veículo medido de traseira (próximo ao eixo traseiro) até a frente.
        O centro geométrico do retângulo está a meio comprimento à frente do eixo traseiro.
        """
        cx = self.position_x + (self.comprimento / 2.0) * math.cos(self.theta)
        cy = self.position_y + (self.comprimento / 2.0) * math.sin(self.theta)
        # width = comprimento (longitudinal, local x), height = largura (lateral, local y)
        return BoundingBox(cx, cy, self.comprimento, self.largura, self.theta)



    def get_perpendicular_to_theta(self) -> float:
        """
        Retorna o ângulo perpendicular ao ângulo de orientação do veículo.
        """
        return self.theta + math.pi / 2
    
    def get_front_axle_position(self) -> tuple[float, float]:
        """Retorna o ponto médio entre as rodas dianteiras do veículo (centro do eixo dianteiro)"""
        front_axle_center_x = self.position_x + self.wheelbase * math.cos(self.theta)
        front_axle_center_y = self.position_y + self.wheelbase * math.sin(self.theta)
        return front_axle_center_x, front_axle_center_y
    
    def get_rear_axle_position(self) -> tuple[float, float]:
        """Retorna o ponto médio entre as rodas traseiras do veículo (centro do eixo traseiro)
        O sistema de referência coloca a origem (position_x, position_y) no eixo traseiro."""
        return self.position_x, self.position_y
    
    def get_wheels_bounding_boxes(self) -> list[BoundingBox]:
        """Retorna as bounding boxes dos pneus do veículo"""
        front_axle_center_x, front_axle_center_y = self.get_front_axle_position()
        rear_axle_center_x, rear_axle_center_y = self.get_rear_axle_position()

        # Lateral offsets (half the vehicle width) perpendicular to the heading
        half_width = self.largura / 2.0
        perpendicular_angle = self.theta + math.pi / 2.0

        front_left_wheel_x = front_axle_center_x + half_width * math.cos(perpendicular_angle)
        front_left_wheel_y = front_axle_center_y + half_width * math.sin(perpendicular_angle)
        front_right_wheel_x = front_axle_center_x - half_width * math.cos(perpendicular_angle)
        front_right_wheel_y = front_axle_center_y - half_width * math.sin(perpendicular_angle)

        rear_left_wheel_x = rear_axle_center_x + half_width * math.cos(perpendicular_angle)
        rear_left_wheel_y = rear_axle_center_y + half_width * math.sin(perpendicular_angle)
        rear_right_wheel_x = rear_axle_center_x - half_width * math.cos(perpendicular_angle)
        rear_right_wheel_y = rear_axle_center_y - half_width * math.sin(perpendicular_angle)

        return [
            BoundingBox(front_left_wheel_x, front_left_wheel_y, self.comprimento_roda, self.largura_roda, self.theta + self.alpha),
            BoundingBox(front_right_wheel_x, front_right_wheel_y, self.comprimento_roda, self.largura_roda, self.theta + self.alpha),
            BoundingBox(rear_left_wheel_x, rear_left_wheel_y, self.comprimento_roda, self.largura_roda, self.theta),
            BoundingBox(rear_right_wheel_x, rear_right_wheel_y, self.comprimento_roda, self.largura_roda, self.theta),
        ]

    def check_collision(self, entity: MapEntity) -> bool:
        """Verifica se o veículo colidiu com uma entidade."""
        target_bbox = entity.get_bounding_box()
        return self.get_bounding_box().check_collision(target_bbox)
 

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
        """Retorna todas as entidades do mapa"""
        return self.entities

    def get_parking_slots(self) -> list[MapEntity]:
        """Retorna todas as vagas de estacionamento do mapa"""
        return [entity for entity in self.entities if entity.type == MapEntity.ENTITY_PARKING_SLOT]

    def get_random_parking_slot(self) -> MapEntity:
        """Retorna uma vaga de estacionamento aleatória"""
        return random.choice(self.get_parking_slots())

    def get_size(self) -> tuple[float, float]:
        """Retorna o tamanho do mapa"""
        return (self.size_x, self.size_y)

    def get_start_position(self) -> MapEntity:
        """Retorna a posição de partida do mapa"""
        if self.start_position is None:
            raise Exception("Deve adicionar uma posição de partida ao mapa antes de obtê-la")
        return self.start_position

    def place_vehicle(self, vehicle: "Vehicle"):
        """Coloca o veículo no mapa na posição de partida"""
        if self.start_position is None:
            raise Exception("Deve adicionar uma posição de partida ao mapa antes de colocar um veículo")
        start_pos = self.get_start_position()
        start_theta = start_pos.theta
        start_x = start_pos.position_x + 2.0 * math.cos(start_theta)
        start_y = start_pos.position_y + 2.0 * math.sin(start_theta)
        vehicle.update_physical_properties(start_x, start_y, 0.0, start_theta, 0.0)
        vehicle.initialize_raycasts()
        vehicle.update_raycasts(self.entities)

    def check_collision_with_entities(self, entity: MapEntity) -> bool:
        for other_entity in self.entities:
            if entity.get_bounding_box().check_collision(other_entity.get_bounding_box()):
                return True
        return False




def fire_raycast(origin_x: float, origin_y: float, theta: float, range: float, entities: list[MapEntity]) -> 'RaycastResult':

    min_collision_distance = range
    min_collision_entity = None
    for entity in entities:
        if entity.is_collidable():
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


# Backwards compatibility alias
ArticulatedVehicle = Vehicle




