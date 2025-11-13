from collisions import BoundingBox
import math
from collisions import Raycast
from casadi import cos, sin


class MapEntity:

    ENTITY_WALL = "wall"
    ENTITY_OBSTACLE = "obstacle"
    ENTITY_PARKING_SLOT = "parking_slot"
    ENTITY_PARKING_GOAL = "parking_goal"

    position_x: float 
    position_y: float
    width: float
    height: float
    theta: float # ângulo em relação ao eixo x do mapa
    type: str

    def __init__(self, position_x: float, position_y: float, width: float, height: float, theta: float, type: str = ENTITY_WALL):
        self.position_x = position_x
        self.position_y = position_y
        self.width = width
        self.height = height
        self.theta = theta
        self.type = type

    def get_bounding_box(self) -> BoundingBox:
        return BoundingBox(self.position_x, self.position_y, self.width, self.height, self.theta)

class ArticulatedVehicle():

    ## propriedades do trator
    comprimento_trator: float
    distancia_eixo_traseiro_quinta_roda: float
    distancia_eixo_dianteiro_quinta_roda: float
    distancia_frente_quinta_roda: float
    largura_trator: float

    ## posição do trator
    position_x_trator: float
    position_y_trator: float
    theta_trator: float
    beta_trator: float

    ## raycasts
    raycasts: dict[str, Raycast]

    # Trailer

    ## propriedades do trailer
    comprimento_trailer: float
    distancia_eixo_traseiro_trailer_quinta_roda: float
    distancia_frente_trailer_quinta_roda: float
    largura_trailer: float
    largura_roda: float
    comprimento_roda: float
    angulo_maximo_articulacao: float

    ## posição do trailer é calculada a partir da posição do trator e do ângulo de articulação


    raycast_positions_and_angle_offsets = {
       #nome  origem     ângulo de offset
        "r1": ("tractor", 0.0), # 0 graus
        "r2": ("tractor", math.pi/4), # 45 graus
        "r3": ("tractor", math.pi/2), # 90 graus
        "r4": ("tractor", 3*math.pi/4), # 135 graus
        "r5": ("tractor", -3*math.pi/4), # -135 graus
        "r6": ("tractor", -math.pi/2), # -90 graus
        "r7": ("tractor", -math.pi/4), # -45 graus

        "r8": ("trailer", math.pi/4), # 45 graus
        "r9": ("trailer", math.pi/2), # 90 graus
        "r10": ("trailer", 3*math.pi/4), # 135 graus
        "r11": ("trailer", -3*math.pi/4), # -135 graus
        "r12": ("trailer", -math.pi/2), # -90 graus
        "r13": ("trailer", -math.pi/4), # -45 graus
        "r14": ("trailer", math.pi), # 180 graus
    }

    MAX_RAYCAST_LENGTH = 150.0 # 10 metros

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
        self.raycasts = dict[str, Raycast]()
        
        self.initialize_raycasts()



    def initialize_raycasts(self):
        trailer_position_x = self.position_x_trator - self.distancia_eixo_traseiro_trailer_quinta_roda * math.cos(self.beta_trator)
        trailer_position_y = self.position_y_trator - self.distancia_eixo_traseiro_trailer_quinta_roda * math.sin(self.beta_trator)

        for raycast_name, (origin_point, angle_offset) in self.raycast_positions_and_angle_offsets.items():
            if origin_point == "tractor":
                self.raycasts[raycast_name] = Raycast(self.position_x_trator, self.position_y_trator, self.theta_trator + angle_offset, self.MAX_RAYCAST_LENGTH)
            elif origin_point == "trailer":
                self.raycasts[raycast_name] = Raycast(trailer_position_x, trailer_position_y, self.get_trailer_theta() + angle_offset, self.MAX_RAYCAST_LENGTH)

    def update_raycasts(self, entities: list[MapEntity]):

        rear_axle_position = self.get_trailer_rear_axle_position()
        trailer_position_x = rear_axle_position[0]
        trailer_position_y = rear_axle_position[1]

        for raycast_name, (origin_point, angle_offset) in self.raycast_positions_and_angle_offsets.items():
            # Reset length each frame
            min_distance = self.MAX_RAYCAST_LENGTH
            # Ensure collision checks use full range for this frame (avoid using last frame's shortened length)
            self.raycasts[raycast_name].length = self.MAX_RAYCAST_LENGTH

            if origin_point == "tractor":
                self.raycasts[raycast_name].origin_x = self.position_x_trator
                self.raycasts[raycast_name].origin_y = self.position_y_trator
                self.raycasts[raycast_name].theta = self.theta_trator + angle_offset

                for entity in entities:
                    if entity.type == MapEntity.ENTITY_WALL:
                        collision_distance = self.raycasts[raycast_name].check_collision(entity.get_bounding_box())
                        if collision_distance is not None:
                            min_distance = min(min_distance, collision_distance)

                self.raycasts[raycast_name].length = min_distance

            elif origin_point == "trailer":
                self.raycasts[raycast_name].origin_x = trailer_position_x
                self.raycasts[raycast_name].origin_y = trailer_position_y
                self.raycasts[raycast_name].theta = self.get_trailer_theta() + angle_offset 

                for entity in entities:
                    if entity.type == MapEntity.ENTITY_WALL:
                        collision_distance = self.raycasts[raycast_name].check_collision(entity.get_bounding_box())
                        if collision_distance is not None:
                            min_distance = min(min_distance, collision_distance)

                self.raycasts[raycast_name].length = min_distance

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

    # Getters for tractor properties
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

    # Getters for trailer properties
    def get_comprimento_trailer(self) -> float:
        return self.comprimento_trailer

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
                                    theta: float,
                                    beta: float,
                                    alpha: float):
        self.position_x_trator = position_x
        self.position_y_trator = position_y
        self.theta_trator = theta
        self.beta_trator = beta
        self.alpha_trator = alpha

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
        # Posição do ponto de articulação (quinta roda) no global
        joint_x = self.position_x_trator + self.distancia_eixo_traseiro_quinta_roda * math.cos(self.theta_trator)
        joint_y = self.position_y_trator + self.distancia_eixo_traseiro_quinta_roda * math.sin(self.theta_trator)

        # Orientação do trailer
        theta_trailer = self.theta_trator - self.beta_trator

        # Deslocamento do centro do trailer a partir da quinta roda
        center_offset = (self.comprimento_trailer / 2.0) - self.distancia_frente_trailer_quinta_roda
        cx = joint_x - center_offset * math.cos(theta_trailer)
        cy = joint_y - center_offset * math.sin(theta_trailer)

        # width = comprimento (longitudinal, local x), height = largura (lateral, local y)
        return BoundingBox(cx, cy, self.comprimento_trailer, self.largura_trailer, theta_trailer)

    def get_perpendicular_to_theta(self) -> float:
        """
        Retorna o ângulo perpendicular ao ângulo de orientação do veículo.
        """
        return self.theta_trator + math.pi / 2

    def get_wheels_bounding_boxes(self) -> list[BoundingBox]:
        # Center of the front axle (from rear axle by 'distancia_eixo_dianteiro_quinta_roda')
        axle_center_x = self.position_x_trator + self.distancia_eixo_dianteiro_quinta_roda * math.cos(self.theta_trator)
        axle_center_y = self.position_y_trator + self.distancia_eixo_dianteiro_quinta_roda * math.sin(self.theta_trator)

        # Lateral offsets (half the tractor width) perpendicular to the heading
        half_width = self.largura_trator / 2.0
        perpendicular_angle = self.theta_trator + math.pi / 2.0

        front_left_wheel_position_x = axle_center_x + half_width * math.cos(perpendicular_angle)
        front_left_wheel_position_y = axle_center_y + half_width * math.sin(perpendicular_angle)
        front_right_wheel_position_x = axle_center_x - half_width * math.cos(perpendicular_angle)
        front_right_wheel_position_y = axle_center_y - half_width * math.sin(perpendicular_angle)

        return [
            BoundingBox(front_left_wheel_position_x, front_left_wheel_position_y, self.comprimento_roda, self.largura_roda, self.theta_trator + self.alpha_trator),
            BoundingBox(front_right_wheel_position_x, front_right_wheel_position_y, self.comprimento_roda, self.largura_roda, self.theta_trator + self.alpha_trator),
        ]
 

class Map:

    vehicle: ArticulatedVehicle
    entities: list[MapEntity]
    size_x: float
    size_y: float

    color_mappings = {
        MapEntity.ENTITY_WALL: (0, 0, 0),
        MapEntity.ENTITY_OBSTACLE: (100, 0, 0),
        MapEntity.ENTITY_PARKING_SLOT: (0, 100, 0),
        MapEntity.ENTITY_PARKING_GOAL: (0, 0, 200),
    }


    def __init__(self, size: tuple[float, float], vehicle: ArticulatedVehicle, entities: list[MapEntity] = None):
        self.size_x = size[0]
        self.size_y = size[1]
        self.entities = []
        self.vehicle = vehicle

    def add_entity(self, entity: MapEntity):
        self.entities.append(entity)

    def get_entities(self) -> list[MapEntity]:
        return self.entities



    def move_vehicle(self, velocity: float, alpha: float, dt: float):
        # Current state
        x, y = self.vehicle.get_position()
        theta = self.vehicle.get_theta()
        beta = self.vehicle.get_beta()

        # Geometry
        D = self.vehicle.get_distancia_eixo_dianteiro_quinta_roda() - self.vehicle.get_distancia_eixo_traseiro_quinta_roda()
        L = self.vehicle.get_distancia_eixo_traseiro_trailer_quinta_roda()
        a = self.vehicle.get_distancia_eixo_traseiro_quinta_roda()

        angular_velocity_tractor = (velocity / D) * math.tan(alpha)
        beta_dot = angular_velocity_tractor * (1 - (alpha * cos(beta)) / L) - (velocity * sin(beta)) / L

        # Kinematics
        x_dot = velocity * math.cos(theta)
        y_dot = velocity * math.sin(theta)
        theta_dot = (velocity / D) * math.tan(alpha)
        beta_dot = beta_dot = angular_velocity_tractor * (1 - (alpha * cos(beta)) / L) - (velocity * sin(beta)) / L

        # Euler step
        new_x = x + x_dot * dt
        new_y = y + y_dot * dt
        new_theta = theta + theta_dot * dt
        new_beta = beta + beta_dot * dt


        self.vehicle.update_physical_properties(new_x, new_y, new_theta, new_beta, alpha)
        self.vehicle.update_raycasts(self.entities)








