from Simulation import Map, MapEntity
import json
from typing import Any
import math


class MapConfigLoader:

    def __init__(self, json_path: str):
        self.json_path = json_path

    def load_map(self, map_name: str = "MAPA_1", width: float = 60.0, height: float = 60.0) -> Map:
        if( map_name == "MAPA_1"):
            return self.create_default_map(width, height)
        else:
            raise ValueError(f"Mapa '{map_name}' não encontrado")

    def create_default_map(self, width: float, height: float) -> Map:
        width_wall = 2.0
        width_parking_slot = 7.0
        height_parking_slot = 12.0
        map = Map((width, height))
        # parede esquerda
        map.add_entity(MapEntity(position_x=0, position_y=height/2, width=width_wall, height=height, theta=0, type=MapEntity.ENTITY_WALL))
        # parede direita
        map.add_entity(MapEntity(position_x=width, position_y=height/2, width=width_wall, height=height, theta=0, type=MapEntity.ENTITY_WALL))
        # parede superior
        map.add_entity(MapEntity(position_x=width/2, position_y=0, width=width, height=width_wall, theta=0, type=MapEntity.ENTITY_WALL))
        # parede inferior
        map.add_entity(MapEntity(position_x=width/2, position_y=height, width=width, height=width_wall, theta=0, type=MapEntity.ENTITY_WALL))

        # vagas de estacionamento
        #for i in range(4):
        #  map.add_entity(MapEntity(position_x=width/2 + i * 1.5 * width_parking_slot, position_y=height/2, width=width_parking_slot, height=height_parking_slot, theta=math.radians(45), type=MapEntity.ENTITY_PARKING_SLOT))

        # vaga de estacionamento
        map.add_goal_randomly()
        return map