import json
from typing import Any, Dict, List, Optional
import os
from src.Simulation import ArticulatedVehicle, Map, MapEntity, Simulation
import random
import math
import numpy as np
from abc import abstractmethod

class SimulationLoader:
    def __init__(self):
        self.vehicle_loader = VehicleConfigLoader("config/lista_veiculos.json")
        self.map_generator = DefaultMapGenerator()

    def load_simulation(self) -> Simulation:
        vehicle = self.vehicle_loader.load_vehicle("BUG1")
        map = self.map_generator.generate_map()
        return Simulation(vehicle, map)




class VehicleConfigLoader:
    """
    Utilitário para carregar a lista de veículos e suas geometrias a partir de um arquivo JSON.

    Estrutura esperada do JSON:
    {
        "veiculos": [
            {
                "nome": "BUG1",
                "geometria_veiculo": { ... campos ... }
            },
            ...
        ]
    }
    """

    def __init__(self, json_path: str):
        self._json_path: str = json_path
        self._vehicles_by_name: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        json_path = self._json_path
        if not os.path.exists(json_path):
            # Fall back to path relative to this file (e.g., when running from repo root)
            candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), json_path))
            if os.path.exists(candidate):
                json_path = candidate
            else:
                raise FileNotFoundError(f"Arquivo JSON não encontrado: {self._json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vehicles: List[Dict[str, Any]] = data.get("veiculos", [])
        by_name: Dict[str, Dict[str, Any]] = {}
        for v in vehicles:
            name = v.get("nome")
            if not name:
                # ignora entradas sem nome
                continue
            geometry = v.get("geometria_veiculo", {})
            if not isinstance(geometry, dict):
                geometry = {}
            by_name[name] = geometry
        self._vehicles_by_name = by_name

    def available_vehicle_names(self) -> List[str]:
        return list(self._vehicles_by_name.keys())

    def get_geometry(self, name: str) -> Dict[str, Any]:
        """Public method to get vehicle geometry dictionary."""
        return self._get_geometry(name)

    def _get_geometry(self, name: str) -> Dict[str, Any]:
        if name not in self._vehicles_by_name:
            raise KeyError(f"Veículo '{name}' não encontrado no arquivo {self._json_path}")
        return self._vehicles_by_name[name]

    def load_vehicle(self, name: str) -> ArticulatedVehicle:
        geometry = self._get_geometry(name)
        return ArticulatedVehicle(geometry)



class MapGenerator:

    @abstractmethod
    def generate_map(self) -> Map:
        pass

class DefaultMapGenerator(MapGenerator):

    MAP_WIDTH = 150.0
    MAP_HEIGHT = 150.0
    WALL_WIDTH = 4.0
    PARKING_SLOT_WIDTH = 5.5
    PARKING_SLOT_HEIGHT = 14.0
    SPAWN_PADDING = 25.0
    WALL_PADDING = 3.0
    N_ROWS = 2

    def generate_map(self) -> Map:
        map = Map((self.MAP_WIDTH, self.MAP_HEIGHT))
        map.add_entity(MapEntity(position_x=0, position_y=self.MAP_HEIGHT/2, width=self.WALL_WIDTH, length=self.MAP_HEIGHT, theta=math.pi/2, type=MapEntity.ENTITY_WALL))  
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH, position_y=self.MAP_HEIGHT/2, width=self.WALL_WIDTH, length=self.MAP_HEIGHT, theta=-math.pi/2, type=MapEntity.ENTITY_WALL))
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=0, width=self.WALL_WIDTH, length=self.MAP_WIDTH, theta=0.0, type=MapEntity.ENTITY_WALL))
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=self.MAP_HEIGHT, width=self.WALL_WIDTH, length=self.MAP_WIDTH, theta=0.0, type=MapEntity.ENTITY_WALL))

        parking_lots_end_x = self.MAP_WIDTH - self.SPAWN_PADDING - self.PARKING_SLOT_WIDTH/2

        top_row_height = self.MAP_HEIGHT/4
        bottom_row_height = 3*self.MAP_HEIGHT/4

        map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=top_row_height, width=self.MAP_WIDTH - 2*self.SPAWN_PADDING, length=1.0, theta=math.pi/2, type=MapEntity.ENTITY_WALL))
        parking_lots_start_x = self.SPAWN_PADDING + self.PARKING_SLOT_WIDTH/2
        #adicionar uma fileira de vagas apontando para baixo 
        while parking_lots_start_x < parking_lots_end_x:
            upper_parking_slot = MapEntity(
                position_x=parking_lots_start_x, 
                position_y=top_row_height + self.PARKING_SLOT_HEIGHT/2.0 + self.WALL_PADDING, 
                width=self.PARKING_SLOT_WIDTH, 
                length=self.PARKING_SLOT_HEIGHT, 
                theta=math.pi/2, 
                type=MapEntity.ENTITY_PARKING_SLOT)
            map.add_entity(upper_parking_slot)
            parking_lots_start_x += self.PARKING_SLOT_WIDTH

        #adicionar uma fileira de vagas apontando para cima
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=bottom_row_height, width=self.MAP_WIDTH - 2*self.SPAWN_PADDING, length=1.0, theta=math.pi/2, type=MapEntity.ENTITY_WALL))
        parking_lots_start_x = self.SPAWN_PADDING + self.PARKING_SLOT_WIDTH/2
        while parking_lots_start_x < parking_lots_end_x:
            lower_parking_slot = MapEntity(
                position_x=parking_lots_start_x, 
                position_y=bottom_row_height - self.PARKING_SLOT_HEIGHT/2.0 - self.WALL_PADDING, 
                width=self.PARKING_SLOT_WIDTH, 
                length=self.PARKING_SLOT_HEIGHT, 
                theta=-math.pi/2, 
                type=MapEntity.ENTITY_PARKING_SLOT)
            map.add_entity(lower_parking_slot)
            parking_lots_start_x += self.PARKING_SLOT_WIDTH

        ## escolhe uma das vagas para se tornar o ponto de partida e o objetivo de estacionamento
        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_PARKING_GOAL
        map.parking_goal = chosen_parking_slot

        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_START
        map.start_position = chosen_parking_slot

        # adiciona carros aleatoriamente nas vagas (exceto a vaga de destino e a vaga de partida)
        for parking_slot in map.get_parking_slots():
            if random.random() < 0.25 and parking_slot.type != MapEntity.ENTITY_PARKING_GOAL and parking_slot.type != MapEntity.ENTITY_START:
                map.add_entity(
                    MapEntity(
                        position_x=parking_slot.position_x, 
                        position_y=parking_slot.position_y, 
                        width=self.PARKING_SLOT_WIDTH - 2.5, 
                        length=self.PARKING_SLOT_HEIGHT - 6.0, 
                        theta=parking_slot.theta, 
                        type=MapEntity.ENTITY_WALL))

        return map







    




