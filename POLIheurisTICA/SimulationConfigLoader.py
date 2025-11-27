import json
from typing import Any, Dict, List, Optional
import os
from Simulation import ArticulatedVehicle, Map, MapEntity, Simulation
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
    PARKING_SLOT_WIDTH = 5.0
    PARKING_SLOT_HEIGHT = 12.0
    SPAWN_PADDING = 30.0
    WALL_PADDING = 3.0
    N_ROWS = 2

    def generate_map(self) -> Map:
        map = Map((self.MAP_WIDTH, self.MAP_HEIGHT))
        # parede esquerda
        map.add_entity(MapEntity(position_x=0, position_y=self.MAP_HEIGHT/2, width=self.WALL_WIDTH, length=self.MAP_HEIGHT, theta=math.pi/2, type=MapEntity.ENTITY_WALL))
        # parede direita
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH, position_y=self.MAP_HEIGHT/2, width=self.WALL_WIDTH, length=self.MAP_HEIGHT, theta=-math.pi/2, type=MapEntity.ENTITY_WALL))
        # parede superior
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=0, width=self.WALL_WIDTH, length=self.MAP_WIDTH, theta=0.0, type=MapEntity.ENTITY_WALL))
        # parede inferior
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=self.MAP_HEIGHT, width=self.WALL_WIDTH, length=self.MAP_WIDTH, theta=0.0, type=MapEntity.ENTITY_WALL))

        parking_lots_rows_heights = [i * self.MAP_HEIGHT/(self.N_ROWS + 1) for i in range(1, self.N_ROWS + 1)]

        parking_lots_end_x = self.MAP_WIDTH - self.SPAWN_PADDING - self.PARKING_SLOT_WIDTH/2
        for row_height in parking_lots_rows_heights:
            #adicionar parede horizontal entre as fileira de vagas
            map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=row_height, width=self.MAP_WIDTH - 2*self.SPAWN_PADDING, length=1.0, theta=math.pi/2, type=MapEntity.ENTITY_WALL))
            parking_lots_start_x = self.SPAWN_PADDING + self.PARKING_SLOT_WIDTH/2
            #adicionar uma fileira de vagas apontando para baixo e outra apontando para cima
            while parking_lots_start_x < parking_lots_end_x:
                upper_parking_slot = MapEntity(
                    position_x=parking_lots_start_x, 
                    position_y=row_height - self.PARKING_SLOT_HEIGHT/2.0 - self.WALL_PADDING, 
                    width=self.PARKING_SLOT_WIDTH, 
                    length=self.PARKING_SLOT_HEIGHT, 
                    theta=-math.pi/2, 
                    type=MapEntity.ENTITY_PARKING_SLOT)
                map.add_entity(upper_parking_slot)
                lower_parking_slot = MapEntity(
                    position_x=parking_lots_start_x, 
                    position_y=row_height + self.PARKING_SLOT_HEIGHT/2.0 + self.WALL_PADDING, 
                    width=self.PARKING_SLOT_WIDTH, 
                    length=self.PARKING_SLOT_HEIGHT, 
                    theta=math.pi/2, 
                    type=MapEntity.ENTITY_PARKING_SLOT)
                map.add_entity(lower_parking_slot)
                parking_lots_start_x += self.PARKING_SLOT_WIDTH



        ## escolhe uma das vagas para se tornar o ponoito de partida e o objetivo de estacionamento
        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_PARKING_GOAL
        map.parking_goal = chosen_parking_slot

        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_START
        map.start_position = chosen_parking_slot

        return map

class ComplexMapGenerator(MapGenerator):

    MAP_WIDTH = 180.0
    MAP_HEIGHT = 180.0
    WALL_WIDTH = 4.0
    PARKING_SLOT_WIDTH = 4.5
    PARKING_SLOT_HEIGHT = 12.0
    SPAWN_PADDING = 20.0
    PARKING_LOTS_ROW_SPACING = 25.0
    CHUNK_DIVISIONS = (3,3)
    CHUNK_SIZE = (MAP_WIDTH/CHUNK_DIVISIONS[0], MAP_HEIGHT/CHUNK_DIVISIONS[1])
    CHUNK_WALL_PADDING = 30.0
    CHUNK_PARKING_LOT_PADDING = 15.0
    
    CHUNK_EMPTY = 0
    CHUNK_BUILDING = 1
    CHUNK_PARKING_LOT_SINGLE = 2
    CHUNK_PARKING_LOT_DOUBLE = 3

    def generate_map(self) -> Map:
        map = Map((self.MAP_WIDTH, self.MAP_HEIGHT))
        
        # parede esquerda
        map.add_entity(MapEntity(position_x=0, position_y=self.MAP_HEIGHT/2, width=self.WALL_WIDTH, length=self.MAP_HEIGHT, theta=math.pi/2, type=MapEntity.ENTITY_WALL))
        # parede direita
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH, position_y=self.MAP_HEIGHT/2, width=self.WALL_WIDTH, length=self.MAP_HEIGHT, theta=-math.pi/2, type=MapEntity.ENTITY_WALL))
        # parede superior
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=0, width=self.WALL_WIDTH, length=self.MAP_WIDTH, theta=0.0, type=MapEntity.ENTITY_WALL))
        # parede inferior
        map.add_entity(MapEntity(position_x=self.MAP_WIDTH/2, position_y=self.MAP_HEIGHT, width=self.WALL_WIDTH, length=self.MAP_WIDTH, theta=0.0, type=MapEntity.ENTITY_WALL))


        map_matrix = np.zeros(self.CHUNK_DIVISIONS, dtype=int)

        # define o chunk [1,1] como estacionamento obrigatório
        map_matrix[1,1] = random.choice([self.CHUNK_PARKING_LOT_SINGLE, self.CHUNK_PARKING_LOT_DOUBLE])


        # não sobrescrever o [1,1] no sorteio
        for i in range(self.CHUNK_DIVISIONS[0]):
            for j in range(self.CHUNK_DIVISIONS[1]):
                if (i, j) == (1, 1):
                    continue
                map_matrix[i,j] = random.choice([self.CHUNK_EMPTY,self.CHUNK_BUILDING, self.CHUNK_PARKING_LOT_SINGLE, self.CHUNK_PARKING_LOT_DOUBLE])

        for i in range(self.CHUNK_DIVISIONS[0]):
            for j in range(self.CHUNK_DIVISIONS[1]):
                chunk_center_x = i*self.CHUNK_SIZE[0] + self.CHUNK_SIZE[0]/2
                chunk_center_y = j*self.CHUNK_SIZE[1] + self.CHUNK_SIZE[1]/2
                chunk_start_x = i*self.CHUNK_SIZE[0]
                chunk_start_y = j*self.CHUNK_SIZE[1]
                chunk_end_x = chunk_start_x + self.CHUNK_SIZE[0]
                chunk_end_y = chunk_start_y + self.CHUNK_SIZE[1]
                if map_matrix[i,j] == self.CHUNK_BUILDING:
                    #preenche o chunk com um edifício
                    map.add_entity(MapEntity(position_x=chunk_center_x, position_y=chunk_center_y, width=self.CHUNK_SIZE[0] - self.CHUNK_WALL_PADDING, length=self.CHUNK_SIZE[1] - self.CHUNK_WALL_PADDING, theta=0.0, type=MapEntity.ENTITY_WALL))
                elif map_matrix[i,j] == self.CHUNK_PARKING_LOT_SINGLE:
                    #preenche com uma fileira de vagas de estacionamento
                    vertical_or_horizontal = random.choice([True, False])
                    if vertical_or_horizontal:
                        parking_lot_row_start_y = chunk_start_y + self.CHUNK_PARKING_LOT_PADDING
                        parking_lot_row_end_y = chunk_end_y - self.CHUNK_PARKING_LOT_PADDING
                        while parking_lot_row_start_y < parking_lot_row_end_y:
                            map.add_entity(MapEntity(position_x=chunk_center_x, position_y=parking_lot_row_start_y, width=self.PARKING_SLOT_WIDTH, length=self.PARKING_SLOT_HEIGHT, theta=random.choice([0.0, math.pi]), type=MapEntity.ENTITY_PARKING_SLOT))
                            parking_lot_row_start_y += self.PARKING_SLOT_WIDTH
                    else:
                        parking_lot_row_start_x = chunk_start_x + self.CHUNK_PARKING_LOT_PADDING
                        parking_lot_row_end_x = chunk_end_x - self.CHUNK_PARKING_LOT_PADDING
                        while parking_lot_row_start_x < parking_lot_row_end_x:
                            map.add_entity(MapEntity(position_x=parking_lot_row_start_x, position_y=chunk_center_y, width=self.PARKING_SLOT_WIDTH, length=self.PARKING_SLOT_HEIGHT, theta=random.choice([-math.pi/2, math.pi/2]), type=MapEntity.ENTITY_PARKING_SLOT))
                            parking_lot_row_start_x += self.PARKING_SLOT_WIDTH
                elif map_matrix[i,j] == self.CHUNK_PARKING_LOT_DOUBLE:
                    #preenche com duas fileiras de vagas de estacionamento
                    vertical_or_horizontal = random.choice([True, False])
                    if vertical_or_horizontal:
                        parking_lot_row_start_y = chunk_start_y + self.CHUNK_PARKING_LOT_PADDING
                        parking_lot_row_end_y = chunk_end_y - self.CHUNK_PARKING_LOT_PADDING
                        while parking_lot_row_start_y < parking_lot_row_end_y:
                            map.add_entity(MapEntity(position_x=chunk_start_x + self.PARKING_SLOT_HEIGHT, position_y=parking_lot_row_start_y, width=self.PARKING_SLOT_WIDTH, length=self.PARKING_SLOT_HEIGHT, theta=0.0, type=MapEntity.ENTITY_PARKING_SLOT))
                            map.add_entity(MapEntity(position_x=chunk_end_x - self.PARKING_SLOT_HEIGHT, position_y=parking_lot_row_start_y, width=self.PARKING_SLOT_WIDTH, length=self.PARKING_SLOT_HEIGHT, theta=math.pi, type=MapEntity.ENTITY_PARKING_SLOT))
                            parking_lot_row_start_y += self.PARKING_SLOT_WIDTH
                    else:
                        parking_lot_row_start_x = chunk_start_x + self.CHUNK_PARKING_LOT_PADDING
                        parking_lot_row_end_x = chunk_end_x - self.CHUNK_PARKING_LOT_PADDING
                        while parking_lot_row_start_x < parking_lot_row_end_x:
                            map.add_entity(MapEntity(position_x=parking_lot_row_start_x, position_y=chunk_start_y + self.PARKING_SLOT_HEIGHT, width=self.PARKING_SLOT_WIDTH, length=self.PARKING_SLOT_HEIGHT, theta=math.pi/2, type=MapEntity.ENTITY_PARKING_SLOT))
                            map.add_entity(MapEntity(position_x=parking_lot_row_start_x, position_y=chunk_end_y - self.PARKING_SLOT_HEIGHT, width=self.PARKING_SLOT_WIDTH, length=self.PARKING_SLOT_HEIGHT, theta=-math.pi/2, type=MapEntity.ENTITY_PARKING_SLOT))
                            parking_lot_row_start_x += self.PARKING_SLOT_WIDTH
        
        ## escolhe uma das vagas para se tornar o ponto de partida e o objetivo de estacionamento
        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_PARKING_GOAL
        map.parking_goal = chosen_parking_slot

        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_START
        map.start_position = chosen_parking_slot

        return map






    




