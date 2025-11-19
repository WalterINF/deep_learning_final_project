import json
from typing import Any, Dict, List, Optional
import os
from Simulation import ArticulatedVehicle, Map, MapEntity
import random
import math
import numpy as np
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



class MapConfigLoader:

    DEFAULT_MAP_NAME = "MAPA_1"
    COMPLEX_MAP_NAME = "MAPA_COMPLEXO"

    def __init__(self):
        self.json_path = "config/lista_mapas.json"

    def load_map(self, map_name: str = DEFAULT_MAP_NAME) -> Map:
        if( map_name == self.DEFAULT_MAP_NAME):
            return self._create_default_map()
        elif( map_name == self.COMPLEX_MAP_NAME):
            return self._create_complex_map()
        else:
            raise ValueError(f"Mapa '{map_name}' não encontrado")


    def _create_default_map(self) -> Map:
        max_parking_lots = 100 # máximo de vagas de estacionamento
        width = 100.0
        height = 100.0
        width_wall = 4.0
        width_parking_slot = 4.5
        height_parking_slot = 12.0
        spawn_margin = min(20.0, width/2, height/2)
        map = Map((width, height))
        # parede esquerda
        map.add_entity(MapEntity(position_x=0, position_y=height/2, width=width_wall, length=height, theta=math.pi/2, type=MapEntity.ENTITY_WALL))
        # parede direita
        map.add_entity(MapEntity(position_x=width, position_y=height/2, width=width_wall, length=height, theta=-math.pi/2, type=MapEntity.ENTITY_WALL))
        # parede superior
        map.add_entity(MapEntity(position_x=width/2, position_y=0, width=width_wall, length=width, theta=0.0, type=MapEntity.ENTITY_WALL))
        # parede inferior
        map.add_entity(MapEntity(position_x=width/2, position_y=height, width=width_wall, length=width, theta=0.0, type=MapEntity.ENTITY_WALL))

        parking_lots_rows_heights = []
        parking_lots_row_spacing = 25.0
        height_row = spawn_margin + height_parking_slot/2
        while height_row < height:
            parking_lots_rows_heights.append(height_row)
            height_row += parking_lots_row_spacing + height_parking_slot

        parking_lots_start_x = spawn_margin + width_parking_slot/2
        parking_lots_end_x = width - spawn_margin - width_parking_slot/2

        
        for row_height in parking_lots_rows_heights:
            parking_lots_start_x = spawn_margin + width_parking_slot/2
            while parking_lots_start_x < parking_lots_end_x:
                angle = random.choice([-math.pi/2, math.pi/2])
                new_parking_slot = MapEntity(position_x=parking_lots_start_x, position_y=row_height, width=width_parking_slot, length=height_parking_slot, theta=angle, type=MapEntity.ENTITY_PARKING_SLOT)
                map.add_entity(new_parking_slot)
                parking_lots_start_x += width_parking_slot

        ## escolhe uma das vagas para se tornar o ponoito de partida e o objetivo de estacionamento
        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_PARKING_GOAL
        map.parking_goal = chosen_parking_slot

        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_START
        map.start_position = chosen_parking_slot


        return map

    def _create_complex_map(self) -> Map:
        max_parking_lots = 100 # máximo de vagas de estacionamento
        width = 250.0
        height = 250.0
        width_wall = 4.0
        width_parking_slot = 4.5
        height_parking_slot = 12.0
        spawn_margin = min(20.0, width/2, height/2)
        map = Map((width, height))
        
        # parede esquerda
        map.add_entity(MapEntity(position_x=0, position_y=height/2, width=width_wall, length=height, theta=math.pi/2, type=MapEntity.ENTITY_WALL))
        # parede direita
        map.add_entity(MapEntity(position_x=width, position_y=height/2, width=width_wall, length=height, theta=-math.pi/2, type=MapEntity.ENTITY_WALL))
        # parede superior
        map.add_entity(MapEntity(position_x=width/2, position_y=0, width=width_wall, length=width, theta=0.0, type=MapEntity.ENTITY_WALL))
        # parede inferior
        map.add_entity(MapEntity(position_x=width/2, position_y=height, width=width_wall, length=width, theta=0.0, type=MapEntity.ENTITY_WALL))



        #divide o mapa em uma matriz de 50x50 metros
        map_divisions = (4,4)
        chunk_size = (width/map_divisions[0], height/map_divisions[1])
        map_matrix = np.zeros(map_divisions, dtype=int)  # FIX: use int, not bool

        # define o chunk [1,1] como estacionamento obrigatório
        map_matrix[1,1] = random.choice([2,3])


        # não sobrescrever o [1,1] no sorteio
        for i in range(map_divisions[0]):
            for j in range(map_divisions[1]):
                if (i, j) == (1, 1):
                    continue
                map_matrix[i,j] = random.choice([0,2,3])
                #0: parede
                #1: nada
                #2: chunk com uma fileira de vagas de estacionamento
                #3: chunk com duas fileiras de vagas de estacionamento

        chunk_parking_lot_padding = 15.0
        chunk_wall_padding = 25.0

        for i in range(map_divisions[0]):
            for j in range(map_divisions[1]):
                chunk_center_x = i*chunk_size[0] + chunk_size[0]/2
                chunk_center_y = j*chunk_size[1] + chunk_size[1]/2
                chunk_start_x = i*chunk_size[0]
                chunk_start_y = j*chunk_size[1]
                chunk_end_x = chunk_start_x + chunk_size[0]
                chunk_end_y = chunk_start_y + chunk_size[1]
                if map_matrix[i,j] == 0:
                    #preenche o chunk com paredes
                    map.add_entity(MapEntity(position_x=chunk_center_x, position_y=chunk_center_y, width=chunk_size[0] - chunk_wall_padding, length=chunk_size[1] - chunk_wall_padding, theta=0.0, type=MapEntity.ENTITY_WALL))
                elif map_matrix[i,j] == 2:
                    #preenche com uma fileira de vagas de estacionamento
                    vertical_or_horizontal = random.choice([True, False])
                    if vertical_or_horizontal:
                        parking_lot_row_start_y = chunk_start_y + chunk_parking_lot_padding
                        parking_lot_row_end_y = chunk_end_y - chunk_parking_lot_padding
                        while parking_lot_row_start_y < parking_lot_row_end_y:
                            map.add_entity(MapEntity(position_x=chunk_center_x, position_y=parking_lot_row_start_y, width=width_parking_slot, length=height_parking_slot, theta=random.choice([0.0, math.pi]), type=MapEntity.ENTITY_PARKING_SLOT))
                            parking_lot_row_start_y += width_parking_slot
                    else:
                        parking_lot_row_start_x = chunk_start_x + chunk_parking_lot_padding
                        parking_lot_row_end_x = chunk_end_x - chunk_parking_lot_padding
                        while parking_lot_row_start_x < parking_lot_row_end_x:
                            map.add_entity(MapEntity(position_x=parking_lot_row_start_x, position_y=chunk_center_y, width=width_parking_slot, length=height_parking_slot, theta=random.choice([-math.pi/2, math.pi/2]), type=MapEntity.ENTITY_PARKING_SLOT))
                            parking_lot_row_start_x += width_parking_slot
                elif map_matrix[i,j] == 3:
                    #preenche com duas fileiras de vagas de estacionamento
                    vertical_or_horizontal = random.choice([True, False])
                    if vertical_or_horizontal:
                        parking_lot_row_start_y = chunk_start_y + chunk_parking_lot_padding
                        parking_lot_row_end_y = chunk_end_y - chunk_parking_lot_padding
                        while parking_lot_row_start_y < parking_lot_row_end_y:
                            map.add_entity(MapEntity(position_x=chunk_start_x + height_parking_slot, position_y=parking_lot_row_start_y, width=width_parking_slot, length=height_parking_slot, theta=0.0, type=MapEntity.ENTITY_PARKING_SLOT))
                            map.add_entity(MapEntity(position_x=chunk_end_x - height_parking_slot, position_y=parking_lot_row_start_y, width=width_parking_slot, length=height_parking_slot, theta=math.pi, type=MapEntity.ENTITY_PARKING_SLOT))
                            parking_lot_row_start_y += width_parking_slot
                    else:
                        parking_lot_row_start_x = chunk_start_x + chunk_parking_lot_padding
                        parking_lot_row_end_x = chunk_end_x - chunk_parking_lot_padding
                        while parking_lot_row_start_x < parking_lot_row_end_x:
                            map.add_entity(MapEntity(position_x=parking_lot_row_start_x, position_y=chunk_start_y + height_parking_slot, width=width_parking_slot, length=height_parking_slot, theta=math.pi/2, type=MapEntity.ENTITY_PARKING_SLOT))
                            map.add_entity(MapEntity(position_x=parking_lot_row_start_x, position_y=chunk_end_y - height_parking_slot, width=width_parking_slot, length=height_parking_slot, theta=-math.pi/2, type=MapEntity.ENTITY_PARKING_SLOT))
                            parking_lot_row_start_x += width_parking_slot
        
        ## escolhe uma das vagas para se tornar o ponto de partida e o objetivo de estacionamento
        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_PARKING_GOAL
        map.parking_goal = chosen_parking_slot

        chosen_parking_slot = map.get_random_parking_slot()
        chosen_parking_slot.type = MapEntity.ENTITY_START
        map.start_position = chosen_parking_slot

        return map
    




