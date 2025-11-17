import json
from typing import Any, Dict, List, Optional
import os
from Simulation import ArticulatedVehicle, Map, MapEntity
import random
import math

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

    def __init__(self, json_path: str):
        self.json_path = json_path

    def load_map(self, map_name: str = "MAPA_1") -> Map:
        if( map_name == "MAPA_1"):
            return self._create_default_map()
        else:
            raise ValueError(f"Mapa '{map_name}' não encontrado")


    def _create_default_map(self) -> Map:
        max_parking_lots = 100 # máximo de vagas de estacionamento
        width = 150.0
        height = 150.0
        width_wall = 2.0
        width_parking_slot = 7.0
        height_parking_slot = 12.0
        spawn_margin = min(20.0, width/2, height/2)
        map = Map((width, height))
        # parede esquerda
        map.add_entity(MapEntity(position_x=0, position_y=height/2, width=width_wall, height=height, theta=0, type=MapEntity.ENTITY_WALL))
        # parede direita
        map.add_entity(MapEntity(position_x=width, position_y=height/2, width=width_wall, height=height, theta=0, type=MapEntity.ENTITY_WALL))
        # parede superior
        map.add_entity(MapEntity(position_x=width/2, position_y=0, width=width, height=width_wall, theta=0, type=MapEntity.ENTITY_WALL))
        # parede inferior
        map.add_entity(MapEntity(position_x=width/2, position_y=height, width=width, height=width_wall, theta=0, type=MapEntity.ENTITY_WALL))

        parking_lots_rows_heights = []
        parking_lots_row_spacing = 20.0
        height_row = spawn_margin + height_parking_slot/2
        while height_row < height:
            parking_lots_rows_heights.append(height_row)
            height_row += parking_lots_row_spacing + height_parking_slot

        parking_lots_start_x = spawn_margin + width_parking_slot/2
        parking_lots_end_x = width - spawn_margin - width_parking_slot/2

        
        for row_height in parking_lots_rows_heights:
            parking_lots_start_x = spawn_margin + width_parking_slot/2
            while parking_lots_start_x < parking_lots_end_x:
                new_parking_slot = MapEntity(position_x=parking_lots_start_x, position_y=row_height, width=width_parking_slot, height=height_parking_slot, theta=0, type=MapEntity.ENTITY_PARKING_SLOT)
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