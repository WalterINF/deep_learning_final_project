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

    def load_map(self, map_name: str = "MAPA_1", width: float = 60.0, height: float = 60.0) -> Map:
        if( map_name == "MAPA_1"):
            return self._create_default_map(width, height)
        else:
            raise ValueError(f"Mapa '{map_name}' não encontrado")


    def _create_default_map(self, width: float, height: float) -> Map:
        width_wall = 2.0
        width_parking_slot = 7.0
        height_parking_slot = 12.0
        spawn_margin = min(10.0, width/2, height/2)
        max_attempts = 1000
        map = Map((width, height))
        # parede esquerda
        map.add_entity(MapEntity(position_x=0, position_y=height/2, width=width_wall, height=height, theta=0, type=MapEntity.ENTITY_WALL))
        # parede direita
        map.add_entity(MapEntity(position_x=width, position_y=height/2, width=width_wall, height=height, theta=0, type=MapEntity.ENTITY_WALL))
        # parede superior
        map.add_entity(MapEntity(position_x=width/2, position_y=0, width=width, height=width_wall, theta=0, type=MapEntity.ENTITY_WALL))
        # parede inferior
        map.add_entity(MapEntity(position_x=width/2, position_y=height, width=width, height=width_wall, theta=0, type=MapEntity.ENTITY_WALL))

        new_parking_goal = MapEntity(position_x=random.uniform(spawn_margin, width -spawn_margin), position_y=random.uniform(spawn_margin, height -spawn_margin), width=20.0, height=20.0, theta=math.radians(random.uniform(0, 360)), type=MapEntity.ENTITY_PARKING_GOAL)
        attempts = 0
        while map.check_collision_with_entities(new_parking_goal):
            attempts += 1
            if attempts > max_attempts:
                raise Exception(f"Failed to add parking goal to map after {max_attempts} attempts")
            new_parking_goal = MapEntity(position_x=random.uniform(spawn_margin, width -spawn_margin), position_y=random.uniform(spawn_margin, height -spawn_margin), width=20.0, height=20.0, theta=math.radians(random.uniform(0, 360)), type=MapEntity.ENTITY_PARKING_GOAL)
        map.add_entity(new_parking_goal)
        map.parking_goal = new_parking_goal

        new_start_position = MapEntity(
            position_x=random.uniform(spawn_margin, width -spawn_margin), 
            position_y=random.uniform(spawn_margin, height -spawn_margin), 
                width=12, 
                height=7.0, 
                theta=random.uniform(0, 2*math.pi), 
                type=MapEntity.ENTITY_START
            )
        while map.check_collision_with_entities(new_start_position):
            attempts += 1
            if attempts > max_attempts:
                raise Exception(f"Failed to add start position to map after {max_attempts} attempts")
            new_start_position = MapEntity(
                position_x=random.uniform(spawn_margin, width -spawn_margin), 
                position_y=random.uniform(spawn_margin, height -spawn_margin), 
                width=7, 
                height=12.0, 
                theta=random.uniform(0, 2*math.pi), 
                type=MapEntity.ENTITY_START
            )
        map.add_entity(new_start_position)
        map.start_position = new_start_position


        return map