import json
from typing import Any, Dict, List, Optional
import os
from Simulation import ArticulatedVehicle, Map, MapEntity

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
        if not os.path.exists(self._json_path):
            raise FileNotFoundError(f"Arquivo JSON não encontrado: {self._json_path}")
        with open(self._json_path, "r", encoding="utf-8") as f:
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