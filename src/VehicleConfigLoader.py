import json
from typing import Any, Dict, List, Optional
import os


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

    def get_geometry(self, name: str) -> Dict[str, Any]:
        if name not in self._vehicles_by_name:
            raise KeyError(f"Veículo '{name}' não encontrado no arquivo {self._json_path}")
        return self._vehicles_by_name[name]


