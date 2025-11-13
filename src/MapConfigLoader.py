from Simulation import Map, MapEntity
import json
from typing import Any
import math
def save_map_to_json(map: Map, filename: str) -> None:
    data = {
        "size": [float(map.size_x), float(map.size_y)],
        "entities": [
            {
                "position_x": float(entity.position_x),
                "position_y": float(entity.position_y),
                "width": float(entity.width),
                "height": float(entity.height),
                "theta": float(entity.theta),
                "type": str(entity.type),
            }
            for entity in map.get_entities()
        ],
    }
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_map_from_json(filename: str) -> Map:
    with open(filename, "r", encoding="utf-8") as fp:
        data: dict[str, Any] = json.load(fp)

    size = data.get("size") or data.get("map_size")
    if size is None:
        raise ValueError("Map JSON missing required 'size' field.")

    if isinstance(size, dict):
        size_x = float(size.get("x"))
        size_y = float(size.get("y"))
    else:
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("Map 'size' must be a list/tuple of length 2 or an object with x/y.")
        size_x = float(size[0])
        size_y = float(size[1])

    loaded_map = Map((size_x, size_y))

    for ent in data.get("entities", []):
        position_x = float(ent["position_x"])
        position_y = float(ent["position_y"])
        width = float(ent["width"])
        height = float(ent["height"])
        theta = float(ent.get("theta", 0.0))
        type_str = str(ent.get("type", MapEntity.ENTITY_WALL))
        loaded_map.add_entity(
            MapEntity(
                position_x=position_x,
                position_y=position_y,
                width=width,
                height=height,
                theta=theta,
                type=type_str,
            )
        )

    return loaded_map

def create_default_map() -> Map:
    height = 90.0
    width = 90.0
    width_wall = 2.0
    width_parking_slot = 15.0
    height_parking_slot = 24.0
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
    map.add_entity(MapEntity(position_x= 15, position_y= 75, width=width_parking_slot, height=height_parking_slot, theta=math.radians(0), type=MapEntity.ENTITY_PARKING_GOAL))
    return map