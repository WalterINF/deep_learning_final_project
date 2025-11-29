"""Generate fallback parking maps when legacy assets are missing.

This module provides a lightweight, dependency-free translation of the legacy
`DefaultMapGenerator` logic. It creates occupancy grids compatible with the
`Mapa` class used by the legacy collision routines: value 1 represents free
space, value 0 represents an obstacle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(slots=True)
class MapEntity:
    """Simple representation of rectangular static geometry."""

    position_x: float
    position_y: float
    width: float
    length: float
    theta: float
    collidable: bool = True

    def axis_aligned_bounds(self) -> tuple[float, float, float, float]:
        """Return (min_x, max_x, min_y, max_y) in meters."""

        # The legacy generator only uses angles 0, pi/2, pi, -pi/2. We exploit
        # that to avoid full polygon rasterisation.
        angle = (abs(self.theta) + math.tau) % math.pi
        half_w = 0.5 * self.width
        half_l = 0.5 * self.length

        if math.isclose(angle, 0.0, abs_tol=1e-6):
            span_x = half_l
            span_y = half_w
        else:
            span_x = half_w
            span_y = half_l

        min_x = self.position_x - span_x
        max_x = self.position_x + span_x
        min_y = self.position_y - span_y
        max_y = self.position_y + span_y
        return min_x, max_x, min_y, max_y


def _default_entities(width: float, height: float) -> List[MapEntity]:
    """Mirror the legacy DefaultMapGenerator layout."""

    wall_width = 4.0
    slot_width = 5.0
    slot_height = 12.0
    spawn_padding = 30.0
    wall_padding = 3.0
    n_rows = 2

    entities: List[MapEntity] = []

    # Boundary walls
    entities.append(MapEntity(0.0, height / 2.0, wall_width, height, math.pi / 2))
    entities.append(MapEntity(width, height / 2.0, wall_width, height, -math.pi / 2))
    entities.append(MapEntity(width / 2.0, 0.0, wall_width, width, 0.0))
    entities.append(MapEntity(width / 2.0, height, wall_width, width, 0.0))

    # Horizontal separators + parking slots
    row_heights = [i * height / (n_rows + 1) for i in range(1, n_rows + 1)]
    parking_end_x = width - spawn_padding - slot_width / 2.0

    rng = np.random.default_rng(42)

    for row_y in row_heights:
        entities.append(
            MapEntity(
                width / 2.0,
                row_y,
                width - 2.0 * spawn_padding,
                1.0,
                math.pi / 2,
            )
        )

        x_pos = spawn_padding + slot_width / 2.0
        while x_pos < parking_end_x:
            # Upper slot (facing down)
            entities.append(
                MapEntity(
                    x_pos,
                    row_y - slot_height / 2.0 - wall_padding,
                    slot_width,
                    slot_height,
                    -math.pi / 2,
                    collidable=False,
                )
            )
            # Lower slot (facing up)
            entities.append(
                MapEntity(
                    x_pos,
                    row_y + slot_height / 2.0 + wall_padding,
                    slot_width,
                    slot_height,
                    math.pi / 2,
                    collidable=False,
                )
            )
            x_pos += slot_width

    # Place a deterministic start/goal pair using RNG for reproducibility
    parking_slots = [entity for entity in entities if not entity.collidable]
    if parking_slots:
        start_idx = rng.integers(len(parking_slots))
        goal_idx = rng.integers(len(parking_slots))
        # Mark as collidable=False to keep the space free; metadata handled elsewhere.
        parking_slots[start_idx].collidable = False
        parking_slots[goal_idx].collidable = False

    return entities


def rasterise_entities(
    entities: Iterable[MapEntity],
    width_m: float,
    height_m: float,
    resolution: float,
) -> np.ndarray:
    """Convert entities to an occupancy grid (1 free, 0 blocked)."""

    width_px = int(math.ceil(width_m * resolution))
    height_px = int(math.ceil(height_m * resolution))
    grid = np.ones((height_px, width_px), dtype=np.uint8)

    for entity in entities:
        if not entity.collidable:
            continue

        min_x, max_x, min_y, max_y = entity.axis_aligned_bounds()
        x0 = max(0, int(math.floor(min_x * resolution)))
        x1 = min(width_px, int(math.ceil(max_x * resolution)))
        y0 = max(0, int(math.floor(min_y * resolution)))
        y1 = min(height_px, int(math.ceil(max_y * resolution)))
        if x0 >= x1 or y0 >= y1:
            continue
        grid[y0:y1, x0:x1] = 0

    return grid


def generate_default_map(path: Path, width: float = 150.0, height: float = 150.0, resolution: float = 2.0) -> Path:
    """Create a deterministic fallback map and persist it as .npy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    entities = _default_entities(width, height)
    grid = rasterise_entities(entities, width, height, resolution)
    np.save(path, grid)
    return path
