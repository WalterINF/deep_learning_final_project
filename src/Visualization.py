from turtle import colormode
from typing import Iterable, List, Union
from pathlib import Path
import numpy as np
from PIL import Image
import pygame
from src.Simulation import Simulation, Map, ArticulatedVehicle, MapEntity
import math
from pygame import gfxdraw
from pygame.font import Font

# Initialize pygame.font before creating Font objects
pygame.font.init()

color_mappings = {
    MapEntity.ENTITY_NOTHING: ((255, 255, 255)), ## branco
    MapEntity.ENTITY_WALL: ((0, 0, 0)), ## cinza médio
    MapEntity.ENTITY_PARKING_SLOT: ((200, 200, 200)), ## cinza claro
    MapEntity.ENTITY_PARKING_GOAL: ((0, 255, 0)), ## verde
    MapEntity.ENTITY_START: ((255, 255, 0)), ## amarelo
}

font = Font(None, 20)

def to_rgb_array(
    simulation: Simulation,
    img_size: tuple[int, int] = (288, 288),
    distance_map: np.ndarray | None = None,
    grid_resolution: float = 1.0,
    observation: tuple[float, float, float, float] | None = None,
    heuristic_value: float = None,
) -> list[list[list[int]]]:
    """Gera uma imagem RGB do mapa e suas entidades.

    - O mapa é um retângulo branco com bordas vermelhas.
    - As entidades são desenhadas como seus retângulos de bounding box (contorno).
    Args:
        map: Map - o mapa a ser desenhado
        img_size: tuple[int, int] - o tamanho da imagem a ser gerada

    Returns:
        list[list[list[int]]]: Array [H][W][3] com valores RGB (0-255)
    """

    map_width, map_height = simulation.map.get_size()

    # Define dimensões de saída da imagem
    width = max(map_width, int(img_size[0]))
    height = max(map_height, int(img_size[1]))

    # Fatores de escala do espaço do mapa -> pixels da imagem
    scale_x = (width / float(simulation.map.size_x)) if simulation.map.size_x else 1.0
    scale_y = (height / float(simulation.map.size_y)) if simulation.map.size_y else 1.0

    # Superfície onde desenharemos (não requer display inicializado)
    surface = pygame.Surface((width, height))


    # Fundo cinza 
    surface.fill((128, 128, 128))

    # Draw cost map if provided
    if distance_map is not None:
        # Find min and max valid costs (excluding -1 and -2 which are unvisited/obstacles)
        valid_mask = distance_map >= 0
        if valid_mask.any():
            min_cost = float(distance_map[valid_mask].min())
            max_cost = float(distance_map[valid_mask].max())
            cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
            
            grid_w, grid_h = distance_map.shape
            circle_radius = 4
            
            for i in range(grid_w):
                for j in range(grid_h):
                    cost = distance_map[i, j]
                    if cost >= 0:
                        normalized = (cost - min_cost) / cost_range
                        # Interpolate from green (lowest cost) to red (highest cost)
                        r = int(255 * normalized)
                        g = int(255 * (1 - normalized))
                        b = 0
                        
                        # Calculate center position of the cell in pixel coordinates
                        cx = (i + 0.5) * grid_resolution * scale_x
                        cy = (j + 0.5) * grid_resolution * scale_y
                        
                        # Draw small circle
                        gfxdraw.filled_circle(surface, int(cx), int(cy), circle_radius, (r, g, b))


    # Desenha cada entidade como o contorno do seu retângulo (bounding box)

    for entity in simulation.map.get_entities():
        if entity.type == MapEntity.ENTITY_PARKING_SLOT:
            color = color_mappings[entity.type]
            bbox = entity.get_bounding_box()
            corners = bbox.get_corners()  # [(x,y), ...] em coords globais
            # Converte coordenadas do espaço do mapa para pixels de saída
            scaled_corners = [(x * scale_x, y * scale_y) for (x, y) in corners]
            gfxdraw.aapolygon(surface, scaled_corners, color)
            gfxdraw.line(surface, int(scaled_corners[1][0]), int(scaled_corners[1][1]), int(scaled_corners[2][0]), int(scaled_corners[2][1]), (128, 128, 128))

        elif entity.type == MapEntity.ENTITY_WALL:
            color = color_mappings[entity.type]
            bbox = entity.get_bounding_box()
            corners = bbox.get_corners()  # [(x,y), ...] em coords globais
            # Converte coordenadas do espaço do mapa para pixels de saída
            scaled_corners = [(x * scale_x, y * scale_y) for (x, y) in corners]
            gfxdraw.filled_polygon(surface, scaled_corners, color)
            gfxdraw.aapolygon(surface, scaled_corners, color)
        elif entity.type == MapEntity.ENTITY_PARKING_GOAL:
            color = color_mappings[entity.type]
            bbox = entity.get_bounding_box()
            corners = bbox.get_corners()  # [(x,y), ...] em coords globais
            # Converte coordenadas do espaço do mapa para pixels de saída
            scaled_corners = [(x * scale_x, y * scale_y) for (x, y) in corners]
            gfxdraw.aapolygon(surface, scaled_corners, color)
            gfxdraw.line(surface, int(scaled_corners[1][0]), int(scaled_corners[1][1]), int(scaled_corners[2][0]), int(scaled_corners[2][1]), (128, 128, 128))
        elif entity.type == MapEntity.ENTITY_START:
            color = color_mappings[entity.type]
            bbox = entity.get_bounding_box()
            corners = bbox.get_corners()  # [(x,y), ...] em coords globais
            # Converte coordenadas do espaço do mapa para pixels de saída
            scaled_corners = [(x * scale_x, y * scale_y) for (x, y) in corners]
            gfxdraw.aapolygon(surface, scaled_corners, color)
            gfxdraw.aapolygon(surface, scaled_corners, color)

    # Desenha as rodas do trator
    wheels_bboxes = simulation.vehicle.get_wheels_bounding_boxes()
    for wheel_bbox in wheels_bboxes:
        wheel_corners = wheel_bbox.get_corners()
        wheel_scaled = [(x * scale_x, y * scale_y) for (x, y) in wheel_corners]
        gfxdraw.aapolygon(surface, wheel_scaled, (0, 0, 0))
        gfxdraw.filled_polygon(surface, wheel_scaled, (0, 0, 0))

    # Desenha o veículo
    if simulation.vehicle is not None:
        # Tractor
        tractor_bbox = simulation.vehicle.get_tractor_bounding_box()
        tractor_corners = tractor_bbox.get_corners()
        tractor_scaled = [(x * scale_x, y * scale_y) for (x, y) in tractor_corners]
        gfxdraw.aapolygon(surface, tractor_scaled, (128, 128, 255)) ## azul claro
        gfxdraw.filled_polygon(surface, tractor_scaled, (128, 128, 255)) ## azul claro

        # Trailer
        trailer_bbox = simulation.vehicle.get_bounding_box_trailer()
        trailer_corners = trailer_bbox.get_corners()
        trailer_scaled = [(x * scale_x, y * scale_y) for (x, y) in trailer_corners]
        gfxdraw.aapolygon(surface, trailer_scaled, (0, 200, 0)) ## verde
        gfxdraw.filled_polygon(surface, trailer_scaled, (0, 200, 0)) ## verde

        # Desenha os raycasts com círculos nas pontas

        for raycast_result in simulation.vehicle.get_raycast_results().values():
            raycast_position = (raycast_result.origin_x * scale_x, raycast_result.origin_y * scale_y)
            raycast_angle = raycast_result.theta
            raycast_length = raycast_result.length * scale_x
            line_end_position = (
                raycast_position[0] + raycast_length * math.cos(raycast_angle),
                raycast_position[1] + raycast_length * math.sin(raycast_angle),
            )

            # Define a cor do raycast com base no tipo da entidade atingida.
            # Quando não há entidade, usa a cor associada a ENTITY_NOTHING.
            if raycast_result.entity is not None:
                color = color_mappings[raycast_result.entity.type]
            else:
                color = (255, 0, 0) # vermelho

            gfxdraw.line(
                surface,
                int(raycast_position[0]),
                int(raycast_position[1]),
                int(line_end_position[0]),
                int(line_end_position[1]),
                color,
            )
            gfxdraw.filled_circle(
                surface,
                int(line_end_position[0]),
                int(line_end_position[1]),
                int(0.5 * scale_x),
                color,
            )

        # desenha ações na tela
        if simulation.vehicle.get_tractor_velocity() is not None:
            velocity_text = f"Velocity: {simulation.vehicle.get_tractor_velocity():.2f}"
            velocity_surface = font.render(velocity_text, True, (0, 0, 0))
            surface.blit(velocity_surface, (10, 10))

        if simulation.vehicle.get_tractor_alpha() is not None:
            alpha_text = f"Alpha: {simulation.vehicle.get_tractor_alpha():.2f}"
            alpha_surface = font.render(alpha_text, True, (0, 0, 0))
            surface.blit(alpha_surface, (10, 30))

        if observation is not None:
            z1_text = f"z1: {observation[0]:.2f}"
            z1_surface = font.render(z1_text, True, (0, 0, 0))
            surface.blit(z1_surface, (10, 50))
            z2_text = f"z2: {observation[1]:.2f}"
            z2_surface = font.render(z2_text, True, (0, 0, 0))
            surface.blit(z2_surface, (10, 70))
            z3_text = f"z3: {observation[2]:.2f}"
            z3_surface = font.render(z3_text, True, (0, 0, 0))
            surface.blit(z3_surface, (10, 90))
            z4_text = f"z4: {observation[3]:.2f}"
            z4_surface = font.render(z4_text, True, (0, 0, 0))
            surface.blit(z4_surface, (10, 110))

        if observation is not None:
            for i in range(4, 4 + len(observation) - 4):
                raycast_text = f"Raycast {i - 4}: {observation[i]:.2f}"
                raycast_surface = font.render(raycast_text, True, (0, 0, 0))
                surface.blit(raycast_surface, (10, 175 + i*15 - 4 * 20))

        if heuristic_value is not None:
            heuristic_text = f"Heuristic: {heuristic_value:.2f}"
            raycast_surface = font.render(heuristic_text, True, (0, 0, 0))
            surface.blit(raycast_surface, (10, 130))




        #desenha distancia do objetivo
        vehicle_position = simulation.vehicle.get_tractor_position()
        vehicle_position_scaled = (vehicle_position[0] * scale_x, vehicle_position[1] * scale_y)
        goal_position = simulation.map.get_parking_goal_position()
        goal_position_scaled = (goal_position[0] * scale_x, goal_position[1] * scale_y)
        gfxdraw.line(surface, int(vehicle_position_scaled[0]), int(vehicle_position_scaled[1]), int(goal_position_scaled[0]), int(goal_position_scaled[1]), (0, 0, 255)) ## azul
        gfxdraw.filled_circle(surface, int(goal_position_scaled[0]), int(goal_position_scaled[1]), int(0.5 * scale_x), (0, 0, 255)) ## azul

    # Extrai os pixels como bytes em ordem RGB
    raw_bytes = pygame.image.tostring(surface, "RGB")

    # Converte bytes em array [H][W][3] de ints sem depender de numpy
    rgb_array: list[list[list[int]]] = []
    idx = 0
    row_stride = width * 3
    for y in range(height):
        row: list[list[int]] = []
        base = y * row_stride
        for x in range(width):
            i = base + x * 3
            r = raw_bytes[i]
            g = raw_bytes[i + 1]
            b = raw_bytes[i + 2]
            row.append([r, g, b])
        rgb_array.append(row)

    return rgb_array

def save_frames_as_mp4(
    frames: Iterable[Union[np.ndarray, List[List[List[int]]]]],
    output_path: Union[str, Path],
    fps: int = 4,
) -> None:
    """
    Save a list of RGB frames as an MP4 video.

    Args:
        frames: Iterable of frames; each frame is [H][W][3] (or convertible) with 0-255 ints
        output_path: Path where the video/GIF will be saved
        fps: Frames per second for the output video/GIF

    Raises:
        ImportError: If imageio is not installed
        ValueError: If frames is empty
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for video/GIF creation. Install it with: pip install imageio imageio-ffmpeg"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize_frame(frame: Union[np.ndarray, List[List[List[int]]]]) -> np.ndarray:
        array = np.asarray(frame)

        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)

        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                "Each frame must have shape [H, W, 3] (RGB) or [H, W] (grayscale)"
            )

        if array.shape[2] == 4:
            array = array[:, :, :3]

        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                # Assume 0-1 normalized floats and scale up.
                if array.max() <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        return array

    frame_iterator = iter(frames)
    try:
        first_frame = _normalize_frame(next(frame_iterator))
    except StopIteration as exc:
        raise ValueError("Frames iterable is empty") from exc

    frame_count = 0
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8) as writer:
        writer.append_data(first_frame)
        frame_count += 1
        for frame in frame_iterator:
            writer.append_data(_normalize_frame(frame))
            frame_count += 1

    print(f"Saved {frame_count} frames as MP4 to {output_path}")


class VideoRecorder:
    """
    Helper class to stream frames directly to video without storing them in memory.

    Example:
        recorder = VideoRecorder("episode.mp4", fps=10)
        for frame in generate_frames():
            recorder.append(frame)
        recorder.close()
    """

    def __init__(self, output_path: Union[str, Path], fps: int = 4, codec: str = "libx264", quality: int = 8):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self._writer = None
        self._frame_count = 0

    def append(self, frame: Union[np.ndarray, List[List[List[int]]]]) -> None:
        if self._writer is None:
            import imageio

            normalized = self._normalize_frame(frame)
            self._writer = imageio.get_writer(
                str(self.output_path), fps=self.fps, codec=self.codec, quality=self.quality
            )
            self._writer.append_data(normalized)
            self._frame_count += 1
        else:
            self._writer.append_data(self._normalize_frame(frame))
            self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            print(f"Saved {self._frame_count} frames as MP4 to {self.output_path}")

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _normalize_frame(frame: Union[np.ndarray, List[List[List[int]]]]) -> np.ndarray:
        array = np.asarray(frame)

        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)

        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                "Each frame must have shape [H, W, 3] (RGB) or [H, W] (grayscale)"
            )

        if array.shape[2] == 4:
            array = array[:, :, :3]

        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        return array


def save_rgb_array_as_png(rgb_array: list[list[list[int]]], path: str = "map.png") -> None:
    """Salva um array RGB [H][W][3] como imagem PNG no caminho indicado, sem depender de numpy."""
    if not rgb_array or not rgb_array[0]:
        raise ValueError("rgb_array vazio ou com dimensões inválidas")

    height = len(rgb_array)
    width = len(rgb_array[0])

    # Achata para bytes em ordem de varredura por linhas (row-major), formato RGB
    flat_bytes = bytearray(width * height * 3)
    idx = 0
    for y in range(height):
        row = rgb_array[y]
        if len(row) != width:
            raise ValueError("Linhas do rgb_array possuem larguras diferentes")
        for x in range(width):
            r, g, b = row[x]
            flat_bytes[idx] = int(r) & 0xFF
            flat_bytes[idx + 1] = int(g) & 0xFF
            flat_bytes[idx + 2] = int(b) & 0xFF
            idx += 3

    image = Image.frombytes("RGB", (width, height), bytes(flat_bytes))
    image.save(path, format="PNG")


def save_as_png(
    map: Map, path: str = "map.png", img_size: tuple[int, int] = None, vehicle: ArticulatedVehicle | None = None
) -> None:
    rgb_array = to_rgb_array(map, vehicle, img_size)
    save_rgb_array_as_png(rgb_array, path)