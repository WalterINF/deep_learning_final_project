from typing import List, Union
from pathlib import Path
import numpy as np
from PIL import Image
import pygame
from Simulation import Map
import math
from pygame import gfxdraw
from Simulation import MapEntity


def to_rgb_array(map: Map, img_size: tuple[int, int] = None) -> list[list[list[int]]]:
    """Gera uma imagem RGB do mapa e suas entidades.

    - O mapa é um retângulo branco com bordas vermelhas.
    - As entidades são desenhadas como seus retângulos de bounding box (contorno).
    Args:
        map: Map - o mapa a ser desenhado
        img_size: tuple[int, int] - o tamanho da imagem a ser gerada

    Returns:
        list[list[list[int]]]: Array [H][W][3] com valores RGB (0-255)
    """

    # Define dimensões de saída da imagem
    if img_size is None:
        width = int(map.size_x)
        height = int(map.size_y)
    else:
        width = max(map.size_x, int(img_size[0]))
        height = max(map.size_y, int(img_size[1]))

    # Fatores de escala do espaço do mapa -> pixels da imagem
    scale_x = (width / float(map.size_x)) if map.size_x else 1.0
    scale_y = (height / float(map.size_y)) if map.size_y else 1.0

    # Superfície onde desenharemos (não requer display inicializado)
    surface = pygame.Surface((width, height))

    # Fundo branco
    surface.fill((255, 255, 255))

    # Borda vermelha do mapa
    pygame.draw.rect(surface, (255, 0, 0), pygame.Rect(0, 0, width, height), width=1)

    # Desenha cada entidade como o contorno do seu retângulo (bounding box)

    for entity in map.entities:
        color = map.color_mappings[entity.type]
        if color is None:
            color = (0, 0, 0)
        bbox = entity.get_bounding_box()
        corners = bbox.get_corners()  # [(x,y), ...] em coords globais
        # Converte coordenadas do espaço do mapa para pixels de saída
        scaled_corners = [(x * scale_x, y * scale_y) for (x, y) in corners]
        gfxdraw.aapolygon(surface, scaled_corners, color)

    # Desenha o veículo
    if map.vehicle is not None:
        # Tractor
        tractor_bbox = map.vehicle.get_bounding_box_tractor()
        tractor_corners = tractor_bbox.get_corners()
        tractor_scaled = [(x * scale_x, y * scale_y) for (x, y) in tractor_corners]
        gfxdraw.aapolygon(surface, tractor_scaled, (0, 0, 255))

        # Desenha as rodas do trator
        wheels_bboxes = map.vehicle.get_wheels_bounding_boxes()
        for wheel_bbox in wheels_bboxes:
            wheel_corners = wheel_bbox.get_corners()
            wheel_scaled = [(x * scale_x, y * scale_y) for (x, y) in wheel_corners]
            gfxdraw.aapolygon(surface, wheel_scaled, (0, 0, 0))
            gfxdraw.filled_polygon(surface, wheel_scaled, (0, 0, 0))

        # Trailer
        trailer_bbox = map.vehicle.get_bounding_box_trailer()
        trailer_corners = trailer_bbox.get_corners()
        trailer_scaled = [(x * scale_x, y * scale_y) for (x, y) in trailer_corners]
        gfxdraw.aapolygon(surface, trailer_scaled, (0, 200, 0))

        # Quinta roda do trator (círculo)
        axle_pos = (map.vehicle.position_x_trator * scale_x, map.vehicle.position_y_trator * scale_y)
        axle_radius = 0.5 * scale_x
        gfxdraw.filled_circle(surface, int(axle_pos[0]), int(axle_pos[1]), int(axle_radius), (200, 0, 0))

        # Desenha os raycasts com círculos nas pontas
        for raycast_name, raycast in map.vehicle.raycasts.items():
            raycast_position = (raycast.origin_x * scale_x, raycast.origin_y * scale_y)
            raycast_angle = raycast.theta
            raycast_length = raycast.length * scale_x
            line_end_position = (raycast_position[0] + raycast_length * math.cos(raycast_angle), raycast_position[1] + raycast_length * math.sin(raycast_angle))
            gfxdraw.line(surface, int(raycast_position[0]), int(raycast_position[1]), int(line_end_position[0]), int(line_end_position[1]), (0, 0, 255))
            gfxdraw.filled_circle(surface, int(line_end_position[0]), int(line_end_position[1]), int(0.5 * scale_x), (0, 0, 255))

        # Desenha as entidades
        for entity in map.get_entities():
            if entity.type == MapEntity.ENTITY_WALL:
                entity_bbox = entity.get_bounding_box()
                entity_corners = entity_bbox.get_corners()
                entity_scaled = [(x * scale_x, y * scale_y) for (x, y) in entity_corners]
                gfxdraw.aapolygon(surface, entity_scaled, (200, 0, 0))
                gfxdraw.filled_polygon(surface, entity_scaled, (200, 0, 0))

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
    frames: List[List[List[int]]],
    output_path: Union[str, Path],
    fps: int = 24,
) -> None:
    """
    Save a list of RGB frames as an MP4 video.

    Args:
        frames: List of frames; each frame is [H][W][3] with 0-255 ints
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

    if not frames:
        raise ValueError("Frames list is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure frames are uint8 and have correct shape
    processed_frames = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            # Convert grayscale to RGB
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            # Convert single channel to RGB
            frame = np.repeat(frame, 3, axis=2)
        processed_frames.append(frame)

    imageio.mimwrite(str(output_path), processed_frames, fps=fps, codec="libx264", quality=8)

    print(f"Saved {len(frames)} frames as MP4 to {output_path}")


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


def save_as_png(map: Map, path: str = "map.png", img_size: tuple[int, int] = None) -> None:
    rgb_array = to_rgb_array(map, img_size)
    save_rgb_array_as_png(rgb_array, path)