from typing import Iterable, List, Union
from pathlib import Path
import numpy as np
from PIL import Image
import pygame
from Simulation import Map
import math
from pygame import gfxdraw
from Simulation import MapEntity
from Simulation import ArticulatedVehicle
from pygame.font import Font

# Initialize pygame.font before creating Font objects
pygame.font.init()

color_and_fill_mappings = {
    MapEntity.ENTITY_WALL: ((200, 0, 0), True), ## vermelho
    MapEntity.ENTITY_OBSTACLE: ((0, 0, 0), True), ## preto
    MapEntity.ENTITY_PARKING_SLOT: ((0, 0, 0), True), ## preto
    MapEntity.ENTITY_PARKING_GOAL: ((0, 200, 0), True), ## verde
}

font = Font(None, 20)


def to_rgb_array(
    map: Map,
    vehicle: ArticulatedVehicle | None = None,
    img_size: tuple[int, int] = None,
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

    # Define dimensões de saída da imagem
    if img_size is None:
        width = int(map.get_size()[0])
        height = int(map.get_size()[1])
    else:
        width = max(map.get_size()[0], int(img_size[0]))
        height = max(map.get_size()[1], int(img_size[1]))

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
        color, fill = color_and_fill_mappings[entity.type]
        bbox = entity.get_bounding_box()
        corners = bbox.get_corners()  # [(x,y), ...] em coords globais
        # Converte coordenadas do espaço do mapa para pixels de saída
        scaled_corners = [(x * scale_x, y * scale_y) for (x, y) in corners]
        if fill:
            gfxdraw.filled_polygon(surface, scaled_corners, color)
        else:
            gfxdraw.aapolygon(surface, scaled_corners, color)

    # Desenha o veículo
    if vehicle is not None:
        # Tractor
        tractor_bbox = vehicle.get_bounding_box_tractor()
        tractor_corners = tractor_bbox.get_corners()
        tractor_scaled = [(x * scale_x, y * scale_y) for (x, y) in tractor_corners]
        gfxdraw.aapolygon(surface, tractor_scaled, (0, 0, 255))

        # Desenha as rodas do trator
        wheels_bboxes = vehicle.get_wheels_bounding_boxes()
        for wheel_bbox in wheels_bboxes:
            wheel_corners = wheel_bbox.get_corners()
            wheel_scaled = [(x * scale_x, y * scale_y) for (x, y) in wheel_corners]
            gfxdraw.aapolygon(surface, wheel_scaled, (0, 0, 0))
            gfxdraw.filled_polygon(surface, wheel_scaled, (0, 0, 0))

        # Trailer
        trailer_bbox = vehicle.get_bounding_box_trailer()
        trailer_corners = trailer_bbox.get_corners()
        trailer_scaled = [(x * scale_x, y * scale_y) for (x, y) in trailer_corners]
        gfxdraw.aapolygon(surface, trailer_scaled, (0, 200, 0))

        # Quinta roda do trator (círculo)
        axle_pos = (vehicle.position_x_trator * scale_x, vehicle.position_y_trator * scale_y)
        axle_radius = 0.5 * scale_x
        gfxdraw.filled_circle(surface, int(axle_pos[0]), int(axle_pos[1]), int(axle_radius), (200, 0, 0))

        # Desenha os raycasts com círculos nas pontas
        for _, raycast in vehicle.raycasts.items():
            raycast_position = (raycast.origin_x * scale_x, raycast.origin_y * scale_y)
            raycast_angle = raycast.theta
            raycast_length = raycast.length * scale_x
            line_end_position = (raycast_position[0] + raycast_length * math.cos(raycast_angle), raycast_position[1] + raycast_length * math.sin(raycast_angle))
            gfxdraw.line(surface, int(raycast_position[0]), int(raycast_position[1]), int(line_end_position[0]), int(line_end_position[1]), (0, 0, 255))
            gfxdraw.filled_circle(surface, int(line_end_position[0]), int(line_end_position[1]), int(0.5 * scale_x), (0, 0, 255))

    # desenha ações na tela
    if vehicle.get_velocity() is not None:
        velocity_text = f"Velocity: {vehicle.get_velocity():.2f}"
        velocity_surface = font.render(velocity_text, True, (0, 0, 0))
        surface.blit(velocity_surface, (10, 10))

    if vehicle.get_alpha() is not None:
        alpha_text = f"Alpha: {vehicle.get_alpha():.2f}"
        alpha_surface = font.render(alpha_text, True, (0, 0, 0))
        surface.blit(alpha_surface, (10, 30))

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