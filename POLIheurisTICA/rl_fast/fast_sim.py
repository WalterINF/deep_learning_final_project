from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .config import ParkingTarget, SimulationAssets, VehicleGeometry
from dominio.data_class_loader.dc_loader_cases_e_poses import CasePair  # type: ignore
from .kin_model import KinematicModel

from dominio.entidades.estados import Estado5  # type: ignore


@dataclass(slots=True)
class ControlLimits:
    speed_limit: float = 1.5  # m/s forward
    reverse_limit: float = 1.2  # m/s reverse
    steer_limit: float = math.radians(35.0)

    def clip(self, velocity: float, steer: float) -> Tuple[float, float]:
        v = float(np.clip(velocity, -self.reverse_limit, self.speed_limit))
        s = float(np.clip(steer, -self.steer_limit, self.steer_limit))
        return v, s


@dataclass(slots=True)
class SimulationResult:
    state: np.ndarray
    theta2: float
    errors: Dict[str, float]
    nav_distance: float
    collision: bool
    overshoot: bool
    jackknife: bool
    raycasts: np.ndarray
    step_distance: float
    speed: float
    control: Tuple[float, float]
    polygons: Dict[str, np.ndarray]


class FastSimulation:
    """High-throughput kinematic simulator for articulated parking."""

    def __init__(
        self,
        assets: SimulationAssets | None = None,
        limits: ControlLimits | None = None,
        seed: int | None = None,
    ) -> None:
        self.assets = assets or SimulationAssets()
        self.geometry = self.assets.geometry
        self.limits = limits or ControlLimits()
        self.dt = self.assets.dt

        self.model = KinematicModel(
            geometry={
                "wheelbase_tractor": self.geometry.wheelbase_tractor,
                "trailer_length": self.geometry.trailer_length,
                "trailer_hitch_offset": self.geometry.trailer_hitch_offset,
            },
            dt=self.dt,
        )

        self.state = np.zeros(4, dtype=np.float32)
        self.goal: ParkingTarget | None = None
        self.case_pairs = tuple(self.assets.iter_case_pairs())
        if not self.case_pairs:
            raise RuntimeError("Nenhum par start/goal encontrado nas configurações legadas.")
        self.current_case: CasePair | None = None

        self.map = self.assets.map
        self.vehicle = self.assets.vehicle_params
        self.px_per_meter = self.assets.px_per_meter
        self._map_matrix = self.assets.map_matrix
        self._map_height, self._map_width = self._map_matrix.shape
        self._rng = np.random.default_rng(seed)

        self.raycast_angles = np.deg2rad(np.linspace(-110.0, 110.0, 14, dtype=np.float32))
        self.sensor_range = 25.0  # metros
        self._sensor_samples = max(4, int(self.sensor_range * self.px_per_meter // 2))

        self.max_beta = float(self.geometry.max_beta)
        self.overshoot_margin = 0.5 * self.geometry.trailer_length
        self.perp_tolerance = max(0.2, self.assets.half_slot_width * 0.5)

    # ------------------------------------------------------------------
    def reset(self, case_index: int | None = None) -> SimulationResult:
        pair = self._select_case(case_index)
        self.current_case = pair
        start_pose = self.assets.get_pose(pair.start)
        self.goal = self.assets.make_target(pair.goal)

        beta = float(start_pose.beta or 0.0)
        self.state = np.array([start_pose.x1, start_pose.y1, start_pose.theta1, beta], dtype=np.float32)

        return self._compose_result(self.state.copy(), self.state.copy(), (0.0, 0.0))

    # ------------------------------------------------------------------
    def step(self, action: Tuple[float, float]) -> SimulationResult:
        velocity, steer = self.limits.clip(action[0], action[1])
        old_state = self.state.copy()
        new_state = self.model.step(old_state, (velocity, steer))
        self.state = new_state
        return self._compose_result(old_state, new_state, (velocity, steer))

    # ------------------------------------------------------------------
    def _compose_result(
        self,
        prev_state: np.ndarray,
        new_state: np.ndarray,
        control: Tuple[float, float],
    ) -> SimulationResult:
        theta = float(new_state[2])
        beta = float(new_state[3])
        theta2 = self._wrap_angle(theta + beta)

        errors = self._compute_errors(new_state)
        nav_distance = errors["euclid"]
        estado = self._build_estado(new_state, theta2)
        polygons = self._vehicle_polygons(estado)
        collision = self._check_collision(polygons)
        overshoot = self._check_overshoot(errors)
        jackknife = abs(beta) > self.max_beta
        raycasts = self._get_raycasts(new_state)

        step_distance = float(np.linalg.norm(new_state[:2] - prev_state[:2]))
        return SimulationResult(
            state=new_state.astype(np.float32, copy=True),
            theta2=theta2,
            errors=errors,
            nav_distance=nav_distance,
            collision=collision,
            overshoot=overshoot,
            jackknife=jackknife,
            raycasts=raycasts,
            step_distance=step_distance,
            speed=float(control[0]),
            control=control,
            polygons=polygons,
        )

    # ------------------------------------------------------------------
    def _compute_errors(self, state: np.ndarray) -> Dict[str, float]:
        if self.goal is None:
            raise RuntimeError("Goal must be definido antes de calcular erros.")
        position = state[:2]
        diff = self.goal.center - position
        axis = np.array([math.cos(self.goal.heading), math.sin(self.goal.heading)], dtype=np.float32)
        perp = np.array([-axis[1], axis[0]], dtype=np.float32)
        e_parallel = float(diff @ axis)
        e_perp = float(diff @ perp)
        theta1 = float(state[2])
        beta = float(state[3])
        theta2 = self._wrap_angle(theta1 + beta)
        e_theta1 = self._wrap_angle(self.goal.heading - theta1)
        e_theta2 = self._wrap_angle(self.goal.trailer_heading - theta2)
        euclid = float(np.linalg.norm(diff))
        return {
            "parallel": e_parallel,
            "perpendicular": e_perp,
            "theta1": e_theta1,
            "theta2": e_theta2,
            "euclid": euclid,
        }

    def _check_collision(self, polygons: Dict[str, np.ndarray]) -> bool:
        for poly in polygons.values():
            if self.map.checarColisaoComObstaculos([tuple(pt) for pt in poly]):
                return True
        return False

    def _check_overshoot(self, errors: Dict[str, float]) -> bool:
        return errors["parallel"] < -self.overshoot_margin and abs(errors["perpendicular"]) < self.perp_tolerance

    # ------------------------------------------------------------------
    def _get_raycasts(self, state: np.ndarray) -> np.ndarray:
        if self.goal is None:
            return np.ones(14, dtype=np.float32)
        origin = self._sensor_origin(state)
        readings = np.ones(len(self.raycast_angles), dtype=np.float32)
        heading = float(state[2])
        step = self.sensor_range / self._sensor_samples
        for idx, offset in enumerate(self.raycast_angles):
            direction = np.array([math.cos(heading + offset), math.sin(heading + offset)], dtype=float)
            distance = self.sensor_range
            for sample_idx in range(1, self._sensor_samples + 1):
                d = sample_idx * step
                point = origin + direction * d
                pixel = self.map.coordenadaGlobalParaPixel((float(point[0]), float(point[1])))
                if not self._pixel_is_free(pixel):
                    distance = d
                    break
            readings[idx] = distance / self.sensor_range
        return readings.astype(np.float32, copy=False)

    def _sensor_origin(self, state: np.ndarray) -> np.ndarray:
        theta = float(state[2])
        offset = self.geometry.tractor_front_overhang
        forward = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        return state[:2] + forward * offset

    def _pixel_is_free(self, pixel: Tuple[int, int]) -> bool:
        x, y = pixel
        if x < 0 or y < 0 or x >= self._map_width or y >= self._map_height:
            return False
        return bool(self._map_matrix[y, x])

    def _select_case(self, index: int | None) -> CasePair:
        if index is None:
            pair = self.case_pairs[self._rng.integers(0, len(self.case_pairs))]
        else:
            pair = self.case_pairs[index % len(self.case_pairs)]
        return pair

    def reseed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_estado(state: np.ndarray, theta2: float) -> Estado5:
        return Estado5(float(state[0]), float(state[1]), float(state[2]), float(state[3]), theta2)

    def _vehicle_polygons(self, estado: Estado5) -> Dict[str, np.ndarray]:
        raw_vertices = self.vehicle.calcula_vertices_em_metros(estado)
        return {name: np.asarray(vertices, dtype=np.float32) for name, vertices in raw_vertices.items()}

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
