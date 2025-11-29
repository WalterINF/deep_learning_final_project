from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


@dataclass(slots=True)
class KinematicModel:
    """Vectorised RK4 integrator for a tractor-trailer bicycle model.

    The state is ``[x, y, theta_truck, beta]`` where ``x``/``y`` locate the
    rear axle of the truck in meters, ``theta_truck`` is its heading in radians
    and ``beta`` the articulation angle between truck and trailer.
    """

    geometry: dict
    dt: float = 0.2
    _wheelbase: float = field(init=False)
    _trailer_length: float = field(init=False)
    _hitch_offset: float = field(init=False)
    _inv_wheelbase: float = field(init=False)
    _inv_trailer_length: float = field(init=False)
    _offset_factor: float = field(init=False)

    def __post_init__(self) -> None:
        self._wheelbase = self._geom_value(
            "wheelbase",
            "wheelbase_tractor",
            "comprimento_entre_eixos_trator",
            "D",
            default=6.0,
        )
        self._trailer_length = self._geom_value("trailer_length", "L", default=12.0)
        self._hitch_offset = self._geom_value(
            "hitch_offset",
            "trailer_hitch_offset",
            "distancia_eixo_traseiro_quinta_roda",
            "quinta_roda",
            default=1.0,
        )
        self._inv_wheelbase = 1.0 / self._wheelbase if self._wheelbase != 0.0 else 0.0
        self._inv_trailer_length = 1.0 / self._trailer_length if self._trailer_length != 0.0 else 0.0
        self._offset_factor = (
            self._hitch_offset * self._inv_wheelbase * self._inv_trailer_length
            if self._hitch_offset > 0.0
            else 0.0
        )

    # ------------------------------------------------------------------
    def _geom_value(self, *keys: str, default: float) -> float:
        source = self.geometry
        if isinstance(source, dict):
            for key in keys:
                if key in source:
                    return float(source[key])
        else:
            for key in keys:
                if hasattr(source, key):
                    return float(getattr(source, key))
        return float(default)

    # ------------------------------------------------------------------
    def step(self, state: np.ndarray, control: Tuple[float, float]) -> np.ndarray:
        state = state.astype(np.float64, copy=False)
        v, alpha = float(control[0]), float(control[1])

        k1 = self._derivatives(state, v, alpha)
        k2 = self._derivatives(state + 0.5 * self.dt * k1, v, alpha)
        k3 = self._derivatives(state + 0.5 * self.dt * k2, v, alpha)
        k4 = self._derivatives(state + self.dt * k3, v, alpha)

        new_state = state + (self.dt / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)
        new_state[2] = self._normalize_angle(new_state[2])
        new_state[3] = self._normalize_angle(new_state[3])
        return new_state.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    def _derivatives(self, state: np.ndarray, v: float, alpha: float) -> np.ndarray:
        theta = float(state[2])
        beta = float(state[3])

        tan_alpha = math.tan(alpha)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        dx = v * cos_theta
        dy = v * sin_theta
        dtheta = v * self._inv_wheelbase * tan_alpha
        dbeta = (
            -v * self._inv_trailer_length * math.sin(beta)
            - v * self._inv_wheelbase * tan_alpha
            + v * self._offset_factor * tan_alpha * math.cos(beta)
        )
        return np.array([dx, dy, dtheta, dbeta], dtype=np.float64)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
        return wrapped
