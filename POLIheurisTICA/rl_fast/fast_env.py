from __future__ import annotations

import math
from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .fast_sim import FastSimulation, SimulationResult


class FastParkingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.sim = FastSimulation(seed=seed)

        self.observation_space = spaces.Box(
            low=np.array(
                [-50.0, -20.0, -math.pi, -math.pi, 0.0, -self.sim.limits.reverse_limit, -math.pi]
                + [0.0] * 14,
                dtype=np.float32,
            ),
            high=np.array(
                [50.0, 20.0, math.pi, math.pi, 200.0, self.sim.limits.speed_limit, math.pi]
                + [1.0] * 14,
                dtype=np.float32,
            ),
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.max_steps = 800
        self.step_count = 0
        self.last_dijkstra = 0.0
        self.last_gear_sign = 0
        self.dist_since_shift = 0.0
        self.shift_count = 0

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if hasattr(self, "np_random") and self.np_random is not None:
            reseed_value = int(self.np_random.integers(0, 2**32 - 1))
        else:
            reseed_value = seed
        self.sim.reseed(reseed_value)
        self.step_count = 0
        self.last_gear_sign = 0
        self.dist_since_shift = 0.0
        self.shift_count = 0

        sim_result = self.sim.reset()
        self.last_dijkstra = sim_result.nav_distance

        observation = self._build_observation(sim_result)
        info = {
            "case": {
                "labels": {
                    "start": self.sim.current_case.start if self.sim.current_case else None,
                    "goal": self.sim.current_case.goal if self.sim.current_case else None,
                },
                "start_pose": self.sim.state[:2].tolist(),
                "goal_pose": self.sim.goal.center.tolist() if self.sim.goal else None,
            },
            "sim_result": sim_result,
        }
        return observation, info

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        control = self._action_to_control(action)
        sim_result = self.sim.step(control)
        observation = self._build_observation(sim_result)

        penalty_gear = self._update_gear_hysteresis(sim_result)

        reward, terminated, truncated = self._calculate_reward(sim_result, penalty_gear)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        info = {"sim_result": sim_result}
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _action_to_control(self, action: np.ndarray) -> Tuple[float, float]:
        v_norm = float(np.clip(action[0], -1.0, 1.0))
        steer_norm = float(np.clip(action[1], -1.0, 1.0))

        if v_norm >= 0.0:
            velocity = v_norm * self.sim.limits.speed_limit
        else:
            velocity = v_norm * self.sim.limits.reverse_limit
        steer = steer_norm * self.sim.limits.steer_limit
        return velocity, steer

    def _build_observation(self, sim_result: SimulationResult) -> np.ndarray:
        errors = sim_result.errors
        obs = np.array(
            [
                np.clip(errors["parallel"], -50.0, 50.0),
                np.clip(errors["perpendicular"], -20.0, 20.0),
                np.clip(errors["theta1"], -math.pi, math.pi),
                np.clip(errors["theta2"], -math.pi, math.pi),
                np.clip(sim_result.nav_distance, 0.0, 200.0),
                np.clip(sim_result.speed, -self.sim.limits.reverse_limit, self.sim.limits.speed_limit),
                np.clip(sim_result.state[3], -math.pi, math.pi),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate((obs, sim_result.raycasts.astype(np.float32, copy=False)))
        return obs

    def _update_gear_hysteresis(self, sim_result: SimulationResult) -> float:
        velocity = sim_result.speed
        current_sign = self.last_gear_sign
        if abs(velocity) > 0.01:
            current_sign = 1 if velocity > 0 else -1
        penalty = 0.0

        if current_sign != self.last_gear_sign and current_sign != 0:
            if self.dist_since_shift < 60.0:
                self.shift_count += 1
                if self.shift_count > 1:
                    penalty = -1.0
            else:
                self.shift_count = 1
            self.dist_since_shift = 0.0
            self.last_gear_sign = current_sign
        else:
            self.dist_since_shift += sim_result.step_distance
        return penalty

    def _calculate_reward(self, sim_result: SimulationResult, penalty_gear: float):
        errors = sim_result.errors
        beta = self.sim.state[3]

        if sim_result.collision or sim_result.jackknife or sim_result.overshoot:
            return -100.0, True, False

        if self._is_success(sim_result):
            return 100.0, True, False

        curr_dijkstra = sim_result.nav_distance
        delta_dijkstra = (self.last_dijkstra - curr_dijkstra) * 10.0
        self.last_dijkstra = curr_dijkstra

        align = 0.5 * (1.0 - abs(errors["theta2"]) / math.pi) + 0.5 * (1.0 - abs(errors["perpendicular"]) / (2.0 * self.sim.assets.half_slot_width))
        time_penalty = -0.01
        beta_penalty = -0.02 if abs(beta) > math.radians(25.0) else 0.0
        nav_penalty = -0.001 * abs(sim_result.speed)

        total = delta_dijkstra + align + time_penalty + beta_penalty + nav_penalty + penalty_gear
        return total, False, False

    def _is_success(self, sim_result: SimulationResult) -> bool:
        errors = sim_result.errors
        return (
            abs(errors["parallel"]) < 0.2
            and abs(errors["perpendicular"]) < 0.2
            and abs(errors["theta2"]) < 0.1
            and abs(sim_result.speed) < 0.1
        )
