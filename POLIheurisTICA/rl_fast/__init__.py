"""Lightweight reinforcement learning sidecar for articulated parking.

This package provides a NumPy-first simulation and a Gymnasium-compatible
environment that leverage the legacy geometry and configuration loaders in
``Inspiracao/src`` while decoupling the control loop for Sac training.

Public exports:
    - FastParkingEnv: Gymnasium environment.
    - FastSimulation: Vectorised physics + collision manager.
    - KinematicModel: RK4-integrated kinematic bicycle with articulation.
"""
from __future__ import annotations

from .fast_env import FastParkingEnv
from .fast_sim import FastSimulation
from .kin_model import KinematicModel

__all__ = [
    "FastParkingEnv",
    "FastSimulation",
    "KinematicModel",
]
