from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

# -- Legacy imports ---------------------------------------------------------
# These modules live under Inspiracao/src. We extend sys.path once so that RL
# workers spawned by SubprocVecEnv can locate them without modifying PYTHONPATH
# globally. The append is idempotent and safe across forked processes.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_SRC = _REPO_ROOT / "Inspiracao" / "src"
if str(_LEGACY_SRC) not in sys.path:
    sys.path.append(str(_LEGACY_SRC))

from dominio.data_class_loader.dc_loader_cases_e_poses import (  # type: ignore
    CaseGroup,
    CasePair,
    CasesLoader,
    Pose,
    PosesDefinition,
    PosesLoader,
)
from dominio.data_class_loader.dc_loader_configuracao import (  # type: ignore
    Configuracao,
    DCLoaderConfiguracao,
)
from dominio.data_class_loader.dc_loader_mapa import ListaDeMapas, MapaDefinition  # type: ignore
from dominio.data_class_loader.dc_loader_parametros_veiculo import (  # type: ignore
    ListaDeVeiculos,
    ParametrosGeometricosVeiculo,
)
from dominio.mapa import Mapa  # type: ignore
from dominio.parametros_trator_trailer import ParametrosTratorTrailer  # type: ignore

from .procedural_map import generate_default_map


@dataclass(slots=True)
class VehicleGeometry:
    """Compact view of the tractor+trailer geometry used by the fast simulator."""

    wheelbase_tractor: float
    trailer_hitch_offset: float
    tractor_front_overhang: float
    tractor_length: float
    tractor_width: float
    trailer_length: float
    trailer_front_overhang: float
    trailer_width: float
    max_beta: float

    @property
    def tractor_half_width(self) -> float:
        return 0.5 * self.tractor_width

    @property
    def trailer_half_width(self) -> float:
        return 0.5 * self.trailer_width


@dataclass(slots=True)
class ParkingTarget:
    center: np.ndarray  # (x, y)
    heading: float      # truck heading at goal
    articulation: float # beta goal (usually 0)

    @property
    def trailer_heading(self) -> float:
        return self.heading + self.articulation


@dataclass(slots=True)
class SimulationAssets:
    """Loads legacy configuration, map and vehicle definitions once."""

    config_name: str = "configuracao_NOV_25-dt01"
    base_path: Path = field(default_factory=lambda: _REPO_ROOT)
    dt: float = 0.2

    _legacy_root: Path = field(init=False)
    _paths: Dict[str, str | Dict[str, str]] = field(init=False)
    configuration: Configuracao = field(init=False)
    vehicles: ListaDeVeiculos = field(init=False)
    vehicle_params: ParametrosTratorTrailer = field(init=False)
    geometry: VehicleGeometry = field(init=False)
    map_definition: MapaDefinition = field(init=False)
    map: Mapa = field(init=False)
    map_matrix: np.ndarray = field(init=False)
    px_per_meter: float = field(init=False)
    map_reference: np.ndarray = field(init=False)
    poses: PosesDefinition = field(init=False)
    cases: CaseGroup = field(init=False)

    def __post_init__(self) -> None:
        self._legacy_root = self.base_path / "Inspiracao"
        # Load master path list to honour alternative directory layouts.
        self._paths = self._load_path_manifest()

        self.configuration = self._load_configuration()

        vehicles_path = self._get_path("veiculo")
        self.vehicles = ListaDeVeiculos(str(vehicles_path))
        vehicle_definition = self.vehicles.get(self.configuration.veiculo)
        self.vehicle_params = ParametrosTratorTrailer(parametros=vehicle_definition)

        self.geometry = self._build_geometry(vehicle_definition)

        self.map_definition = self._load_map_definition(self.configuration.mapa)
        self.map = Mapa(self.map_definition)
        self.map_matrix = self.map.obterMapa()
        self.px_per_meter = float(self.map_definition.resolucao)
        self.map_reference = np.asarray(self.map_definition.ponto_de_referencia_global, dtype=float)

        self.poses = self._load_poses(self.configuration.poses)
        self.cases = self._load_cases(self.configuration.cases)

    # ------------------------------------------------------------------
    def _load_path_manifest(self) -> Dict[str, str | Dict[str, str]]:
        manifest_path = self._legacy_root / "config" / "lista_caminhos.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = data["caminhos"][0]["caminhos_das_listas"]
        return entry

    def _resolve_path(self, relative: str) -> Path:
        # Accept nested dictionaries (e.g. historico). For RL we only request files.
        candidate = Path(relative)
        if candidate.is_absolute():
            return candidate
        return self._legacy_root / relative

    def _get_path(self, key: str) -> Path:
        value = self._paths.get(key)
        if not isinstance(value, str):
            raise ValueError(f"Manifest entry '{key}' must be a string path, found {type(value)!r}")
        return self._resolve_path(value)

    def _load_configuration(self) -> Configuracao:
        config_path = self._get_path("configuracao")
        loader = DCLoaderConfiguracao(str(config_path))
        return loader.get(self.config_name)

    def _load_map_definition(self, nome: str) -> MapaDefinition:
        mapa_path = self._get_path("mapa")
        loader = ListaDeMapas(str(mapa_path))
        definition = loader.get(nome)
        resolved = self._resolve_map_path(definition.path_file)
        ensured = self._ensure_map_file(resolved, definition)
        return replace(definition, path_file=str(ensured))

    def _load_poses(self, nome: str) -> PosesDefinition:
        poses_path = self._get_path("poses")
        loader = PosesLoader(str(poses_path))
        return loader.get(nome)

    def _load_cases(self, identificador: str) -> CaseGroup:
        cases_path = self._get_path("cases")
        loader = CasesLoader(str(cases_path))
        return loader.get(identificador)

    def _resolve_map_path(self, raw: str) -> Path:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate

        search_roots = (
            self._legacy_root,
            self.base_path,
            self.base_path / "Inspiracao",
        )

        for root in search_roots:
            candidate_path = root / candidate
            if candidate_path.exists():
                return candidate_path

        if candidate.name:
            matches = list(self.base_path.glob(f"**/{candidate.name}"))
            if matches:
                return matches[0]

        # Default to the legacy root path to keep relative structure for fallbacks
        return self._legacy_root / candidate

    def _ensure_map_file(self, path: Path, definition: MapaDefinition) -> Path:
        if path.exists():
            return path

        fallback_dir = self.base_path / "rl_fast" / "generated_assets"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        safe_name = definition.nome.replace(" ", "_") or "mapa"
        fallback_path = fallback_dir / f"{safe_name}_placeholder.npy"

        if not fallback_path.exists():
            width, height = self._placeholder_dimensions(definition)
            # The procedural generator expects physical dimensions and resolution.
            resolution = float(definition.resolucao or 1.0)
            if resolution < 1.0:
                resolution = 1.0
            generate_default_map(
                fallback_path,
                width=float(width),
                height=float(height),
                resolution=resolution,
            )
        warnings.warn(
            (
                f"Mapa '{definition.nome}' não encontrado em '{path}'. "
                f"Usando placeholder sintético em '{fallback_path}'."
            ),
            RuntimeWarning,
        )
        return fallback_path

    @staticmethod
    def _placeholder_dimensions(definition: MapaDefinition) -> tuple[float, float]:
        # Use conservative physical dimensions when the original map is unknown.
        width = max(150.0, float(abs(definition.ponto_de_referencia_global[0]) * 0.1 + 150.0))
        height = max(150.0, float(abs(definition.ponto_de_referencia_global[1]) * 0.1 + 150.0))
        return width, height

    @staticmethod
    def _build_geometry(params: ParametrosGeometricosVeiculo) -> VehicleGeometry:
        return VehicleGeometry(
            wheelbase_tractor=params.distancia_eixo_traseiro_quinta_roda
            + params.distancia_eixo_dianteiro_quinta_roda,
            trailer_hitch_offset=params.distancia_eixo_traseiro_quinta_roda,
            tractor_front_overhang=params.distancia_frente_quinta_roda,
            tractor_length=params.comprimento_trator,
            tractor_width=params.largura_trator,
            trailer_length=params.comprimento_trailer,
            trailer_front_overhang=params.distancia_frente_trailer_quinta_roda,
            trailer_width=params.largura_trailer,
            max_beta=float(params.angulo_maximo_articulacao),
        )

    # ------------------------------------------------------------------
    def iter_case_pairs(self) -> Iterable[CasePair]:
        for definition in self.cases.all_cases:
            for pair in definition.cases:
                yield pair

    def get_pose(self, label: str) -> Pose:
        return self.poses.poses[label]

    def make_target(self, label: str) -> ParkingTarget:
        pose = self.get_pose(label)
        center = np.array([pose.x1, pose.y1], dtype=float)
        return ParkingTarget(center=center, heading=float(pose.theta1), articulation=float(pose.beta or 0.0))

    # Accessors -----------------------------------------------------------
    @property
    def vehicle_length(self) -> float:
        return self.geometry.tractor_length + self.geometry.trailer_length

    @property
    def half_slot_width(self) -> float:
        # Conservative default for reward normalisation (2.0 m wide slot)
        return max(self.geometry.trailer_half_width, 2.0)
