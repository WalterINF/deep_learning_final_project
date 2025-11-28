from dataclasses import dataclass
from pathlib import Path
import json

# 1) Data‑Classes para tolerâncias
@dataclass(frozen=True)
class ToleranciasConvergencia:
    X: float
    Y: float
    THETA: float
    THETA2: float

@dataclass(frozen=True)
class ToleranciasParaIgualidade:
    dx: float
    dy: float
    dtheta: float
    dtheta2: float

@dataclass(frozen=True)
class ToleranciaEntreAngulos:
    d_ang: float

@dataclass(frozen=True)
class Tolerancias:
    nome: str
    convergencia: ToleranciasConvergencia
    igualidade: ToleranciasParaIgualidade
    entre_angulos: ToleranciaEntreAngulos

# 2) Loader centralizado
class ListaDeTolerancias:
    def __init__(self, config_path: str):
        try:
            config_path = Path(config_path)
        except TypeError:
            raise ValueError("O caminho do arquivo de configuração deve ser uma string ou um objeto Path.")
        data = json.loads(config_path.read_text(encoding='utf-8'))
        self._defs: dict[str, Tolerancias] = {}
        for entry in data["tolerancias"]:
            nome = entry["nome"]
            tc = ToleranciasConvergencia(**entry["tolerancias_convergencia"])
            ti = ToleranciasParaIgualidade(**entry["tolerancias_para_igualidade"])
            te = ToleranciaEntreAngulos(**entry["tolerancia_entre_angulos"])
            self._defs[nome] = Tolerancias(nome, tc, ti, te)

    def get(self, nome: str) -> Tolerancias:
        try:
            return self._defs[nome]
        except KeyError:
            raise ValueError(f"Tolerâncias '{nome}' não encontradas em lista_tolerancias.json")
