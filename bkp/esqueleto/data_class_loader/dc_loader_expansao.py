from dataclasses import dataclass
from pathlib import Path
import json
from typing import Union, List, Dict

@dataclass(frozen=True)
class ParametrosExpansao:
    nome: str
    delta_dir: float
    limite_angulo_direcao: float
    divisores: Union[int, List[int]]
    limite_velocidade: float
    dt: float
    sentidos: Union[int, List[int]]


# Loader que lê o JSON e expõe as definições por nome
class ListaDeParametrosExpansao:
    def __init__(self, config_path: Path):
        try:
            config_path = Path(config_path)
        except TypeError:
            raise ValueError("O caminho do arquivo de configuração da expansão deve ser uma string ou um objeto Path.")

        data = json.loads(config_path.read_text(encoding="utf-8"))

        self._defs: Dict[str, ParametrosExpansao] = {}
        for entry in data["parametros_expansao"]:
            params = entry["parametros"]

            # --- Normalização ---
            sentidos = params.get("sentidos", [1, -1])
            if not isinstance(sentidos, (list, tuple)):
                sentidos = [sentidos]
            sentidos = list(sentidos)

            # --- Validação ---
            for s in sentidos:
                if s not in (1, -1):
                    raise ValueError(
                        f"Valor inválido em 'sentidos' para '{entry['nome']}': {s}. "
                        f"Permitidos: 1, -1 ou [1, -1]."
                    )

            params["sentidos"] = sentidos

            # --- Cria objeto dataclass ---
            self._defs[entry["nome"]] = ParametrosExpansao(nome=entry["nome"], **params)

    def get(self, nome: str) -> ParametrosExpansao:
        try:
            return self._defs[nome]
        except KeyError:
            raise ValueError(
                f"Parâmetros de expansão '{nome}' não encontrados em lista_parametros_expansao.json"
            )