import json
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ParametrosGeometricosVeiculo:
    comprimento_trator: float
    distancia_eixo_traseiro_quinta_roda: float
    distancia_eixo_dianteiro_quinta_roda: float
    distancia_frente_quinta_roda: float
    distancia_entre_eixos_trator: float
    largura_trator: float
    comprimento_trailer: float
    distancia_eixo_traseiro_trailer_quinta_roda: float
    distancia_frente_trailer_quinta_roda: float
    largura_trailer: float
    largura_roda: float
    comprimento_roda: float
    angulo_maximo_articulacao: float


class ListaDeVeiculos:
    def __init__(self, json_path: str):
        if isinstance(json_path, str):
            json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._defs = {
            v["nome"]: ParametrosGeometricosVeiculo(**v["geometria_veiculo"])
            for v in data["veiculos"]
        }

    def get(self, nome: str) -> ParametrosGeometricosVeiculo:
        try:
            return self._defs[nome]
        except KeyError:
            raise ValueError(f"Veículo '{nome}' não encontrado em lista_veiculo.json")

if __name__ == "__main__":
    from dominio.parametros_trator_trailer import ParametrosTratorTrailer

    # Carregue o loader:
    loader = ListaDeVeiculos("config\lista_veiculos.json")

    # Obtenha os parâmetros do BUG1:
    parametros_bug1 = loader.get("BUG1")

    # Instancie o objeto de domínio:
    bug1 = ParametrosTratorTrailer(parametros=parametros_bug1)

    # Pronto para uso!
    print(bug1.get_parametros())
