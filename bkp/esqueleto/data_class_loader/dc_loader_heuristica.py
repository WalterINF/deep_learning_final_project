# dc_loader_heuristica.py
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Union

# 1) Data-Class para a heurística EUCLIDIANA (anteriormente ParametrosHeuristica)
@dataclass(frozen=True)
class ParametrosHeuristicaEuclidiana:
    nome: str
    peso_euclidiano: float
    peso_diagonal: float
    peso_theta: float
    peso_theta2: float
    threshold_distancia_theta2: float
    threshold_distancia_theta1: float
    threshold_theta: float
    
# 2) Data-Class para a nova heurística PERPENDICULAR
@dataclass(frozen=True)
class ParametrosHeuristicaPerpendicular:
    nome: str
    peso_coordenada_perpendicular: float
    peso_coordenada_nao_perpendicular: float
    peso_theta1: float
    peso_theta2: float
    threshold_distancia_paralela: float
    threshold_distancia_theta1: float
    threshold_distancia_theta2: float

# 3) Tipo de união para a type hint de retorno
#    Indica que o loader pode retornar qualquer um dos tipos de parâmetros
TipoParametrosHeuristica = Union[ParametrosHeuristicaEuclidiana, ParametrosHeuristicaPerpendicular]

# 4) Loader centralizado (agora uma Factory)
class ListaDeParametrosHeuristica:
    def __init__(self, config_path: str):
        try:
            config_path = Path(config_path)
        except TypeError:
            raise ValueError("O caminho do arquivo de configuração deve ser uma string ou um objeto Path.")
        
        data = json.loads(config_path.read_text(encoding='utf-8'))
        
        self._defs = {}
        # Garante que estamos iterando sobre a lista correta do JSON
        if "parametros_heuristica" not in data:
            raise ValueError("Chave 'parametros_heuristica' não encontrada no JSON.")

        for h in data["parametros_heuristica"]:
            nome_heuristica = h["nome"]
            parametros = h["parametros"]

            # Lógica da Factory: decide qual classe instanciar
            # verificando a presença de uma chave única de cada tipo.
            if "peso_euclidiano" in parametros:
                # É a heurística Euclidiana
                try:
                    self._defs[nome_heuristica] = ParametrosHeuristicaEuclidiana(
                        nome=nome_heuristica, **parametros
                    )
                except TypeError as e:
                    # Erro comum se faltar um parâmetro no JSON
                    raise ValueError(f"Erro ao carregar '{nome_heuristica}' (Euclidiana): {e}. Parâmetros recebidos: {parametros}")

            elif "peso_coordenada_perpendicular" in parametros:
                # É a nova heurística Perpendicular
                try:
                    self._defs[nome_heuristica] = ParametrosHeuristicaPerpendicular(
                        nome=nome_heuristica, **parametros
                    )
                except TypeError as e:
                    raise ValueError(f"Erro ao carregar '{nome_heuristica}' (Perpendicular): {e}. Parâmetros recebidos: {parametros}")
            
            else:
                # Tipo de heurística não reconhecido
                raise ValueError(f"Tipo de heurística não reconhecido para '{nome_heuristica}'. "
                                 "Não foi encontrado 'peso_euclidiano' ou 'peso_coordenada_perpendicular'.")

    def get(self, nome: str) -> TipoParametrosHeuristica: # Type hint atualizada
        """
        Retorna o objeto de dataclass de parâmetros correspondente ao nome.
        O objeto pode ser ParametrosHeuristicaEuclidiana ou ParametrosHeuristicaPerpendicular.
        """
        try:
            return self._defs[nome]
        except KeyError:
            raise ValueError(f"Heurística '{nome}' não definida em lista_parametros_heuristica.json")

# 5) Exemplo de uso atualizado
if __name__ == "__main__":
    # Supondo que o JSON esteja em 'config/lista_parametros_heuristica.json'
    # e que seu JSON esteja formatado corretamente (veja nota abaixo)
    try:
        loader = ListaDeParametrosHeuristica("config/lista_parametros_heuristica.json")
        
        # Teste 1: Carregar a Euclidiana
        params_euclidiana = loader.get("heuristica_euclidiana_05_thetas_07")
        print("--- Euclidiana ---")
        print(params_euclidiana)
        print(f"Tipo: {type(params_euclidiana)}")

        # Teste 2: Carregar a Perpendicular
        params_perp = loader.get("heuristica_coordenada_perpendicular")
        print("\n--- Perpendicular ---")
        print(params_perp)
        print(f"Tipo: {type(params_perp)}")

        # O código que instancia a heurística real agora precisa
        # verificar o tipo de 'params' para saber qual classe de heurística usar.
        # Exemplo:
        # if isinstance(params_perp, ParametrosHeuristicaPerpendicular):
        #     heuristica = HCoordenadaPerpendicular(poseInicial, poseFinal, params_perp)
        # elif isinstance(params_euclidiana, ParametrosHeuristicaEuclidiana):
        #     heuristica = HThetasEuclidiana(poseInicial, poseFinal, params_euclidiana)

    except FileNotFoundError:
        print("Erro: 'config/lista_parametros_heuristica.json' não encontrado.")
    except ValueError as e:
        print(f"Erro ao carregar heurística: {e}")