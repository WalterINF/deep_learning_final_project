from interfaces.i_colisao import IColisao
class Colisao(IColisao):
    def __init__(self, mapa, angulo_maximo_articulacao: float, diferenca_entre_angulos):
        """
        Inicializa a classe de colisão com o mapa e parâmetros de ângulo.
        Args:
            mapa (object Mapa): Objeto que representa o mapa, deve ter um método checarColisaoComObstaculos.
            angulo_maximo_articulacao (float): Ângulo máximo permitido entre trator e reboque.
            diferenca_entre_angulos (object Angulos): Objeto responsável por realizar calculos entre ângulos.
        
        """
        self.mapa = mapa
        self.angulo_maximo_articulacao = angulo_maximo_articulacao
        self.calcular_beta = diferenca_entre_angulos
        
    def checarColisaoVeiculo(self, theta1:float, theta2:float) -> bool:
        """
        Verifica se o ângulo relativo entre trator e reboque excede o limite permitido.

        Args:
            theta1 (float): Ângulo do trator em radianos.
            theta2 (float): Ângulo do reboque em radianos.

        Returns:
            bool: True se houver colisão potencial (ângulo excede angulo_maximo_articulacaoimo), False caso contrário.
        """
        beta = self.calcular_beta(theta1, theta2)
        # print(f"theta1: {theta1}, theta2: {theta2}, beta: {beta}, angulo_maximo_articulacao: {self.angulo_maximo_articulacao}, saida: {abs(beta) > self.angulo_maximo_articulacao}")
        return abs(beta) > self.angulo_maximo_articulacao # Se beta for maior que o maximo, retorna True (colisão potencial)

    def checarColisaoComObstaculos(self, vertices: dict) -> bool:
        """
        Verifica se os vértices do veículo colidem com obstáculos no mapa.

        Args:
            vertices (dict): Dicionário contendo vértices do veículo, onde cada chave
                            mapeia para uma lista de pontos (x, y).

        Returns:
            bool: True se houver colisão com obstáculos, False caso contrário.
        """
        # Converte dicionário de vértices em uma lista plana de pontos
        vertices_lista = [vertex for key in vertices for vertex in vertices[key]]
        return self.mapa.checarColisaoComObstaculos(vertices_lista)
    
    def checar_colisao(self, vertices: dict, theta1: float, theta2:float) -> bool:
        """
        Verifica se há colisão com obstáculos ou entre trator e reboque.

        Args:
            vertices (dict): Dicionário com os vértices do veículo (trator e reboque).
            theta1 (float): Ângulo do trator em radianos.
            theta2 (float): Ângulo do reboque em radianos.

        Returns:
            bool: True se houver colisão (com obstáculos ou veículo), False caso contrário.
        """
        colisao_veiculo = self.checarColisaoVeiculo(theta1, theta2)
        colisao_mapa = self.checarColisaoComObstaculos(vertices)
        return colisao_veiculo or colisao_mapa
    


