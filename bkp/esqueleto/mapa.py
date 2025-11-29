
import numpy as np
import math
from dominio.data_class_loader.dc_loader_mapa import ListaDeMapas
# 3) Entidade de domínio com Singleton por definição
class Mapa():
    _instances: dict[str, "Mapa"] = {}

    def __new__(cls, definition):
        """
        Garante que cada mapa seja uma instância única por nome.
            Args:
                definition (MapaDefinition): Definição do mapa a ser carregado.
            Returns:
                Mapa: Instância única do mapa.
            """
        # Garante instância única para cada mapa (por nome)
        if definition.nome not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[definition.nome] = instance
        return cls._instances[definition.nome]

    def __init__(self, definition: ListaDeMapas):
        # Protege contra múltiplas inicializações
        if getattr(self, "_initialized", False):
            return

        # Atributos da definição
        self.nome            = definition.nome
        self.path_file       = definition.path_file
        self.resolucao       = definition.resolucao
        self.ponto_de_referencia_global        = definition.ponto_de_referencia_global 
        self.fonte_de_coleta = definition.fonte_de_coleta

        # Carregamento pesado do mapa
        self._matriz = np.load(self.path_file)
        # Flipar em y
        # self._matriz = np.flipud(self._matriz)
        self.ALTURA  = int(self._matriz.shape[0])
        self.LARGURA = int(self._matriz.shape[1])
        self._initialized = True

    def obterMapa(self) -> np.ndarray:
        return self._matriz

    def linhaDeVisao(self, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        """
        Verifica se há linha de visão entre dois pontos no mapa.
        """
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
        err = dx - dy

        while True:
            if not (0 <= x1 < self.LARGURA and 0 <= y1 < self.ALTURA):
                return False
            if self._matriz[y1, x1] == 0:
                return False
            if x1 == x2 and y1 == y2:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def coordenadaGlobalParaPixel(self, ponto_global: tuple[float, float]) -> tuple[int, int]:
        """
        Converte coordenadas do mundo real (em metros) para coordenadas de pixel no mapa.
        """
        x_m, y_m = ponto_global
        ref_x, ref_y = self.ponto_de_referencia_global
        x_px = int(round(ref_x + x_m * self.resolucao))
        y_px = int(round(ref_y - y_m * self.resolucao))  # y invertido se origem está no topo da imagem
        return x_px, y_px

    def checarColisaoComObstaculos(self, vertices_mundo: list[tuple[float, float]]) -> bool:
        """
        Verifica colisão com obstáculos, recebendo os vértices em coordenadas do mundo real (metros).
        Converte para pixel antes da checagem.
        """
        if not vertices_mundo:
            return False

        # Converte todos os vértices do veículo para pixels
        vertices_px = [self.coordenadaGlobalParaPixel(v) for v in vertices_mundo]

        for i, (x, y) in enumerate(vertices_px):
            if not (0 <= x < self.LARGURA and 0 <= y < self.ALTURA):
                return True  # Fora do mapa → colisão
            if self._matriz[y, x] == 0:
                return True  # Obstáculo detectado
            if i > 0:
                if not self.linhaDeVisao(vertices_px[i - 1], (x, y)):
                    return True
        return False

    def conectarAoCentroMaisProximo(
        self,
        ponto: tuple[float, float],
        centros: list[tuple[int, int]]
    ) -> tuple[tuple[int, int] | None, float]:
        """
        Conecta um ponto ao centro mais próximo, retornando o centro e a distância.
        Args:
            ponto (tuple): Ponto a ser conectado, como (x, y).
            centros (list): Lista de centros disponíveis, cada um como (x, y).
        Returns:
            tuple: Tupla contendo o centro mais próximo e a distância até ele.
        """
        if not centros:
            return None, float('inf')
        px, py = float(ponto[0]), float(ponto[1])
        best, mind = None, float('inf')
        for cx, cy in centros:
            d = math.hypot(cx - px, cy - py)
            if d < mind:
                best, mind = (cx, cy), d
        return best, mind

    def obterCentrosLivres(self, intervalo: int = 35) -> set[tuple[int, int]]:
        """
        Obtém os centros livres do mapa, considerando um intervalo específico.
        Args:
            intervalo (int): Intervalo para considerar os centros livres.
        Returns:
            set: Conjunto de coordenadas (x, y) dos centros livres.
        """
        copia = np.copy(self._matriz)
        half = intervalo // 2
        if half <= 0:
            return set()
        livres = set()
        for y in range(half, self.ALTURA - half, intervalo):
            for x in range(half, self.LARGURA - half, intervalo):
                y0, y1 = y - half, y + half + (intervalo % 2)
                x0, x1 = x - half, x + half + (intervalo % 2)
                bloco = copia[y0:y1, x0:x1]
                if bloco.shape == (intervalo, intervalo) and np.all(bloco == 1):
                    livres.add((x, y))
        return livres



