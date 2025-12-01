from math import sin, cos
from .data_class_loader.dc_loader_parametros_veiculo import ParametrosGeometricosVeiculo
from .estados import Estado5, Estado3
## Single-Trailer

class ParametrosTratorTrailer():
    """ Parâmetros geométricos do conjunto trator + trailer.
    ---------------------------------- TRATOR ----------------------------------
    :param comprimento_trator:                [m] comprimento total do trator.
    :param distancia_eixo_traseiro_quinta_roda: [m] distância entre o eixo traseiro do trator e a quinta roda.
    :param distancia_eixo_dianteiro_quinta_roda: [m] distância entre o eixo dianteiro do trator e a quinta roda.
    :param distancia_frente_quinta_roda:        [m] distância entre a dianteira do trator e a quinta roda.
    :param largura_trator:                  [m] largura externa total do trator.

    ---------------------------------- TRAILER ----------------------------------
    :param comprimento_trailer:             [m] comprimento total do trailer.
    :param distancia_eixo_traseiro_trailer_quinta_roda:     [m] distância entre o eixo traseiro do trailer e o pino-rei.
    :param distancia_frente_trailer_quinta_roda:           [m] distância entre a dianteira do trailer e o pino-rei.
    :param largura_trailer:                 [m] largura externa total do trailer.

    -------------------------------- RODAS -------------------------------------
    medidas teoricas de pneus, não são utilizadas no modelo.
    :param largura_roda:                    [m] largura de cada roda (padrão 0.295 m).
    :param comprimento_roda:                [m] diâmetro de cada roda (padrão 0.807 m).
    """

    def __init__(self, parametros: ParametrosGeometricosVeiculo):
        self.angulo_maximo_articulacao = parametros.angulo_maximo_articulacao
        # ---------------- Parâmetros do trator ------------------
        self.comprimento_trator = parametros.comprimento_trator
        self.comprimento_entre_eixos_trator = parametros.distancia_eixo_traseiro_quinta_roda + parametros.distancia_eixo_dianteiro_quinta_roda
        self.distancia_eixo_traseiro_quinta_roda = parametros.distancia_eixo_traseiro_quinta_roda
        self.distancia_frente_quinta_roda = parametros.distancia_frente_quinta_roda 
        self.largura_trator = parametros.largura_trator

        # ---------------- Parâmetros do trailer ------------------
        self.comprimento_trailer = parametros.comprimento_trailer
        self.distancia_eixo_traseiro_trailer_quinta_roda = parametros.distancia_eixo_traseiro_trailer_quinta_roda
        self.distancia_frente_trailer_quinta_roda = parametros.distancia_frente_trailer_quinta_roda
        self.largura_trailer = parametros.largura_trailer

        # ---------------- Parâmetros de Peneus (teorico) ------------------
        self.largura_roda = parametros.largura_roda
        self.comprimento_roda = parametros.comprimento_roda

        self.metade_largura_trator = self.largura_trator / 2
        self.metade_largura_trailer = self.largura_trailer / 2

    def get_larguras(self):
        return {
            'largura_trator': self.largura_trator, 
            'largura_trailer': self.largura_trailer
        }
    
    def get_comprimentos(self):
        return {
            'comprimento_trator': self.comprimento_trator,
            'comprimento_trailer': self.comprimento_trailer,
            'comprimento_entre_eixos_trator': self.comprimento_entre_eixos_trator,
            'distancia_eixo_traseiro_trailer_quinta_roda': self.distancia_eixo_traseiro_trailer_quinta_roda
        }
    
    def get_parametros(self):
        return {
            'comprimento_trator': self.comprimento_trator,
            'comprimento_entre_eixos_trator': self.comprimento_entre_eixos_trator,
            'distancia_frente_quinta_roda': self.distancia_frente_quinta_roda,
            'distancia_eixo_traseiro_quinta_roda': self.distancia_eixo_traseiro_quinta_roda, # M
            'largura_trator': self.largura_trator,
            'comprimento_trailer': self.comprimento_trailer,
            'distancia_eixo_traseiro_trailer_quinta_roda': self.distancia_eixo_traseiro_trailer_quinta_roda,
            'distancia_frente_trailer_quinta_roda': self.distancia_frente_trailer_quinta_roda,
            'largura_trailer': self.largura_trailer,
            'largura_roda': self.largura_roda,
            'comprimento_roda': self.comprimento_roda
        }

    def calcula_vertices_em_metros(self, estado5: Estado5) -> dict:
        """
        Retorna os vértices dos retângulos do trator e do trailer em metros.

        Saída:
        {
            'trator': [(x, y), ...4 pontos...],
            'trailer': [(x, y), ...4 pontos...]
        }
        Ordem dos pontos: traseira esquerda, traseira direita, dianteira direita, dianteira esquerda.
        """
        # Obtém apenas x1, y1, theta1 e theta2
        x_trator, y_trator, ang_trator, _, ang_trailer= estado5.obter_estado_em_metros()

        # Pré-cálculo de seno e cosseno
        seno_trator, cosseno_trator = sin(ang_trator), cos(ang_trator)
        seno_trailer, cosseno_trailer = sin(ang_trailer), cos(ang_trailer)

        # Metades das larguras
        meia_largura_trator = self.metade_largura_trator
        meia_largura_trailer = self.metade_largura_trailer

        # --- Quinta roda do trator / Pino-rei do trailer ---
        # Q_{r} = (x1 + M cos(theta1), y1 + M sen(theta1))
        # M = distância entre o eixo traseiro do trator e a quinta roda (ou pino-rei do trailer)
        quinta_roda_x = x_trator + self.distancia_eixo_traseiro_quinta_roda * cosseno_trator
        quinta_roda_y = y_trator + self.distancia_eixo_traseiro_quinta_roda * seno_trator

        # --- Trator ---
        # Ct_{trator} = comprimento do trator
        # Lm_{trator} = metade da largura do trator
        deslocamento_x_longitudinal_frontal_trator = self.comprimento_trator * cosseno_trator
        deslocamento_y_longitudinal_frontal_trator = self.comprimento_trator * seno_trator
        deslocamento_x_lateral_trator = meia_largura_trator * seno_trator
        deslocamento_y_lateral_trator = meia_largura_trator * cosseno_trator
        deslocamento_x_longitudinal_traseiro_trator = (self.comprimento_trator - self.distancia_frente_quinta_roda) * cosseno_trator
        deslocamento_y_longitudinal_traseiro_trator = (self.comprimento_trator - self.distancia_frente_quinta_roda) * seno_trator

        # V_{FDtrator} = (x1 + Ct_{trator} cos(theta1) + Lm_{trator} sen(theta1), y1 + Ct_{trator} sen(theta1) - Lm_{trator} cos(theta1))
        v_frontal_direita_trator = (
            x_trator + deslocamento_x_longitudinal_frontal_trator + deslocamento_x_lateral_trator,
            y_trator + deslocamento_y_longitudinal_frontal_trator - deslocamento_y_lateral_trator
            )
        # V_{FEtrator} = (x1 + Ct_{trator} cos(theta1) - Lm_{trator} sen(theta1), y1 + Ct_{trator} sen(theta1) + Lm_{trator} cos(theta1))
        v_frontal_esquerda_trator = (
            x_trator + deslocamento_x_longitudinal_frontal_trator - deslocamento_x_lateral_trator, 
            y_trator + deslocamento_y_longitudinal_frontal_trator + deslocamento_y_lateral_trator
        )
        # V_{TDtrator} = (x(Q_{r}) - (Ct_{trator} - D_{FrTrator}) cos(theta1) + Lm_{trator} sen(theta1), y(Q_{r}) - (Ct_{trator} - D_{FrTrator}) sen(theta1) - Lm_{trator} cos(theta1))
        v_traseira_direita_trator = (
            quinta_roda_x - deslocamento_x_longitudinal_traseiro_trator + deslocamento_x_lateral_trator,
            quinta_roda_y - deslocamento_y_longitudinal_traseiro_trator - deslocamento_y_lateral_trator
        )
        # V_{TEtrator} = (x(Q_{r}) - (Ct_{trator} - D_{FrTrator}) cos(theta1) - Lm_{trator} sen(theta1), y(Q_{r}) - (Ct_{trator} - D_{FrTrator}) sen(theta1) + Lm_{trator} cos(theta1))
        v_traseira_esquerda_trator = (
            quinta_roda_x - deslocamento_x_longitudinal_traseiro_trator - deslocamento_x_lateral_trator,
            quinta_roda_y - deslocamento_y_longitudinal_traseiro_trator + deslocamento_y_lateral_trator
        )
        vertices_trator = [
            v_traseira_esquerda_trator, # V_{TEtrator}
            v_traseira_direita_trator,  # V_{TDtrator}
            v_frontal_direita_trator,   # V_{FDtrator}
            v_frontal_esquerda_trator    # V_{FEtrator}
        ]

        # --- Trailer ---
        # D_{FrTrailer} = distância entre a frente do trailer e o pino-rei
        # Ct_{trailer} = comprimento do trailer
        # Lm_{trailer} = metade da largura do trailer
        deslocamento_x_longitudinal_frontal_trailer = self.distancia_frente_trailer_quinta_roda * cosseno_trailer
        deslocamento_y_longitudinal_frontal_trailer = self.distancia_frente_trailer_quinta_roda * seno_trailer
        deslocamento_x_lateral_trailer = meia_largura_trailer * seno_trailer
        deslocamento_y_lateral_trailer = meia_largura_trailer * cosseno_trailer
        deslocamento_x_longitudinal_traseiro_trailer = (self.comprimento_trailer - self.distancia_frente_trailer_quinta_roda) * cosseno_trailer
        deslocamento_y_longitudinal_traseiro_trailer = (self.comprimento_trailer - self.distancia_frente_trailer_quinta_roda) * seno_trailer
        # V_{FDtrailer} = (x(Q_{r}) + D_{FrTrailer} cos(theta2) + Lm_{trailer} sen(theta2), y(Q_{r}) + D_{FrTrailer} sen(theta2) - Lm_{trailer} cos(theta2))
        v_frontal_direita_trailer = (
            quinta_roda_x + deslocamento_x_longitudinal_frontal_trailer + deslocamento_x_lateral_trailer,
            quinta_roda_y + deslocamento_y_longitudinal_frontal_trailer - deslocamento_y_lateral_trailer
        )
        # V_{FEtrailer} = (x(Q_{r}) + D_{FrTrailer} cos(theta2) - Lm_{trailer} sen(theta2), y(Q_{r}) + D_{FrTrailer} sen(theta2) + Lm_{trailer} cos(theta2))
        v_frontal_esquerda_trailer = (
            quinta_roda_x + deslocamento_x_longitudinal_frontal_trailer - deslocamento_x_lateral_trailer,
            quinta_roda_y + deslocamento_y_longitudinal_frontal_trailer + deslocamento_y_lateral_trailer
        )
        # V_{TDtrailer} = (x(Q_{r}) - D_{TrTrailer} cos(theta2) + Lm_{trailer} sen(theta2), y(Q_{r}) - D_{TrTrailer} sen(theta2) - Lm_{trailer} cos(theta2))
        v_traseira_direita_trailer = (
            quinta_roda_x - deslocamento_x_longitudinal_traseiro_trailer + deslocamento_x_lateral_trailer,
            quinta_roda_y - deslocamento_y_longitudinal_traseiro_trailer - deslocamento_y_lateral_trailer
        )

        # V_{TEtrailer} = (x(Q_{r}) - D_{TrTrailer} cos(theta2) - Lm_{trailer} sen(theta2), y(Q_{r}) - D_{TrTrailer} sen(theta2) + Lm_{trailer} cos(theta2))
        v_traseira_esquerda_trailer = (
            quinta_roda_x - deslocamento_x_longitudinal_traseiro_trailer - deslocamento_x_lateral_trailer,
            quinta_roda_y - deslocamento_y_longitudinal_traseiro_trailer + deslocamento_y_lateral_trailer
        )
        vertices_trailer = [
            v_traseira_esquerda_trailer, # V_{TEtrailer}
            v_traseira_direita_trailer,  # V_{TDtrailer}
            v_frontal_direita_trailer,   # V_{FDtrailer}
            v_frontal_esquerda_trailer    # V_{FEtrailer}
        ]

        return {'trator': vertices_trator, 'trailer': vertices_trailer}
    
    def calculate_vertices_pixel(self, estado5: Estado5, resolution: float = 0.0625) -> dict:
        """
        Retorna os vértices dos retângulos do trator e do trailer em pixels.

        Usa o método calcula_vertices_em_metros e aplica fator de resolução (m/pixel).

        Saída:
        {
            'trator': [(x_px, y_px), ...4 pontos...],
            'trailer': [(x_px, y_px), ...4 pontos...]
        }
        Mesma ordem: traseira esquerda, traseira direita, dianteira direita, dianteira esquerda.
        """
        # Calcula vértices em metros
        verts_m = self.calcula_vertices_em_metros(estado5)

        # Converte para pixels
        verts_px = {}
        for parte, pontos in verts_m.items():
            pixel_list = []
            for x_m, y_m in pontos:
                x_px = int(round(x_m / resolution))
                y_px = int(round(y_m / resolution))
                pixel_list.append((x_px, y_px))
            verts_px[parte] = pixel_list

        return verts_px
    
    def calcular_vertices_com_rodas(self, estado5: Estado5, steering_angle: float) -> dict:
        """
        Retorna os vértices do trator, trailer e das rodas dianteiras em metros.

        Saída:
        {
            'trator': [...4 pontos...],
            'trailer': [...4 pontos...],
            'roda_dianteira_esquerda': [...4 pontos...],
            'roda_dianteira_direita': [...4 pontos...]
        }
        Ordem das rodas: 4 vértices no sentido anti-horário.
        """
        # Obtém vértices básicos do trator e trailer
        vertices = self.calcula_vertices_em_metros(estado5)

        # Estado e ângulos
        x_trator, y_trator, theta_trator, *_ = estado5.obter_estado_em_metros()

        # Seno e cosseno do ângulo do trator
        seno_trator = sin(theta_trator)
        cosseno_trator = cos(theta_trator)

        # Ângulo das rodas dianteiras no chassi
        angulo_rodas = theta_trator + steering_angle
        seno_rodas = sin(angulo_rodas)
        cosseno_rodas = cos(angulo_rodas)

        # Posição central das rodas dianteiras
        deslocamento_lateral = self.metade_largura_trator - self.largura_roda / 2
        deslocamento_frontal = self.comprimento_trator - self.comprimento_roda / 2

        # Centro da roda dianteira esquerda
        cx_roda_esq = x_trator + deslocamento_frontal * cosseno_trator - deslocamento_lateral * seno_trator
        cy_roda_esq = y_trator + deslocamento_frontal * seno_trator + deslocamento_lateral * cosseno_trator
        # Centro da roda dianteira direita
        cx_roda_dir = x_trator + deslocamento_frontal * cosseno_trator + deslocamento_lateral * seno_trator
        cy_roda_dir = y_trator + deslocamento_frontal * seno_trator - deslocamento_lateral * cosseno_trator

        # Função auxiliar para calcular os vértices de uma roda
        def calcular_vertices_roda(cx, cy, seno, cosseno):
            metade_comprimento = self.comprimento_roda / 2
            metade_largura = self.largura_roda / 2
            # 4 vértices relativos ao centro
            rel = [(-metade_comprimento, -metade_largura), 
                (-metade_comprimento, +metade_largura), 
                (+metade_comprimento, +metade_largura), 
                (+metade_comprimento, -metade_largura)]
            vertices_roda = []
            for dx, dy in rel:
                x = cx + dx * cosseno - dy * seno
                y = cy + dx * seno + dy * cosseno
                vertices_roda.append((x, y))
            return vertices_roda

        # Monta dicionário de rodas
        vertices_rodas = {
            'roda_dianteira_esquerda': calcular_vertices_roda(cx_roda_esq, cy_roda_esq, seno_rodas, cosseno_rodas),
            'roda_dianteira_direita':  calcular_vertices_roda(cx_roda_dir, cy_roda_dir, seno_rodas, cosseno_rodas)
        }

        # Atualiza e retorna
        vertices.update(vertices_rodas)
        return vertices

class ParametrosTrator():
    """ Parâmetros geométricos do conjunto trator + trailer.
    ---------------------------------- TRATOR ----------------------------------
    :param comprimento_trator:                [m] comprimento total do trator.
    :param distancia_eixo_traseiro_quinta_roda: [m] distância entre o eixo traseiro do trator e a quinta roda.
    :param distancia_eixo_dianteiro_quinta_roda: [m] distância entre o eixo dianteiro do trator e a quinta roda.
    :param distancia_frente_quinta_roda:        [m] distância entre a dianteira do trator e a quinta roda.
    :param largura_trator:                  [m] largura externa total do trator.

    ---------------------------------- TRAILER ----------------------------------
    :param comprimento_trailer:             [m] comprimento total do trailer.
    :param distancia_eixo_traseiro_trailer_quinta_roda:     [m] distância entre o eixo traseiro do trailer e o pino-rei.
    :param distancia_frente_trailer_quinta_roda:           [m] distância entre a dianteira do trailer e o pino-rei.
    :param largura_trailer:                 [m] largura externa total do trailer.

    -------------------------------- RODAS -------------------------------------
    medidas teoricas de pneus, não são utilizadas no modelo.
    :param largura_roda:                    [m] largura de cada roda (padrão 0.295 m).
    :param comprimento_roda:                [m] diâmetro de cada roda (padrão 0.807 m).
    """

    def __init__(self, parametros: ParametrosGeometricosVeiculo):
        self.angulo_maximo_articulacao = parametros.angulo_maximo_articulacao
        # ---------------- Parâmetros do trator ------------------
        self.comprimento_trator = parametros.comprimento_trator
        self.comprimento_entre_eixos_trator = parametros.distancia_eixo_traseiro_quinta_roda + parametros.distancia_eixo_dianteiro_quinta_roda
        self.distancia_eixo_traseiro_quinta_roda = parametros.distancia_eixo_traseiro_quinta_roda
        self.distancia_frente_quinta_roda = parametros.distancia_frente_quinta_roda 
        self.largura_trator = parametros.largura_trator

        # ---------------- Parâmetros de Peneus (teorico) ------------------
        self.largura_roda = parametros.largura_roda
        self.comprimento_roda = parametros.comprimento_roda

        self.metade_largura_trator = self.largura_trator / 2

    def get_larguras(self):
        return {
            'largura_trator': self.largura_trator, 
        }
    
    def get_comprimentos(self):
        return {
            'comprimento_trator': self.comprimento_trator,
            'comprimento_entre_eixos_trator': self.comprimento_entre_eixos_trator,
        }
    
    def get_parametros(self):
        return {
            'comprimento_trator': self.comprimento_trator,
            'comprimento_entre_eixos_trator': self.comprimento_entre_eixos_trator,
            'distancia_frente_quinta_roda': self.distancia_frente_quinta_roda,
            'distancia_eixo_traseiro_quinta_roda': self.distancia_eixo_traseiro_quinta_roda, # M
            'largura_trator': self.largura_trator,
            'largura_roda': self.largura_roda,
            'comprimento_roda': self.comprimento_roda
        }

    def calcula_vertices_em_metros(self, estado3: Estado3) -> dict:
        """
        Retorna os vértices dos retângulos do trator e do trailer em metros.

        Saída:
        {
            'trator': [(x, y), ...4 pontos...],
            'trailer': [(x, y), ...4 pontos...]
        }
        Ordem dos pontos: traseira esquerda, traseira direita, dianteira direita, dianteira esquerda.
        """
        # Obtém apenas x1, y1, theta1 e theta2
        x_trator, y_trator, ang_trator = estado3.obter_estado_em_metros()

        # Pré-cálculo de seno e cosseno
        seno_trator, cosseno_trator = sin(ang_trator), cos(ang_trator)

        # Metades das larguras
        meia_largura_trator = self.metade_largura_trator

        # --- Quinta roda do trator / Pino-rei do trailer ---
        # Q_{r} = (x1 + M cos(theta1), y1 + M sen(theta1))
        # M = distância entre o eixo traseiro do trator e a quinta roda (ou pino-rei do trailer)
        quinta_roda_x = x_trator + self.distancia_eixo_traseiro_quinta_roda * cosseno_trator
        quinta_roda_y = y_trator + self.distancia_eixo_traseiro_quinta_roda * seno_trator

        # --- Trator ---
        # Ct_{trator} = comprimento do trator
        # Lm_{trator} = metade da largura do trator
        deslocamento_x_longitudinal_frontal_trator = self.comprimento_trator * cosseno_trator
        deslocamento_y_longitudinal_frontal_trator = self.comprimento_trator * seno_trator
        deslocamento_x_lateral_trator = meia_largura_trator * seno_trator
        deslocamento_y_lateral_trator = meia_largura_trator * cosseno_trator
        deslocamento_x_longitudinal_traseiro_trator = (self.comprimento_trator - self.distancia_frente_quinta_roda) * cosseno_trator
        deslocamento_y_longitudinal_traseiro_trator = (self.comprimento_trator - self.distancia_frente_quinta_roda) * seno_trator

        # V_{FDtrator} = (x1 + Ct_{trator} cos(theta1) + Lm_{trator} sen(theta1), y1 + Ct_{trator} sen(theta1) - Lm_{trator} cos(theta1))
        v_frontal_direita_trator = (
            x_trator + deslocamento_x_longitudinal_frontal_trator + deslocamento_x_lateral_trator,
            y_trator + deslocamento_y_longitudinal_frontal_trator - deslocamento_y_lateral_trator
            )
        # V_{FEtrator} = (x1 + Ct_{trator} cos(theta1) - Lm_{trator} sen(theta1), y1 + Ct_{trator} sen(theta1) + Lm_{trator} cos(theta1))
        v_frontal_esquerda_trator = (
            x_trator + deslocamento_x_longitudinal_frontal_trator - deslocamento_x_lateral_trator, 
            y_trator + deslocamento_y_longitudinal_frontal_trator + deslocamento_y_lateral_trator
        )
        # V_{TDtrator} = (x(Q_{r}) - (Ct_{trator} - D_{FrTrator}) cos(theta1) + Lm_{trator} sen(theta1), y(Q_{r}) - (Ct_{trator} - D_{FrTrator}) sen(theta1) - Lm_{trator} cos(theta1))
        v_traseira_direita_trator = (
            quinta_roda_x - deslocamento_x_longitudinal_traseiro_trator + deslocamento_x_lateral_trator,
            quinta_roda_y - deslocamento_y_longitudinal_traseiro_trator - deslocamento_y_lateral_trator
        )
        # V_{TEtrator} = (x(Q_{r}) - (Ct_{trator} - D_{FrTrator}) cos(theta1) - Lm_{trator} sen(theta1), y(Q_{r}) - (Ct_{trator} - D_{FrTrator}) sen(theta1) + Lm_{trator} cos(theta1))
        v_traseira_esquerda_trator = (
            quinta_roda_x - deslocamento_x_longitudinal_traseiro_trator - deslocamento_x_lateral_trator,
            quinta_roda_y - deslocamento_y_longitudinal_traseiro_trator + deslocamento_y_lateral_trator
        )
        vertices_trator = [
            v_traseira_esquerda_trator, # V_{TEtrator}
            v_traseira_direita_trator,  # V_{TDtrator}
            v_frontal_direita_trator,   # V_{FDtrator}
            v_frontal_esquerda_trator    # V_{FEtrator}
        ]


        return {'trator': vertices_trator}
    
    def calculate_vertices_pixel(self, estado3: Estado3, resolution: float = 0.0625) -> dict:
        """
        Retorna os vértices dos retângulos do trator e do trailer em pixels.

        Usa o método calcula_vertices_em_metros e aplica fator de resolução (m/pixel).

        Saída:
        {
            'trator': [(x_px, y_px), ...4 pontos...],
            'trailer': [(x_px, y_px), ...4 pontos...]
        }
        Mesma ordem: traseira esquerda, traseira direita, dianteira direita, dianteira esquerda.
        """
        # Calcula vértices em metros
        verts_m = self.calcula_vertices_em_metros(estado3)

        # Converte para pixels
        verts_px = {}
        for parte, pontos in verts_m.items():
            pixel_list = []
            for x_m, y_m in pontos:
                x_px = int(round(x_m / resolution))
                y_px = int(round(y_m / resolution))
                pixel_list.append((x_px, y_px))
            verts_px[parte] = pixel_list

        return verts_px
    
    def calcular_vertices_com_rodas(self, estado3: Estado3, steering_angle: float) -> dict:
        """
        Retorna os vértices do trator, trailer e das rodas dianteiras em metros.

        Saída:
        {
            'trator': [...4 pontos...],
            'trailer': [...4 pontos...],
            'roda_dianteira_esquerda': [...4 pontos...],
            'roda_dianteira_direita': [...4 pontos...]
        }
        Ordem das rodas: 4 vértices no sentido anti-horário.
        """
        # Obtém vértices básicos do trator e trailer
        vertices = self.calcula_vertices_em_metros(estado3)

        # Estado e ângulos
        x_trator, y_trator, theta_trator = estado3.obter_estado_em_metros()

        # Seno e cosseno do ângulo do trator
        seno_trator = sin(theta_trator)
        cosseno_trator = cos(theta_trator)

        # Ângulo das rodas dianteiras no chassi
        angulo_rodas = theta_trator + steering_angle
        seno_rodas = sin(angulo_rodas)
        cosseno_rodas = cos(angulo_rodas)

        # Posição central das rodas dianteiras
        deslocamento_lateral = self.metade_largura_trator - self.largura_roda / 2
        deslocamento_frontal = self.comprimento_trator - self.comprimento_roda / 2

        # Centro da roda dianteira esquerda
        cx_roda_esq = x_trator + deslocamento_frontal * cosseno_trator - deslocamento_lateral * seno_trator
        cy_roda_esq = y_trator + deslocamento_frontal * seno_trator + deslocamento_lateral * cosseno_trator
        # Centro da roda dianteira direita
        cx_roda_dir = x_trator + deslocamento_frontal * cosseno_trator + deslocamento_lateral * seno_trator
        cy_roda_dir = y_trator + deslocamento_frontal * seno_trator - deslocamento_lateral * cosseno_trator

        # Função auxiliar para calcular os vértices de uma roda
        def calcular_vertices_roda(cx, cy, seno, cosseno):
            metade_comprimento = self.comprimento_roda / 2
            metade_largura = self.largura_roda / 2
            # 4 vértices relativos ao centro
            rel = [(-metade_comprimento, -metade_largura), 
                (-metade_comprimento, +metade_largura), 
                (+metade_comprimento, +metade_largura), 
                (+metade_comprimento, -metade_largura)]
            vertices_roda = []
            for dx, dy in rel:
                x = cx + dx * cosseno - dy * seno
                y = cy + dx * seno + dy * cosseno
                vertices_roda.append((x, y))
            return vertices_roda

        # Monta dicionário de rodas
        vertices_rodas = {
            'roda_dianteira_esquerda': calcular_vertices_roda(cx_roda_esq, cy_roda_esq, seno_rodas, cosseno_rodas),
            'roda_dianteira_direita':  calcular_vertices_roda(cx_roda_dir, cy_roda_dir, seno_rodas, cosseno_rodas)
        }

        # Atualiza e retorna
        vertices.update(vertices_rodas)
        return vertices

