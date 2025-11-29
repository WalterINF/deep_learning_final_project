from esqueleto.estados import Estado4, Estado3# import numpy as np
import math
# import numpy as np
from math import pi
from esqueleto.calculos_com_angulos import Angulos
from esqueleto.data_class_loader.dc_loader_heuristica import ParametrosHeuristicaPerpendicular as ParametrosHeuristica

class XYDecomposta():
    def __init__(self, poseInicial: Estado4, poseFinal: Estado4):
        # Parametro para normalização da distância. distancia de x e y máxima possível no ambiente.
        self.max_distancia_x = 100.0  # Exemplo: 100 metros
        self.max_distancia_y = 90.0  # Exemplo: 90 metros

    def estimar(self, poseInicial: Estado4, poseFinal: Estado4) -> tuple[list, tuple[float, float]]:
        """ Calcula a distância decomposto nos eixos x e y entre poseInicial e poseFinal """
        poseInicial = poseInicial.obter_estado_em_metros()
        poseFinal = poseFinal.obter_estado_em_metros()

        erro_x = abs(poseFinal[0] - poseInicial[0]) / self.max_distancia_x
        erro_y = abs(poseFinal[1] - poseInicial[1]) / self.max_distancia_y

        pontos = []  # Lista de estados intermediários (pode ser preenchida conforme necessário)

        return (pontos, (erro_x, erro_y))


angulos = Angulos()
class HTheta(IHeuristica):
    def __init__(self):  
        pass

    def estimar(self, poseInicial: Estado4, poseFinal: Estado4) -> tuple[list, float]:
        poseInicial = poseInicial.obter_estado_em_metros()
        poseFinal = poseFinal.obter_estado_em_metros()
        theta_erro = angulos.menor_diferenca_entre_angulos(poseInicial[2], poseFinal[2])
        return ([], theta_erro)

class HTheta2(IHeuristica):
    def __init__(self):
        pass

    def estimar(self, poseInicial: Estado4, poseFinal: Estado4) -> tuple[list, float]:
        theta2_poseInicial = poseInicial.obter_theta2()
        theta2_poseFinal = poseFinal.obter_theta2()
        theta2_erro = angulos.menor_diferenca_entre_angulos(theta2_poseInicial, theta2_poseFinal)
        theta2_erro = theta2_erro/pi # normalizando para [0, 1]
        return ([], theta2_erro)
    
# 3) Refatoração da heurística combinada para usar parâmetros carregados
class HCoordenadaPerpendicularThetas(IHeuristica):
    def __init__(
        self,
        poseInicial: Estado4,
        poseFinal: Estado4,
        params: ParametrosHeuristica,
    ):
        # armazena parâmetros carregados
        self.params = params

        self.xy_decomposta = XYDecomposta(poseInicial, poseFinal)
        
        # Descobre o eixo mais perpendicular e a porcentagem de perpendicularidade
        self.eixo_perpendicular, self.porcentagem_perpendicularidade = self.analise_perpendicularidade_eficiente(poseFinal.obter_theta())
            
    def analise_perpendicularidade_eficiente(self, theta: float) -> tuple[str, float]:
        """
        Identifica o eixo cartesiano mais perpendicular e calcula sua porcentagem 
        de perpendicularidade (de 0 a 1) em relação à direção theta.

        A Porcentagem de Perpendicularidade é definida aqui pelo ALINHAMENTO do 
        eixo mais PARALELO ao ângulo theta, que corresponde aos valores de 1 
        nas direções cardeais (eixos x ou y).
        
        Args:
            theta: O ângulo em radianos.

        Returns:
            Uma tupla (eixo_mais_perpendicular, porcentagem_perpendicularidade).
        """
        
        # 1. Calcula o alinhamento de theta com os eixos x e y.
        abs_cos = abs(math.cos(theta)) # Alinhamento com X (Paralelismo de X)
        abs_sin = abs(math.sin(theta)) # Alinhamento com Y (Paralelismo de Y)
        
        # 2. Identifica o eixo mais perpendicular (aquele com menor alinhamento).
        # Se abs_cos > abs_sin: X é mais paralelo. Logo, Y é o mais perpendicular.
        if abs_cos >= abs_sin:
            eixo_perpendicular = 'y'
            # A porcentagem de perpendicularidade é o alinhamento do eixo X (abs_cos).
            # Ex: Se theta=pi, abs_cos=1. Y é 100% perpendicular.
            porcentagem = abs_cos 
        else: # abs_sin > abs_cos
            eixo_perpendicular = 'x'
            # A porcentagem de perpendicularidade é o alinhamento do eixo Y (abs_sin).
            # Ex: Se theta=-pi/2, abs_sin=1. X é 100% perpendicular.
            porcentagem = abs_sin 
            
        return eixo_perpendicular, porcentagem

    def estimar(self, poseAtual: Estado4, poseObjetivo: Estado4) -> tuple[list, float]:
        """ Estima o custo heurístico baseado em coordenada chefe e ângulos. 
            De modo que:
                * Enquanto o erro da coordenada perpendicular for maior que o threshold1,
                a heurística prioriza a correção dessa coordenada.
                * Apos estar dentro do threshold1, a heurística começa a considerar a variação dos ângulos (theta1 e theta2).
                * Apos estar dentro do threshold2, a heurística começa a conciderar a variação da coordenada não-perpendicular.
                Enquanto estiver fora dos thresholds, a heurística considera o erro maximo possível (1.0) para os termos não priorizados.
        
        """        
        # 1. Obter erros de X e Y (coordenadas)
        _, (erro_x, erro_y) = self.xy_decomposta.estimar(poseAtual, poseObjetivo)
        
        # 2. Mapear erros e pesos
        
        # Define qual é o erro perpendicular e qual é o erro paralelo
        if self.eixo_perpendicular == 'x':
            erro_perp = erro_x
            erro_paralela = erro_y
        else: # self.eixo_perpendicular == 'y'
            erro_perp = erro_y
            erro_paralela = erro_x
            
        # Pesos dos termos (carregados do dataclass ParametrosHeuristica)
        peso_perp = self.params.peso_coordenada_perpendicular
        peso_paralela = self.params.peso_coordenada_nao_perpendicular
        peso_theta1 = self.params.peso_theta1
        peso_theta2 = self.params.peso_theta2
        
        # Thresholds
        threshold1 = self.params.threshold_distancia_paralela
        threshold2 = self.params.threshold_distancia_theta2
        
        # 3. Obter erros angulares (normalizados para [0, 1])
        
        # Erro Theta1 (Normalização: angulo/pi)
        _, theta1_erro_abs = HTheta().estimar(poseAtual, poseObjetivo)
        theta1_erro_norm = theta1_erro_abs / pi 
        
        # Erro Theta2 (Já normalizado dentro de HTheta2)
        _, theta2_erro_norm = HTheta2().estimar(poseAtual, poseObjetivo)
        
        # 4. Cálculo do Custo por Fases
        
        # Phase 1: Priorizar Erro Perpendicular (erro_perp > threshold1)
        if erro_perp > threshold1:
            # Termos não priorizados (Paralela, Theta1, Theta2) recebem erro máximo (1.0)
            custo = (peso_perp * erro_perp +
                     peso_paralela * 1.0 +
                     peso_theta1 * 1.0 +
                     peso_theta2 * 1.0)
            
        # Phase 2: Priorizar Erro Angular (threshold2 < erro_perp <= threshold1)
        elif erro_perp > threshold2:
            # Erro Paralelo recebe máximo (1.0). Ângulos recebem o erro real.
            custo = (peso_perp * erro_perp +
                     peso_paralela * 1.0 +
                     peso_theta1 * theta1_erro_norm +
                     peso_theta2 * theta2_erro_norm)

        # Phase 3: Considerar Todos os Erros (erro_perp <= threshold2)
        else:
            # Todos os erros recebem seus valores reais
            custo = (peso_perp * erro_perp +
                     peso_paralela * erro_paralela +
                     peso_theta1 * theta1_erro_norm +
                     peso_theta2 * theta2_erro_norm)

        # A lista de pontos intermediários (a ser preenchida por outras heurísticas) é vazia aqui
        return ([], custo)
    
    
class HCoordenadaPerpendicularTheta(IHeuristica):
    def __init__(
        self,
        poseInicial: Estado3,
        poseFinal: Estado3,
        params: ParametrosHeuristica,
    ):
        # armazena parâmetros carregados
        self.params = params

        self.xy_decomposta = XYDecomposta(poseInicial, poseFinal)
        
        # Descobre o eixo mais perpendicular e a porcentagem de perpendicularidade
        self.eixo_perpendicular, self.porcentagem_perpendicularidade = self.analise_perpendicularidade_eficiente(poseFinal.obter_theta())
            
    def analise_perpendicularidade_eficiente(self, theta: float) -> tuple[str, float]:
        """
        Identifica o eixo cartesiano mais perpendicular e calcula sua porcentagem 
        de perpendicularidade (de 0 a 1) em relação à direção theta.

        A Porcentagem de Perpendicularidade é definida aqui pelo ALINHAMENTO do 
        eixo mais PARALELO ao ângulo theta, que corresponde aos valores de 1 
        nas direções cardeais (eixos x ou y).
        
        Args:
            theta: O ângulo em radianos.

        Returns:
            Uma tupla (eixo_mais_perpendicular, porcentagem_perpendicularidade).
        """
        
        # 1. Calcula o alinhamento de theta com os eixos x e y.
        abs_cos = abs(math.cos(theta)) # Alinhamento com X (Paralelismo de X)
        abs_sin = abs(math.sin(theta)) # Alinhamento com Y (Paralelismo de Y)
        
        # 2. Identifica o eixo mais perpendicular (aquele com menor alinhamento).
        # Se abs_cos > abs_sin: X é mais paralelo. Logo, Y é o mais perpendicular.
        if abs_cos >= abs_sin:
            eixo_perpendicular = 'y'
            # A porcentagem de perpendicularidade é o alinhamento do eixo X (abs_cos).
            # Ex: Se theta=pi, abs_cos=1. Y é 100% perpendicular.
            porcentagem = abs_cos 
        else: # abs_sin > abs_cos
            eixo_perpendicular = 'x'
            # A porcentagem de perpendicularidade é o alinhamento do eixo Y (abs_sin).
            # Ex: Se theta=-pi/2, abs_sin=1. X é 100% perpendicular.
            porcentagem = abs_sin 
            
        return eixo_perpendicular, porcentagem

    def estimar(self, poseAtual: Estado3, poseObjetivo: Estado3) -> tuple[list, float]:
        """ Estima o custo heurístico baseado em coordenada chefe e ângulos. 
            De modo que:
                * Enquanto o erro da coordenada perpendicular for maior que o threshold1,
                a heurística prioriza a correção dessa coordenada.
                * Apos estar dentro do threshold1, a heurística começa a considerar a variação dos ângulos (theta1 e theta2).
                * Apos estar dentro do threshold2, a heurística começa a conciderar a variação da coordenada não-perpendicular.
                Enquanto estiver fora dos thresholds, a heurística considera o erro maximo possível (1.0) para os termos não priorizados.
        
        """        
        # 1. Obter erros de X e Y (coordenadas)
        _, (erro_x, erro_y) = self.xy_decomposta.estimar(poseAtual, poseObjetivo)
        
        # 2. Mapear erros e pesos
        
        # Define qual é o erro perpendicular e qual é o erro paralelo
        if self.eixo_perpendicular == 'x':
            erro_perp = erro_x
            erro_paralela = erro_y
        else: # self.eixo_perpendicular == 'y'
            erro_perp = erro_y
            erro_paralela = erro_x
            
        # Pesos dos termos (carregados do dataclass ParametrosHeuristica)
        peso_perp = self.params.peso_coordenada_perpendicular
        peso_paralela = self.params.peso_coordenada_nao_perpendicular
        peso_theta1 = self.params.peso_theta1
        
        # Thresholds
        threshold1 = self.params.threshold_distancia_paralela
        threshold2 = self.params.threshold_distancia_theta2
        
        # 3. Obter erros angulares (normalizados para [0, 1])
        
        # Erro Theta1 (Normalização: angulo/pi)
        _, theta1_erro_abs = HTheta().estimar(poseAtual, poseObjetivo)
        theta1_erro_norm = theta1_erro_abs / pi 
        
        # Erro Theta2 (Já normalizado dentro de HTheta2)
        
        # 4. Cálculo do Custo por Fases
        
        # Phase 1: Priorizar Erro Perpendicular (erro_perp > threshold1)
        if erro_perp > threshold1:
            # Termos não priorizados (Paralela, Theta1, Theta2) recebem erro máximo (1.0)
            custo = (peso_perp * erro_perp +
                     peso_paralela * 1.0 +
                     peso_theta1 * 1.0)
        
        # Phase 2: Priorizar Erro Angular (threshold2 < erro_perp <= threshold1)
        elif erro_perp > threshold2:
            # Erro Paralelo recebe máximo (1.0). Ângulos recebem o erro real.
            custo = (peso_perp * erro_perp +
                     peso_paralela * 1.0 +
                     peso_theta1 * theta1_erro_norm )
        # Phase 3: Considerar Todos os Erros (erro_perp <= threshold2)
        else:
            # Todos os erros recebem seus valores reais
            custo = (peso_perp * erro_perp +
                     peso_paralela * erro_paralela +
                     peso_theta1 * theta1_erro_norm)

        # A lista de pontos intermediários (a ser preenchida por outras heurísticas) é vazia aqui
        return ([], custo)
    