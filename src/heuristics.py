import numpy as np
import math

def calcular_distancia_nao_holonomica_carro(estado_atual, estado_alvo, l_carro):
    """
    Calcula a distância homogênea (custo) entre dois estados para um carro cinemático.
    
    A métrica utiliza coordenadas privilegiadas e pesos não-holonômicos conforme
    definido em [Rosenfelder2022]_, penalizando erros laterais com potências menores
    (maior custo relativo próximo à origem).

    .. math::
    
        J(z) = z_1^{12} + z_3^6 + z_4^4

    :param estado_atual: Vetor de estado atual [x, y, theta].
    :type estado_atual: np.ndarray
    :param estado_alvo: Vetor de estado desejado [x_d, y_d, theta_d].
    :type estado_alvo: np.ndarray
    :param l_carro: Distância entre eixos (wheelbase).
    :type l_carro: float
    :return: Valor escalar representando a 'distância' de custo não-holonômico.
    :rtype: float

    .. note::
       Esta função assume pesos w=(1, 1, 2, 3) e grau homogêneo d=12.
       
    :Exemplo:

    >>> x_a = np.array([0.0, 0.5, 0.0]) # Desalinhado lateralmente (difícil)
    >>> x_b = np.array([0.5, 0.0, 0.0]) # Desalinhado longitudinalmente (fácil)
    >>> alvo = np.array([0.0, 0.0, 0.0])
    >>> # Lateral (z4) tem peso 3 -> expoente 4 -> 0.5^4 = 0.0625
    >>> # Longitudinal (z1) tem peso 1 -> expoente 12 -> 0.5^12 = 0.0002
    >>> # Conclusão: O custo lateral é muito maior, refletindo a dificuldade.
    """
    
    # 1. Calcular erro no referencial do alvo (simplificado para alvo na origem local)
    dx = estado_atual[0] - estado_alvo[0]
    dy = estado_atual[1] - estado_alvo[1]
    dtheta = estado_atual[2] - estado_alvo[2]
    
    # Rotação para o frame do alvo
    # Nota: Em MPC local, muitas vezes se usa o erro direto se o frame for móvel,
    # mas a forma rigorosa exige rotacionar dx, dy pelo theta_alvo.
    c, s = np.cos(estado_alvo[2]), np.sin(estado_alvo[2])
    x_erro_rot = dx * c + dy * s
    y_erro_rot = -dx * s + dy * c
    
    # 2. Transformar para Coordenadas Privilegiadas (Eq. 18 do paper)
    # z = [x, -l*theta, l*y]
    z1 = x_erro_rot
    z3 = -l_carro * dtheta
    z4 = l_carro * y_erro_rot
    
    # 3. Calcular Custo/Distância Homogênea (Eq. 21 do paper)
    # Exponentes: d/w -> 12/1, 12/1, 12/2, 12/3
    custo = (z1**12) + (z3**6) + (z4**4)

    
    return math.log(1+ custo,10)

