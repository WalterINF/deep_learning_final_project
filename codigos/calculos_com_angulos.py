from math import pi
from interfaces.i_angulos import IEntre2Angulos
from functools import lru_cache

class Angulos(IEntre2Angulos):
    def __init__(self):
        pass
    # TODO: Implementar o calculo das variações angulares possiveis e armazenar em um cache
     
    # Cache com limite de 100 entradas.
    @lru_cache(maxsize=100)
    def menor_diferenca_entre_angulos(self, angulo1, angulo2, usarDirecao=False):
        """
        Calcula o menor ângulo em radianos entre dois ângulos distintos.    
        """
        # Normalizao angulo para o intervalo [-pi, pi] caso ele seja maior que pi ou menor que -pi
        angulo1 = self.normalize_if_needed(angulo1)
        angulo2 = self.normalize_if_needed(angulo2)
        
        # Calcula a diferença entre os ângulos normalizados
        diff = self.normalize_if_needed(angulo1 - angulo2)

        if usarDirecao:
            return diff
        else:
            return abs(diff)

    @lru_cache(maxsize=100)
    def normalize_if_needed(self, angle):
        """
        Normaliza um ângulo para o intervalo [-pi, pi] se necessário.
        
        Parâmetros:
            angle (float): O ângulo a ser normalizado.

        Retorna:
            float: O ângulo normalizado no intervalo [-pi, pi].
        """
        if angle > pi or angle < -pi:
            angle = self.normalize_angle(angle)
        
        return angle

    @lru_cache(maxsize=100)
    def normalize_angle(self, angle):
        """
        Normaliza um ângulo para o intervalo [-pi, pi].
        
        Parâmetros:
            angle (float): O ângulo a ser normalizado.

        Retorna:
            float: O ângulo normalizado no intervalo [-pi, pi].
        """
        return (angle + pi) % (2 * pi) - pi
    
    def soma_angulo_normalizado(self, angulo1, angulo2):
        """
        Soma dois ângulos e normaliza o resultado para o intervalo [-pi, pi].
        
        Parâmetros:
            angulo1 (float): O primeiro ângulo.
            angulo2 (float): O segundo ângulo.

        Retorna:
            float: A soma dos ângulos normalizada no intervalo [-pi, pi].
        """
        angulo1 = self.normalize_if_needed(angulo1)
        angulo2 = self.normalize_if_needed(angulo2)
        return self.normalize_if_needed(angulo1 + angulo2)