# dominio.estrutura_de_dados.estados

from esqueleto.calculos_com_angulos import Angulos

somar_angulos = Angulos().soma_angulo_normalizado

class Estado3():
    def __init__(self, x, y, theta, resolucao=0.0625):
        self.x = x
        self.y = y
        self.theta = theta
        self.resolucao = resolucao
    
    def obter_x(self):
        return self.x
    
    def obter_y(self):
        return self.y
    
    def obter_theta(self):
        return self.theta
    
    def obter_estado_em_metros(self):
        return (self.x, self.y, self.theta)
    
    def obter_estado_em_pixel(self):
        return (round(self.x/self.resolucao, 4), self.theta)

    def __eq__(self, other):
        """Verifica a igualdade entre dois estados"""
        return self.x == other.x and self.y == other.y and self.theta == other.theta
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))

class Estado4():
    def __init__(self, x, y, theta, beta, resolucao=0.0625):
        """ x e y são as coordenadas do centro do veículo, theta é o ângulo do veículo e beta é o ângulo do reboque.

        Args:
            x (float): coordenada x do centro do veículo em metros
            y (float): coordenada y do centro do veículo em metros
            theta (float): ângulo do veículo em radianos
            beta (float): ângulo de articulação em radianos
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.beta = beta
        self.resolucao = resolucao
    
    def obter_x(self):
        return self.x
    
    def obter_y(self):
        return self.y
    
    def obter_theta(self):
        return self.theta
    
    def obter_theta2(self):
        return somar_angulos(self.theta, self.beta)
    
    def obter_beta(self):
        return self.beta
    
    def obter_estado_em_metros(self):
        return (self.x, self.y, self.theta, self.beta)
    
    def obter_estado_em_pixel(self):
        return (round(self.x/self.resolucao, 4), round(self.y/self.resolucao, 4), self.theta, self.beta)

    def em_dicionario(self):
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'beta': self.beta,
        }

    def __eq__(self, other):
        """Verifica a igualdade entre dois estados"""
        return self.x == other.x and self.y == other.y and self.theta == other.theta and self.beta == other.beta
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta, self.beta))

class Estado5():
    def __init__(self, x, y, theta, beta, theta2, resolucao=0.0625):
        self.x = x
        self.y = y
        self.theta = theta
        self.beta = beta
        self.theta2 = theta2
        self.resolucao = resolucao
    
    def obter_x(self):
        return self.x
    
    def obter_y(self):
        return self.y
    
    def obter_theta(self):
        return self.theta
    
    def obter_beta(self):
        return self.beta
    
    def obter_theta2(self):
        return self.theta2
    
    def obter_estado_em_metros(self):
        return (self.x, self.y, self.theta, self.beta, self.theta2)
    
    def obter_estado_em_pixel(self):
        return (round(self.x/self.resolucao, 4), round(self.y/self.resolucao, 4), self.theta, 
                round(self.x/self.resolucao, 4), round(self.y/self.resolucao, 4), self.theta2, self.beta)
    def em_dicionario(self):
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'beta': self.beta,
            'theta2': self.theta2
        }
    def __eq__(self, other):
        """Verifica a igualdade entre dois estados"""
        return self.x == other.x and self.y == other.y and self.theta == other.theta and self.beta == other.beta and self.theta2 == other.theta2
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta, self.beta, self.theta2))
    
class Estado7():
    def __init__(self, x1, y1, theta1, x2, y2, theta2, beta, resolucao=0.0625):
        self.x1 = x1
        self.y1 = y1
        self.theta1 = theta1
        self.beta = beta
        self.x2 = x2
        self.y2 = y2
        self.theta2 = theta2
        self.resolucao = resolucao
    
    def obter_x(self):
        return self.x1
    
    def obter_y(self):
        return self.y1
    
    def obter_theta(self):
        return self.theta1
    
    def obter_beta(self):
        return self.beta
    
    def obter_x2(self):
        return self.x2
    
    def obter_y2(self):
        return self.y2
    
    def obter_theta2(self):
        return self.theta2
    
    def obter_estado_em_metros(self):
        return (self.x1, self.y1, self.theta1, self.x2, self.y2, self.theta2, self.beta)
    
    def obter_estado_em_pixel(self):
        return (round(self.x1/self.resolucao, 4), round(self.y1/self.resolucao, 4), self.theta1, 
                round(self.x2/self.resolucao, 4), round(self.y2/self.resolucao, 4), self.theta2, self.beta)
    
    # Função para converter State7 em dicionário
    def em_dicionario(self):
        return {
            'x': self.x1,
            'y': self.y1,
            'theta': self.theta1,
            'x2': self.x2,
            'y2': self.y2,
            'theta2': self.theta2,
            'beta': self.beta
        }

    def __eq__(self, other):
        """Verifica a igualdade entre dois estados"""
        return self.x1 == other.x1 and self.y1 == other.y1 and self.theta1 == other.theta1 and self.x2 == other.x2 and self.y2 == other.y2 and self.theta2 == other.theta2 and self.beta == other.beta
    
    def __hash__(self):
        return hash((self.x1, self.y1, self.theta1, self.x2, self.y2, self.theta2, self.beta))