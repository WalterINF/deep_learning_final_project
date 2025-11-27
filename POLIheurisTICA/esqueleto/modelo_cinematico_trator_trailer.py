from casadi import SX, DM, vertcat, cos, sin, tan, integrator
# from utils.decorators.validate_estados import validate4estado
from esqueleto.calculos_com_angulos import Angulos
from esqueleto.estados import Estado7, Estado4, Estado5, Estado3

class TratorComUmTrailer():
    def __init__(self, parametros: dict, dt: float):
        """
        Modelo cinemático do truck-trailer em estado reduzido [x1, y1, theta1, beta]
        Equações de movimento de acordo com o problema de estacionamento:
          ẋ = v·cos(theta1)
          ẏ = v·sin(theta1)
          thetȧ = (v/D)·tan(alpha)
          betȧ = - (v/L)·sin(beta) - (v/D)·tan(alpha)
        Args:
            parametros (dict): parâmetros do modelo (metros)
                - 'comprimento_entre_eixos_trator' = D (entre-eixos do caminhão)
                - 'distancia_eixo_traseiro_trailer_quinta_roda' = L (entre-eixos do trailer)
                - 'distancia_eixo_traseiro_quinta_roda' = distância da quinta roda ao eixo traseiro do trator
            dt (float): passo de integração (s)
        """
        # Parâmetros físicos
        self.comprimento_trator = parametros.comprimento_entre_eixos_trator
        self.comprimento_trailer = parametros.distancia_eixo_traseiro_trailer_quinta_roda
        self.quinta_roda = parametros.distancia_eixo_traseiro_quinta_roda
        self.angulos = Angulos()

        # Notação simplificada
        self.D = self.comprimento_trator
        self.L = self.comprimento_trailer
        self.dt = dt

        # Variáveis simbólicas de estado e controle
        self.x1 = SX.sym('x1')       # posição x do truck
        self.y1 = SX.sym('y1')       # posição y do truck
        self.theta1 = SX.sym('theta1') # orientação do truck
        self.beta = SX.sym('beta')     # ângulo relativo (truck-trailer)
        self.v = SX.sym('v')          # velocidade longitudinal
        self.alpha = SX.sym('alpha')  # ângulo de esterçamento (delta)

        # Definição das equações de estado
        # https://doi.org/10.1007/s40747-021-00330-z
        self.dx1dt = self.v * cos(self.theta1)
        self.dy1dt = self.v * sin(self.theta1)
        self.dtheta1dt = (self.v / self.D) * tan(self.alpha)
        # Equação ajustada de beta incluindo o offset da quinta-roda a uma distância a = self.quinta_roda

        self.dbetadt = - (self.v/self.L) * sin(self.beta) \
                    - (self.v/self.D) * tan(self.alpha) \
                    + (self.quinta_roda * self.v / (self.L * self.D)) * tan(self.alpha) * cos(self.beta)


        # Vetores de estados e controles para integrador
        self.estados = vertcat(self.x1, self.y1, self.theta1, self.beta)
        self.controles = vertcat(self.v, self.alpha)
        self.ode = vertcat(self.dx1dt, self.dy1dt, self.dtheta1dt, self.dbetadt)

        # Configuração do integrador (CVODES)
        self.dae = {'x': self.estados, 'p': self.controles, 'ode': self.ode}
        self.integrator = integrator('integrator', 'cvodes', self.dae, {'tf': self.dt})

    def atualizar_estado(self, estado: Estado4, controle: tuple) -> Estado4:
        
        try:
            estado_dm = DM([estado.x, estado.y, estado.theta, estado.beta])
            controle_dm = DM([controle[0], controle[1]])
            res = self.integrator(x0=estado_dm, p=controle_dm)       
            novo = res['xf'].full().flatten()
            
            return Estado4(*novo.tolist())
        except Exception as e:
            print(f"Erro ao atualizar o estado: {e}")
            raise e

    def atualizar_estado_caminho(self, estado: Estado4, controle: tuple) -> (list):
        try:
            caminho = []
            estado_dm = DM([estado.x, estado.y, estado.theta, estado.beta])
            controle_dm = DM([controle[0], controle[1]])
            for _ in range(self.num_passos):
                res = self.integrator(x0=estado_dm, p=controle_dm)
                estado_dm = res['xf']    
                caminho.append(res['xf'].full().flatten().tolist())
            novo = estado_dm.full().flatten()
            return Estado4(*novo.tolist()), caminho
        except Exception as e:
            print(f"Erro ao atualizar o caminho: {e}")
            raise e

    def obter_estado7(self, estado: Estado4) -> Estado7:
        
        try:
            theta2 = self.angulos.soma_angulo_normalizado(estado.theta, estado.beta)
            x2 = (estado.x + self.quinta_roda * cos(estado.theta)) - self.comprimento_trailer * cos(theta2)
            y2 = (estado.y + self.quinta_roda * sin(estado.theta)) - self.comprimento_trailer * sin(theta2)
            return Estado7(estado.x, estado.y, estado.theta, x2, y2, theta2, estado.beta)
        except Exception as e:
            print(f"Erro ao converter estado: {e}")
    
    def obter_estado5(self, estado_4: Estado4) -> Estado5:
        try:
            theta2 = self.angulos.soma_angulo_normalizado(estado_4.theta, estado_4.beta)
            # x, y, theta, beta, theta2,
            return Estado5(estado_4.x, estado_4.y, estado_4.theta, estado_4.beta, theta2)
        except Exception as e:
            print(f"Erro ao converter estado: {e}")
            
    def obter_dt(self):
        return self.dt

class Trator():
    def __init__(self, parametros: dict, dt: float):
        """
        Modelo cinemático do truck-trailer em estado reduzido [x1, y1, theta1, beta]
        Equações de movimento de acordo com o problema de estacionamento:
          ẋ = v·cos(theta1)
          ẏ = v·sin(theta1)
          thetȧ = (v/D)·tan(alpha)
          betȧ = - (v/L)·sin(beta) - (v/D)·tan(alpha)
        Args:
            parametros (dict): parâmetros do modelo (metros)
                - 'comprimento_entre_eixos_trator' = D (entre-eixos do caminhão)
                - 'distancia_eixo_traseiro_trailer_quinta_roda' = L (entre-eixos do trailer)
                - 'distancia_eixo_traseiro_quinta_roda' = distância da quinta roda ao eixo traseiro do trator
            dt (float): passo de integração (s)
        """
        # Parâmetros físicos
        self.comprimento_trator = parametros.comprimento_entre_eixos_trator
        self.angulos = Angulos()

        # Notação simplificada
        self.D = self.comprimento_trator
        self.dt = dt

        # Variáveis simbólicas de estado e controle
        self.x1 = SX.sym('x1')       # posição x do truck
        self.y1 = SX.sym('y1')       # posição y do truck
        self.theta1 = SX.sym('theta1') # orientação do truck
        self.v = SX.sym('v')          # velocidade longitudinal
        self.alpha = SX.sym('alpha')  # ângulo de esterçamento (delta)

        # Definição das equações de estado
        # https://doi.org/10.1007/s40747-021-00330-z
        self.dx1dt = self.v * cos(self.theta1)
        self.dy1dt = self.v * sin(self.theta1)
        self.dtheta1dt = (self.v / self.D) * tan(self.alpha)
        # Equação ajustada de beta incluindo o offset da quinta-roda a uma distância a = self.quinta_roda


        # Vetores de estados e controles para integrador
        self.estados = vertcat(self.x1, self.y1, self.theta1)
        self.controles = vertcat(self.v, self.alpha)
        self.ode = vertcat(self.dx1dt, self.dy1dt, self.dtheta1dt)

        # Configuração do integrador (CVODES)
        self.dae = {'x': self.estados, 'p': self.controles, 'ode': self.ode}
        self.integrator = integrator('integrator', 'cvodes', self.dae, {'tf': self.dt})

    def atualizar_estado(self, estado: Estado3, controle: tuple) -> Estado3:
        
        try:
            estado_dm = DM([estado.x, estado.y, estado.theta])
            controle_dm = DM([controle[0], controle[1]])
            res = self.integrator(x0=estado_dm, p=controle_dm)       
            novo = res['xf'].full().flatten()
            
            return Estado3(*novo.tolist())
        except Exception as e:
            print(f"Erro ao atualizar o estado: {e}")
            raise e

    def atualizar_estado_caminho(self, estado: Estado3, controle: tuple) -> (list):
        try:
            caminho = []
            estado_dm = DM([estado.x, estado.y, estado.theta])
            controle_dm = DM([controle[0], controle[1]])
            for _ in range(self.num_passos):
                res = self.integrator(x0=estado_dm, p=controle_dm)
                estado_dm = res['xf']    
                caminho.append(res['xf'].full().flatten().tolist())
            novo = estado_dm.full().flatten()
            return Estado4(*novo.tolist()), caminho
        except Exception as e:
            print(f"Erro ao atualizar o caminho: {e}")
            raise e

            
    def obter_dt(self):
        return self.dt
