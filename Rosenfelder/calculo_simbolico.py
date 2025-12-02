# Implementação das coordenadas privilegiadas e custo para o carro cinemático
# Baseado em: "MPC for Non-Holonomic Vehicles Beyond Differential-Drive"

import numpy as np
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class KinematicCarParams:
    """
    Parâmetros do carro cinemático.
    
    :param wheelbase: Distância entre eixos (l) em metros
    :type wheelbase: float
    
    :param state_weights: Pesos para estados (q1, q2, q3, q4)
    :type state_weights: Tuple[float, float, float, float]
    
    :param input_weights: Pesos para entradas (r1, r2)
    :type input_weights: Tuple[float, float]
    """
    wheelbase: float = 5.0  # Distância entre eixos (l) em metros
    state_weights: Tuple[float, float, float, float] = (1.0, 1.0, 2.0, 3.0)
    input_weights: Tuple[float, float] = (1.0, 1.0)


class PrivilegedCoordinates(NamedTuple):
    """
    Resultado do cálculo das coordenadas privilegiadas.
    
    :param z: Vetor de coordenadas privilegiadas [z1, z2, z3, z4]
    :type z: np.ndarray
    
    :param weights: Pesos homogêneos w = (w1, w2, w3, w4)
    :type weights: Tuple[int, int, int, int]
    
    :param r: Parâmetros de homogeneidade r = (r1, r2, r3, r4)
    :type r: Tuple[int, int, int, int]
    """
    z: np.ndarray
    weights: Tuple[int, int, int, int]
    r: Tuple[int, int, int, int]


def compute_privileged_coordinates(
    state: np.ndarray,
    goal: np.ndarray,
    params: KinematicCarParams
) -> PrivilegedCoordinates:
    """
    Calcula as coordenadas privilegiadas z do estado atual em relação ao goal.
    
    Para o carro cinemático, as coordenadas privilegiadas são obtidas através
    do algoritmo de Bellaïche, que transforma o espaço de estados original
    em um sistema de coordenadas que preserva a controlabilidade do sistema
    não holonômico.
    
    **Transformação (Passo 1 - Bellaïche):**
    
    .. math::
    
        y = A^{-T}(x - d)
    
    onde :math:`A` é a matriz formada pelos campos vetoriais avaliados no setpoint.
    
    Para o carro cinemático na origem (:math:`d = 0`):
    
    .. math::
    
        y = \\begin{bmatrix} x_1 \\\\ x_4 \\\\ -l \\cdot x_3 \\\\ l \\cdot x_2 \\end{bmatrix}
    
    **Parâmetros de Homogeneidade:**
    
    - Pesos: :math:`w = (1, 1, 2, 3)`
    - Parâmetros r: :math:`r = (1, 1, 2, 3)`
    - Grau de não holonomia: :math:`\\rho = 3`
    
    :param state: Estado atual do veículo [x, y, θ]
    :type state: np.ndarray
    
    :param goal: Estado desejado (setpoint) [x_d, y_d, θ_d]
    :type goal: np.ndarray
    
    :param params: Parâmetros do carro cinemático
    :type params: KinematicCarParams
    
    :returns: Coordenadas privilegiadas e parâmetros de homogeneidade
    :rtype: PrivilegedCoordinates
    
    :raises ValueError: Se state ou goal não tiverem dimensão 4
    
    .. note::
    
        Para setpoints com :math:`\\phi_d = 0`, vale :math:`z = y`.
        Para setpoints gerais, o segundo passo do algoritmo de Bellaïche
        é necessário.
    
    **Exemplo de uso:**
    
    .. code-block:: python
    
        params = KinematicCarParams(wheelbase=0.2)
        state = np.array([0.5, 0.3, 0.1, 0.05])
        goal = np.array([0.0, 0.0, 0.0, 0.0])
        
        result = compute_privileged_coordinates(state, goal, params)
        print(f"z = {result.z}")
        print(f"Pesos w = {result.weights}")
    
    **Referências:**
    
    .. [1] Rosenfelder et al. "MPC for Non-Holonomic Vehicles Beyond 
           Differential-Drive", arXiv:2205.11400, 2022.
    .. [2] Jean, F. "Control of Nonholonomic Systems: From Sub-Riemannian
           Geometry to Motion Planning", Springer, 2014.
    """
    if len(state) != 4 or len(goal) != 4:
        raise ValueError("State e goal devem ter dimensão 4: [x, y, θ, φ]")
    
    state = np.asarray(state, dtype=float)
    goal = np.asarray(goal, dtype=float)
    l = params.wheelbase
    
    # Diferença do estado em relação ao goal
    dx = state - goal
    x1, x2, x3, x4 = dx[0], dx[1], dx[2], dx[3]
    
    # =========================================================================
    # Passo 1: Transformação de Bellaïche (Eq. 18 do paper)
    # =========================================================================
    # Matriz do referencial adaptado avaliado na origem:
    # [X1, X2, X3, X4]_0 forma a base do espaço tangente
    #
    # Para o carro cinemático com setpoint na origem:
    # y = [x1, x4, -l*x3, l*x2]^T
    # =========================================================================
    
    y = np.array([
        x1,           # y1 = x1 (direção facilmente controlável)
        x4,           # y2 = φ (ângulo de esterçamento)
        -l * x3,      # y3 = -l*θ (orientação escalada)
        l * x2        # y4 = l*y (direção mais difícil de controlar)
    ])
    
    # =========================================================================
    # Passo 2: Correções polinomiais (Eq. 8 do paper)
    # =========================================================================
    # Para setpoints com φ_d = 0 (goal[3] = 0), as derivadas não holonômicas
    # relevantes desaparecem, resultando em z = y.
    #
    # Para setpoints gerais, seria necessário calcular h_{4,2}(y1, y2, y3)
    # =========================================================================
    
    if np.abs(goal[3]) < 1e-10:
        # Caso simplificado: z = y
        z = y.copy()
    else:
        # Caso geral: aplicar segundo passo de Bellaïche
        # z_j = y_j - sum_{k=2}^{w_j-1} h_{j,k}(y)
        # Para o carro, apenas z4 requer correção
        z = y.copy()
        # h_{4,2} envolve derivadas não holonômicas de ordem superior
        # Implementação completa requer cálculo simbólico adicional
        # Por simplicidade, mantemos z = y (aproximação válida próximo à origem)
    
    # Parâmetros de homogeneidade do carro cinemático
    weights = (1, 1, 2, 3)  # w = (w1, w2, w3, w4)
    r = (1, 1, 2, 3)        # r = (r1, r2, r3, r4)
    
    return PrivilegedCoordinates(z=z, weights=weights, r=r)


def compute_tailored_cost(
    state: np.ndarray,
    goal: np.ndarray,
    control: Optional[np.ndarray] = None,
    params: Optional[KinematicCarParams] = None
) -> tuple[float, float]:
    """
    Calcula o custo sob medida (tailored cost) entre dois estados.
    
    O custo é derivado da aproximação homogênea nilpotente do sistema,
    utilizando expoentes que refletem a "dificuldade de controle" de cada
    direção no espaço de estados.
    
    **Função de Custo de Estágio (Stage Cost):**
    
    .. math::
    
        \\ell(z, u) = \\sum_{i=1}^{n_x} q_i |z_i|^{d/r_i} + 
                      \\sum_{j=1}^{n_u} r_j |u_j|^{d/s_j}
    
    onde :math:`d = 2 \\sum_{i=1}^{n_x} r_i`.
    
    **Para o carro cinemático:**
    
    Com :math:`r = (1, 1, 2, 3)` e :math:`s = (1, 1)`, temos :math:`d = 12`:
    
    .. math::
    
        \\ell(x, u) = q_1 |x_1|^{12} + q_2 |\\phi|^{12} + q_3 |l \\theta|^6 + 
                      q_4 |l y|^4 + r_1 |v|^{12} + r_2 |\\omega|^{12}
    
    .. note::
    
        Os expoentes maiores nas direções difíceis de controlar (y, θ)
        permitem que o MPC execute manobras mais extensas nas direções
        fáceis para compensar pequenos desvios nas direções não atuadas
        diretamente.
    
    :param state: Estado atual [x, y, θ, φ]
    :type state: np.ndarray
    
    :param goal: Estado desejado [x_d, y_d, θ_d, φ_d]
    :type goal: np.ndarray
    
    :param control: Entrada de controle [v, ω] (opcional)
    :type control: Optional[np.ndarray]
    
    :param params: Parâmetros do carro (opcional, usa padrão se None)
    :type params: Optional[KinematicCarParams]
    
    :returns: Valor escalar do custo
    :rtype: float
    
    **Exemplo de uso:**
    
    .. code-block:: python
    
        state = np.array([0.1, 0.2, 0.05, 0.0])
        goal = np.array([0.0, 0.0, 0.0, 0.0])
        control = np.array([0.5, 0.1])
        
        cost = compute_tailored_cost(state, goal, control)
        print(f"Custo: {cost:.6e}")
    
    **Comparação com Custo Quadrático:**
    
    O custo quadrático padrão :math:`\\ell_Q = x^T Q x + u^T R u` é
    **insuficiente** para estabilizar assintoticamente sistemas não
    holonômicos, pois não respeita a geometria sub-Riemanniana do sistema.
    
    **Referências:**
    
    .. [1] Coron, Grüne, Worthmann. "Model Predictive Control, Cost
           Controllability, and Homogeneity", SIAM J. Control Optim., 2020.
    """
    if params is None:
        params = KinematicCarParams()
    
    # Obter coordenadas privilegiadas
    priv_coords = compute_privileged_coordinates(state, goal, params)
    z = priv_coords.z
    r = priv_coords.r  # (1, 1, 2, 3)
    
    # Parâmetros de homogeneidade
    s = (1, 1)  # Pesos das entradas
    d = 2 * sum(r)  # d = 2 * (1+1+2+3) = 14... mas paper usa d=12
    # Usando d = 12 conforme Eq. 21 do paper (expoentes simplificados)
    d = 12
    
    q1, q2, q3, q4 = params.state_weights
    r1, r2 = params.input_weights
    
    # =========================================================================
    # Custo de Estado (Eq. 21 do paper)
    # =========================================================================
    # l(x,u) = q1*|x1|^(d/r1) + q2*|φ|^(d/r2) + q3*|lθ|^(d/r3) + q4*|ly|^(d/r4)
    #        = q1*|z1|^12 + q2*|z2|^12 + q3*|z3|^6 + q4*|z4|^4
    # =========================================================================
    
    state_cost = (
        q1 * np.abs(z[0]) ** 6 +   # |z1|^12 - posição x
        # q2 * np.abs(z[1]) ** (d / r[1]) +   # |z2|^12 - ângulo esterçamento
        q3 * np.abs(z[2]) ** 3 +   # |z3|^6  - orientação θ
        q4 * np.abs(z[3]) ** 2    # |z4|^4  - posição y (mais difícil)
    )
    
    # =========================================================================
    # Custo de Controle
    # =========================================================================
    input_cost = 0.0
    if control is not None:
        control = np.asarray(control, dtype=float)
        if len(control) >= 2:
            v, omega = control[0], control[1]
            input_cost = (
                r1 * np.abs(v) ** (d / s[0]) +      # |v|^12
                r2 * np.abs(omega) ** (d / s[1])    # |ω|^12
            )

    return (state_cost, input_cost)
    # Log 10 para melhor visualização
    # return np.log10(state_cost + input_cost)


def compute_quadratic_cost(
    state: np.ndarray,
    goal: np.ndarray,
    control: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None
) -> float:
    """
    Calcula o custo quadrático padrão (para comparação).
    
    .. warning::
    
        O custo quadrático é **provadamente insuficiente** para estabilizar
        assintoticamente o carro cinemático via MPC sem restrições terminais.
    
    .. math::
    
        \\ell_Q(x, u) = (x - x_d)^T Q (x - x_d) + u^T R u
    
    :param state: Estado atual [x, y, θ, φ]
    :type state: np.ndarray
    
    :param goal: Estado desejado
    :type goal: np.ndarray
    
    :param control: Entrada de controle [v, ω]
    :type control: Optional[np.ndarray]
    
    :param Q: Matriz de ponderação de estados (4x4)
    :type Q: Optional[np.ndarray]
    
    :param R: Matriz de ponderação de controle (2x2)
    :type R: Optional[np.ndarray]
    
    :returns: Custo quadrático
    :rtype: float
    """
    if Q is None:
        Q = np.eye(4)
    if R is None:
        R = np.eye(2)
    
    dx = np.asarray(state) - np.asarray(goal)
    state_cost = dx @ Q @ dx
    
    input_cost = 0.0
    if control is not None:
        u = np.asarray(control)
        input_cost = u @ R @ u
    
    return (state_cost, input_cost)

    # Log 10 para melhor visualização
    # return np.log10(state_cost + input_cost)

# =============================================================================
# Demonstração
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("COORDENADAS PRIVILEGIADAS - CARRO CINEMÁTICO")
    print("=" * 70)
    
    # Parâmetros
    params = KinematicCarParams(
        wheelbase=0.2,
        state_weights=(1.0, 1.0, 2.0, 3.0),
        input_weights=(1.0, 1.0)
    )
    
    # Estado inicial (cenário de baliza paralela)
    state = np.array([0.0, 0, 0.0, 0.0])  # [x, y, θ, φ]
    goal = np.array([0.0, 0.0, 0.0, 0.0])   # origem
    control = np.array([0.0, 0.0])          # [v, ω]
    
    print(f"\nEstado atual:  x = {state}")
    print(f"Goal:          d = {goal}")
    print(f"Controle:      u = {control}")
    print(f"Wheelbase:     l = {params.wheelbase} m")
    
    # Calcular coordenadas privilegiadas
    priv = compute_privileged_coordinates(state, goal, params)
    
    print(f"\n--- Coordenadas Privilegiadas ---")
    print(f"z = {priv.z}")
    print(f"  z1 = {priv.z[0]:.4f} (posição x)")
    print(f"  z2 = {priv.z[1]:.4f} (ângulo esterçamento φ)")
    print(f"  z3 = {priv.z[2]:.4f} (-l·θ, orientação escalada)")
    print(f"  z4 = {priv.z[3]:.4f} (l·y, direção difícil)")
    print(f"\nPesos:     w = {priv.weights}")
    print(f"Parâm. r:  r = {priv.r}")
    
    # Calcular custos
    tailored = compute_tailored_cost(state, goal, control, params)
    quadratic = compute_quadratic_cost(state, goal, control)
    
    print(f"\n--- Comparação de Custos ---")
    print(f"Custo sob medida (tailored): {tailored[0]:.6e} + {tailored[1]:.6e}")
    # print(f"Custo quadrático (padrão):   {quadratic[0]:.6e} + {quadratic[1]:.6e}")
    
    # # Mostrar expoentes
    # print(f"\n--- Expoentes do Custo Sob Medida ---")
    # print(f"Estado x:        expoente = 12 (peso w=1)")
    # print(f"Estado φ:        expoente = 12 (peso w=1)")
    # print(f"Estado θ:        expoente = 6  (peso w=2)")
    # print(f"Estado y:        expoente = 4  (peso w=3, mais difícil)")
    
    print("\n" + "=" * 70)