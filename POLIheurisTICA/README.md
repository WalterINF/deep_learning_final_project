
# 🚛 RL-Fast: Autonomous Articulated Parking Agent

**Versão:** 1.0 (Fast Track)
**Status:** Otimizado para Treinamento Acelerado (\< 48h)

Este repositório contém uma implementação de alta performance para o treinamento de agentes de Aprendizado por Reforço (RL) em tarefas de estacionamento de veículos articulados.

O módulo `rl_fast` foi desenhado para desacoplar a simulação física do código legado (CasADi), utilizando **NumPy** puro e **Integração Runge-Kutta 4 (RK4)** para maximizar o FPS (Frames Per Second) e permitir paralelismo massivo na CPU durante a coleta de dados.

-----

## 📂 Estrutura do Projeto

O projeto adota uma arquitetura de "Sidecar", onde o RL roda em um módulo isolado que consome apenas os dados de configuração do sistema legado.

```text
projeto/
│
├── config/                  # [LEGADO/LEITURA] Arquivos JSON de geometria e parâmetros
│   └── lista_veiculos.json
│
├── esqueleto/               # [LEGADO/LEITURA] Loaders de configuração
│   └── ...
│
├── rl_fast/                 # <--- CORE DO TREINAMENTO (Novo)
│   ├── __init__.py
│   ├── kin_model.py         # Física do veículo (NumPy + RK4)
│   ├── fast_sim.py          # Gerenciador de colisão (SAT), mapa e sensores
│   ├── fast_env.py          # Ambiente Gymnasium (Lógica de Recompensa e Estados)
│   └── train.py             # Script de treinamento (SAC + SB3 + GPU)
│
└── ...
```

-----

## ⚙️ Dependências

  * **Python 3.8+**
  * **NumPy:** Computação vetorial e física.
  * **Gymnasium:** Interface padrão de ambiente RL.
  * **Stable-Baselines3:** Implementação do algoritmo SAC.
  * **PyTorch:** Backend de aprendizado profundo (com suporte a CUDA).
  * **Pygame:** Visualização leve (opcional para render).

<!-- end list -->

```bash
pip install gymnasium stable-baselines3[extra] numpy pygame torch
```

-----

## 🧠 Arquitetura Técnica

### 1\. Modelo Cinemático (`kin_model.py`)

Implementação vetorizada do modelo bicicleta para veículos articulados.

  * **Integrador:** Runge-Kutta 4ª Ordem (RK4).
  * **Estado:** Array `[x, y, theta_trator, beta]`.
  * **Performance:** \~100x mais rápido que integradores simbólicos (CasADi) para *steps* discretos de RL.

### 2\. Simulação e Colisão (`fast_sim.py`)

  * **Map Baking:** Na inicialização, todas as paredes estáticas (`MapEntity`) são convertidas para uma matriz NumPy `(N, 5)` fixa.
  * **Colisão em Duas Fases:**
    1.  **Broad-Phase:** Filtra paredes fora de um raio de 25m do veículo.
    2.  **Narrow-Phase:** Aplica o **Teorema do Eixo Separador (SAT)** otimizado, sem instanciar objetos Python a cada frame.
  * **Sensores:** 14 Raycasts virtuais (LiDAR) calculados via geometria analítica.

### 3\. Ambiente e Recompensas (`fast_env.py`)

#### Espaço de Observação (21 Dimensões)

Todos os valores são normalizados (aprox. entre -1 e 1 ou 0 e 1).

| Índice | Descrição | Detalhe |
| :--- | :--- | :--- |
| `0` | **Erro Paralelo** | Distância longitudinal ao centro da vaga. |
| `1` | **Erro Perpendicular** | Distância lateral ao centro da vaga. |
| `2` | **Erro $\theta_1$** | Diferença angular Trator vs Vaga. |
| `3` | **Erro $\theta_2$** | Diferença angular Trailer vs Vaga (Crítico). |
| `4` | **Dijkstra** | Distância de navegação (Pathfinding) até o alvo. |
| `5` | **Velocidade** | $v$ atual normalizado. |
| `6` | **Articulação ($\beta$)** | Ângulo entre trator e trailer. |
| `7-20` | **Sensores** | 14 leituras de raycast (proximidade de obstáculos). |

#### Função de Recompensa (Reward Shaping)

A função é projetada para convergência rápida (Dense Reward).

$$R_{total} = R_{terminal} + R_{shaping} + R_{penalties}$$

  * **Estados Terminais (Fim de Episódio):**

      * ✅ **Sucesso:** `+100` (Critérios: $dist < 0.2m$, $\theta_2 < 0.1rad$, $v \approx 0$).
      * ❌ **Colisão:** `-100` (Parede ou Obstáculo).
      * ❌ **Overshoot:** `-100` (Fundo da vaga tratado como parede virtual).
      * ❌ **Jackknife:** `-100` (Se $|\beta| > 45^\circ$).

  * **Shaping (Incentivo Contínuo):**

      * **Navegação:** `(Dijkstra_Antigo - Dijkstra_Novo) * 10`. (Ganha pontos por se aproximar pelo caminho certo).
      * **Alinhamento Fino:**
        $$R_{align} = 0.5 \cdot (1 - \frac{|e_{\theta 2}|}{\pi}) + 0.5 \cdot (1 - \frac{|e_{perp}|}{L_{vaga}})$$

  * **Penalidades (Custos):**

      * **Troca de Sentido (Histerese):** `-1.0` se trocar de sentido (Frente/Ré) **mais de 1 vez** em um deslocamento menor que **60 metros**.
      * **Tempo:** `-0.01` por step.
      * **Restrição de Articulação:** `-0.02` se $|\beta| > 25^\circ$.

-----

## 🚀 Estratégia de Treinamento (Hardware Híbrido)

Para cumprir o prazo de 2 dias, o script `train.py` utiliza uma estratégia híbrida:

1.  **CPU (Simulação):** O `SubprocVecEnv` cria **8 a 16 processos** independentes. Cada um roda uma instância leve (`numpy`) da simulação. Isso satura a CPU com geração de dados.
2.  **GPU (Aprendizado):** O algoritmo SAC roda na **NVIDIA RTX 2000 Ada**.
      * **Batch Size:** `2048` ou `4096` (Maximizando o throughput da GPU).
      * **Buffer:** `1_000_000` transições.

### Como Executar

**1. Configuração**
Certifique-se de que o arquivo `config/lista_veiculos.json` está acessível.

**2. Iniciar Treinamento**
Execute como módulo para garantir que os imports funcionem:

```bash
python -m rl_fast.train
```

**3. Monitoramento**
Acompanhe o progresso via TensorBoard:

```bash
tensorboard --logdir rl_fast/logs
```

-----

## 📝 Notas de Desenvolvimento (Checklist Rápido)

  * [x] **Física:** Substituição do CasADi por NumPy RK4.
  * [x] **Colisão:** Implementação de `check_collision_fast` com OBB+SAT.
  * [x] **Ambiente:** Implementação da lógica de histerese (60m) na troca de marcha.
  * [x] **Recompensas:** Escalonamento de magnitude (±100) para gradientes fortes.
  * [x] **Overshoot:** Integrado à lógica de colisão (parede de fundo).

-----

> **Aviso:** Este arcabouço ignora propositalmente as classes `Estado4`, `Estado5` e interfaces complexas do diretório `dominio/` durante o loop de RL para garantir velocidade. Apenas os dados geométricos são importados.

Este checklist detalhado foi projetado para servir como seu **guia de implementação passo-a-passo**. Ele contém as assinaturas exatas, tipos de dados e lógica interna crítica para cada arquivo.

-----

### 📂 1. Módulo de Física (`rl_fast/kin_model.py`)

**Objetivo:** Evolução de estado rápida e determinística.

  * [ ] **Classe `KinematicModel`**
    ```python
    class KinematicModel:
        def __init__(self, geometry_dict: dict, dt: float = 0.2):
            """
            Carrega L (trailer), D (trator), offsets e pré-calcula inversos (1/L, 1/D)
            para evitar divisões no loop.
            """
            pass

        def step(self, state: np.ndarray, control: tuple) -> np.ndarray:
            """
            Aplica RK4.
            Args:
                state: np.array([x, y, theta, beta], dtype=float32)
                control: tuple(velocidade, steering_angle)
            Returns:
                next_state: np.array([x, y, theta, beta]) (Normalizado -pi a pi)
            """
            pass

        def _derivatives(self, state: np.ndarray, control: tuple) -> np.ndarray:
            """
            Calcula [dx, dy, dtheta, dbeta].
            Lógica:
                dx = v * cos(theta)
                dy = v * sin(theta)
                dtheta = v/D * tan(alpha)
                dbeta = ... (incluindo offset da quinta roda)
            """
            pass

        @staticmethod
        def _normalize_angle(angle: float) -> float:
            """Garante intervalo [-pi, pi]"""
            pass
    ```

-----

### 📂 2. Módulo de Simulação (`rl_fast/fast_sim.py`)

**Objetivo:** Gerenciar o mundo, assar (bake) o mapa e detectar colisões.

  * [ ] **Classe `FastSimulation`**
    ```python
    class FastSimulation:
        def __init__(self):
            """
            1. Carrega SimulationConfigLoader.
            2. Extrai paredes para self.walls_data (np.array N x 5).
            3. Instancia self.model = KinematicModel(...).
            4. Define self.state = np.zeros(4).
            """
            pass

        def reset(self) -> np.ndarray:
            """
            Sorteia posição inicial válida longe de obstáculos.
            Zera self.state.
            Returns: self.state
            """
            pass

        def step(self, action: np.ndarray) -> dict:
            """
            1. self.state = self.model.step(self.state, action)
            2. Verifica colisão (self.check_collision_fast).
            3. Calcula raycasts.
            Returns: {
                'state': np.array,
                'collision': bool,
                'jackknife': bool, # abs(beta) > 45 graus
                'raycasts': np.array (14,)
            }
            """
            pass

        def check_collision_fast(self, state: np.ndarray) -> bool:
            """
            1. Broad-Phase: Filtra paredes onde dist^2 > 25^2.
            2. Narrow-Phase: Chama _sat_collision para paredes restantes.
            3. Checa 'Overshoot': Verifica se o eixo traseiro do trailer passou do fundo da vaga.
            """
            pass

        def _get_vehicle_corners(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """
            Calcula os 4 cantos (OBB) do Trator e do Trailer baseados no estado atual.
            Usa geometria vetorial (sem classes BoundingBox lentas).
            """
            pass

        def _sat_collision(self, poly1: np.ndarray, poly2: np.ndarray) -> bool:
            """Implementação vetorial do Teorema do Eixo Separador."""
            pass

        def _get_raycasts(self, state: np.ndarray) -> np.ndarray:
            """
            Calcula intersecção de 14 vetores com self.walls_data.
            Returns: Array normalizado [0.0, 1.0].
            """
            pass
    ```

-----

### 📂 3. Módulo de Ambiente (`rl_fast/fast_env.py`) - **CRÍTICO**

**Objetivo:** Lógica de negócio, Histerese de Marcha e Recompensas.

  * [ ] **Classe `FastParkingEnv(gym.Env)`**
    ```python
    class FastParkingEnv(gym.Env):
        def __init__(self):
            self.sim = FastSimulation()
            # Definir observation_space (Box 21,) e action_space (Box 2,)
            
            # Variáveis de Histerese de Marcha
            self.last_gear_sign = 0      # 1 (Frente), -1 (Ré)
            self.dist_since_shift = 0.0  # Odômetro parcial
            self.shift_count = 0         # Contador de trocas na janela curta
            pass

        def reset(self, seed=None, options=None):
            # Reseta simulação e variáveis de histerese
            self.last_gear_sign = 0
            self.dist_since_shift = 0.0
            self.shift_count = 0
            self.last_dijkstra = self._get_dijkstra_dist()
            pass

        def step(self, action: np.ndarray):
            # 1. Simula
            sim_result = self.sim.step(action)
            
            # 2. Lógica de Histerese (Troca de Sentido)
            vel = action[0]
            current_sign = np.sign(vel) if abs(vel) > 0.01 else self.last_gear_sign
            penalty_gear = 0.0
            
            dist_step = np.linalg.norm(...) # Distância percorrida neste frame
            
            if current_sign != self.last_gear_sign and current_sign != 0:
                # Ocorreu troca
                if self.dist_since_shift < 60.0:
                    self.shift_count += 1
                    if self.shift_count > 1:
                         penalty_gear = -1.0 # <--- PENALIDADE APLICADA
                else:
                    self.shift_count = 1 # Reset, nova janela
                
                self.dist_since_shift = 0.0
                self.last_gear_sign = current_sign
            else:
                self.dist_since_shift += dist_step

            # 3. Monta Observação e Calcula Recompensa
            obs = self._get_obs(sim_result)
            reward, terminated, truncated = self._calculate_reward(sim_result, penalty_gear)
            
            return obs, reward, terminated, truncated, {}

        def _calculate_reward(self, sim_data, penalty_gear) -> tuple[float, bool, bool]:
            """
            R_total = 0
            
            # Terminais
            if sim_data['collision'] or sim_data['jackknife']:
                return -100.0, True, False
            if success_condition (e_par < 0.2, e_perp < 0.2, e_theta2 < 0.1, v < 0.1):
                return +100.0, True, False
            
            # Shaping
            R_dijkstra = (self.last_dijkstra - curr_dijkstra) * 10.0
            R_align = 0.5 * (1 - abs(e_theta2)/pi) + 0.5 * (1 - abs(e_perp)/2.0)
            
            # Custos
            R_time = -0.01
            R_beta = -0.02 if abs(beta) > 25deg else 0.0
            R_nav = -0.001 * abs(v) # Custo de movimento/combustível
            
            total = R_dijkstra + R_align + R_time + R_beta + R_nav + penalty_gear
            return total, False, timeout_check
            """
            pass

        def _get_obs(self, sim_data) -> np.ndarray:
            """
            Calcula erros (e_par, e_perp, e_thetas) transformando coordenadas globais
            para o frame local da vaga alvo.
            Concatena com [dijkstra, v, beta, raycasts].
            """
            pass
    ```

-----

### 📂 4. Script de Treinamento (`rl_fast/train.py`)

**Objetivo:** Orquestrar CPU e GPU.

  * [ ] **Função `make_env(rank, seed)`**

    ```python
    def make_env(seed=0):
        def _init():
            env = FastParkingEnv()
            env.reset(seed=seed)
            return env
        return _init
    ```

  * [ ] **Bloco `if __name__ == "__main__":`**

    ```python
    if __name__ == "__main__":
        # Hardware Check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Parallel Environments (CPU Saturation)
        num_cpu = 8 # Ajustar conforme sua máquina
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        
        # Model Configuration (GPU Throughput)
        model = SAC(
            "MlpPolicy",
            env,
            device=device,
            batch_size=2048,      # Batch grande para GPU
            buffer_size=1_000_000,
            learning_starts=10_000,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            ent_coef="auto",
            verbose=1
        )
        
        # Run
        model.learn(total_timesteps=5_000_000)
        model.save("sac_fast_parking_final")
    ```

-----

### 🔍 Checklist de Verificação Final (Pré-Execução)

1.  **Física:** O ângulo `beta` e `theta` estão sendo normalizados para `[-pi, pi]` a cada passo no `kin_model`? (Essencial para a rede neural não se perder).
2.  **Colisão:** O "Overshoot" está implementado no `fast_sim.py`? (Considere colocar uma parede virtual atrás da vaga no array `walls_data` durante o `__init__`, é a forma mais fácil).
3.  **Histerese:** A variável `dist_since_shift` está sendo resetada corretamente **apenas** quando ocorre a troca?
4.  **Performance:** O `SimulationConfigLoader` está sendo instanciado **apenas uma vez** no `__init__` do ambiente (ou da simulação), e não a cada passo? (Carregar JSONs é lento).
5.  **Caminhos:** O `train.py` está sendo rodado como módulo (`python -m rl_fast.train`) para enxergar a pasta `config/` na raiz?