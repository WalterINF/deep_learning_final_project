# deep_learning_final_project
Repositório do projeto final de AGL10225 -  Aprendizado Por Reforço

## objetivo

Este projeto tem por objetivo avaliar diferentes heurísticas aliadas ao HuRL - Heuristic Reinforcement Learning, para treinar um agente para planejamento de rotas de veículos articulados em espaço de estacionamento.

## introdução
O planejamento de movimento para veículos articulados (ex: uniciclo com reboque) em ambientes não estruturados enfrenta o desafio da complexidade cinemática local e topológica global.

Controladores baseados em otimização (MPC) resolvam a estabilização local, mas sofrem em navegação global com obstáculos (mínimos locais). 

Abordagens de Aprendizado por Reforço (RL) prometem robustez global, mas sofrem com o problema do horizonte longo

Nesse contexto, exploramos o potencial do Heuristic Guided Reinforcement Learning (HuRL)para aumentar a eficiência e velocidade treinamento de agente de RL em espaço de estacionamento.

## Metodologia
A metodologia consiste em três pilares: a modelagem do domínio físico, a definição da heurística física para reward shaping e a integração no framework HuRL.

## HuRL

Cheng et al,  introduziram o Heuristic-Guided Reinforcement Learning (HuRL) como uma solução para acelerar o RL. O HuRL utiliza uma heurística (mesmo que imperfeita) para "encurtar" o horizonte efetivo do problema através da moldagem de recompensa (reward shaping) baseada em potencial. No entanto, a eficácia do HuRL depende criticamente da qualidade desta heurística inicial.

Nesse contexto, exploramos diferentes heurísticas para modelar a recompensa do agente (reward shaping), buscando aumentar a eficiência e rapidez do aprendizado através do aumento da densidade de recompensa.

## Modelagem do domínio

### Veículo

Representamos o veículo como um sistema cinemático de bicicleta articulado com três componentes principais:
Trator: posição (x, y), orientação θ, eixo traseiro como referência
Trailer: acoplado ao trator pela quinta roda, orientação θ_trailer = θ - β
Articulação: ângulo β entre trator e trailer

#### Equações Cinemáticas - Modelo Bicicleta Cinético

O sistema é não-holonômico, o que significa que o veículo não pode se mover instantaneamente em qualquer direção (ele não pode andar de lado, por exemplo).

Para a simulação do movimento do veículo articulado durante a expansão de nós, foi adotado o modelo cinemático descrito por Guan e Jiang~\cite{guan2022tractor}. O modelo considera o trator e o reboque como sistemas rigidamente acoplados, com controles baseados em velocidade longitudinal $v$ e ângulo de esterçamento $\alpha$. As equações diferenciais que regem a evolução dos estados são:
\begin{equation}
\begin{split}
    \dot{x} &= v \cdot \cos(\theta_1) \\
    \dot{y} &= v \cdot \sin(\theta_1) \\
    \dot{\theta}_1 &= \frac{v}{D} \cdot \tan(\alpha) \\
    \dot{\beta} &= -\frac{v}{L} \cdot \sin(\beta) - \frac{v}{D} \cdot \tan(\alpha)
\end{split}
\end{equation}

### Ambiente físico

#### Mapa

Pares de fileiras de vagas apontando em direções opostas, com paredes entre elas

#### Dimensões

150x150 metros (22500 m^2)

#### Randomização de domínio 

A cada geração, uma vaga aleatória é escolhida como ponto de partida e outra é escolhida como ponto de chegada (alvo)
Quaisquer outras vagas tem uma chance de 25% de possuir um veículo estacionado sobre elas (obstáculo)

## tecnologias utilizadas
- Python
- NumPy
- Jupyter Notebook
- Stable-Baselines3
- TensorBoard
- OpenAI Gymnasium
- Pytorch
- Pytest
- Matplotlib
- NumPy
- Casadi

### Estrutura do repositório
├── codigos
├── logs: diretório para logs do TensorBoard
├── models: diretório para modelos treinados
├── README.md: este arquivo
├── requirements.txt: arquivo de dependências do projeto
├── scripts: diretório para scripts auxiliares uteis para treinamento
│   ├── clear_logs_models.py: script para limpar logs e modelos treinados
│   └── launch_tensorboard.py: script para iniciar o TensorBoard
├── src: diretório para códigos do projeto final de AGL10225 - Aprendizado Por Reforço
│   ├── ParkingEnv.py: classe do ambiente de simulação
│   ├── ParkingVehicle.ipynb: notebook de treinamento e teste do agente
│   ├── test: diretório para testes unitários
│   │   ├── conftest.py: arquivo de configuração para os testes
│   │   ├── test_collisions.py: teste de colisões
│   │   ├── test_env.py: teste do ambiente
│   │   └── test_map_generation.py: teste de geração do mapa (gera imagens de 3 mapas diferentes)
│   ├── train.py: script para treinar o modelo
│   ├── video: diretório para vídeos de avaliação do agente
│   └── Visualization.py: classe para renderização das simulações (e gravação de vídeos)
└── veiculo_com_os_raycast.png: imagem do veículo com os raycasts


### Ambiente 

O ambiente consiste em um espaço de estacionamento retangular de 150x150 metros (22500 m^2) contendo vagas de estacionamento.
O agente deve posicionar o trailer sobre o objetivo, que é uma vaga de estacionamento, enquanto evita passar por cima das demais vagas.

### Agente:

#### Espaço de observação (o que o agente observa)

Estado: [velocity, theta, beta, alpha*, [r1, r2, ..., r14], goal_proximity, goal_direction_relative, tractor_angle_diff, trailer_angle_diff], onde:
* velocity: velocidade atual do veículo normalizada [-1, 1].
* theta: ângulo de orientação do trator.
* beta: ângulo relativo entre o trator e o trailer.
* alpha*: ângulo de esterçamento do trator.
* r1, ..., r14: distâncias dos raycasts posicionados com origem no veiculo (normalizadas 0-1).
* goal_proximity: proximidade do veículo ao objetivo, calculada como 1 / (1 + distância euclidiana).
* goal_direction_relative: direção até o objetivo RELATIVA ao heading do veículo (egocêntrica).
* tractor_angle_diff: diferença entre a orientação do trator e a orientação da vaga.
* trailer_angle_diff: diferença entre a orientação do TRAILER e a orientação da vaga (crítico para estacionamento).
    
![veiculo com raycasts](veiculo_com_os_raycast.png)


#### Espaço de ação (o que o agente pode fazer)

Controle: [v, alpha], onde:
* v: velocidade do veiculo.	
* alpha: ângulo de esterçamento do trator.

#### Função de recompensa (o que o agente recebe)
* +100 por compleção do objetivo (estacionar na vaga de destino)
* +100 por alinhar o trailer na vaga corretamente ao estacionar (baseado na orientação do trailer)
* +1.0 por progresso até o objetivo (definido pela heurística)
* -100 por colisão com paredes ou outras vagas de estacionamento
* -100 por jackknife
* -20 distribuídos ao longo do episódio como penalidade por tempo: (-10/MAX_STEPS) por passo
* -0.1 por velocidade zero (penalidade por ficar parado)
* -0.02 por mudança brusca de esterçamento (smoothness penalty)

#### critérios de parada
* colisão com obstáculos - incluindo outras vagas de estacionamento que não sejam a de origem ou destino - com o próprio veículo ou paredes do ambiente
* tempo: 90 segundos
* objetivo atingido: veiculo estacionado no ponto de destino.
* episódio terminado: tempo limite atingido.

### Parâmetros de simulação (configurações do ambiente)

limites de angulo de esterçamento do trator: +-28 graus.
limites de taxa de esterçamento do trator: +-10 graus/s.
limites de velocidade do trator: 20km/h ou 5m/s.
angulo de canivete: 65 graus.
limite de visão do sensor de distância: 20 metros.
tempo limite do episódio: 90 segundos.
passo de tempo: 0.2 segundos.
distancia minima para considerar o veículo estacionado: 2 metros do centro da vaga de estacionamento.

### Heurísticas avaliadas

1. Heurística Euclidiana
Usamos a distância euclidiana do veículo até a vaga destino para fazer o reward shaping. O agente é recompensado por reduzir essa distância a cada passo.
2. BFS
Primeiramente, discretizamos o mapa em pixels de 1m e usamos o BFS (Breadth-first-search) a partir do alvo para computar um mapa de distâncias considerando todos os obstáculos do ambiente. A recompensa do agente é então calculada consultando-se esse mapa a cada passo e computando a diferença entre a distância anterior - também pelo BFS - e a atual.


### SAC

Variação do método Actor-Critic que busca maximizar a combinação de recompensa esperada e entropia da política.

Algoritmo off-policy de entropia máxima
Off-policy
Model-free
Estado-da-arte para tarefas de robótica com controle contínuo

Definimos a mesma arquitetura de rede para o ator e o crítico, consistindo em uma rede de três camadas ocultas de 512, 256 e 256 neurônios, respectivamente.

Taxa de aprendizado:  0.0003        
Tamanho do buffer: 1.000.000   
Tamanho do minibatch: 512            
gamma: 0.99                   
tau: 0.005   

### Resultados

Vantagem do BFS para o reward shaping fica evidente. O modelo que usa a heurística euclidiana estabiliza a recompensa perto de 0, indicando que aprendeu a ‘sobreviver’, mas não chegar ao objetivo de fato. Enquanto a heurística BFS premitiu ao agente aprender o caminho correto até a vaga destino.
Definimos taxa de sucesso como a frequência que o agente atinge o objetivo principal - estacionar na vaga. Novamente, BFS se sobressai com 70% de sucesso após 3.5M de passos


### Conclusão

A escolha da heurística para o reward shaping é crucial para o sucesso do aprendizado. A heurística BFS, apesar de mais complexa, se mostrou mais eficiente para o problema de estacionamento de veículos articulados.










