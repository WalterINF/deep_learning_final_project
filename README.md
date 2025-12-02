# deep_learning_final_project
Repositório do projeto final de AGL10225 -  Aprendizado Por Reforço

## objetivo

Este projeto tem por objetivo avaliar diferentes heurísticas aliadas ao HuRL - Heuristic Reinforcement Learning, para treinar um agente para planejamento de rotas de veículos articulados em espaço de estacionamento.

### Ambiente físico

Pares de fileiras de vagas apontando em direções opostas, com paredes entre elas

150x150 metros (22500 m^2)

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
logs: diretório para logs do TensorBoard
models: diretório para modelos treinados
README.md: este arquivo
requirements.txt: arquivo de dependências do projeto
scripts: diretório para scripts auxiliares uteis para treinamento
clear_logs_models.py: script para limpar logs e modelos treinados
launch_tensorboard.py: script para iniciar o TensorBoard
src: diretório para códigos que implementam o ambiente de simulação e o agente de aprendizado por reforço
test: diretório para testes unitários
train.py: script para treinar o modelo
video: diretório para vídeos de avaliação do agente



## Modelos treinados

| Modelo | Heurística |
SAC_Improved_v6: modelo treinado sem heurística
SAC_Improved_v8: modelo treinado com heurística euclidiana
SAC_Improved_v9: modelo treinado com heurística BFS

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
distancia minima para considerar o veículo estacionado: 1 metro do centro da vaga de estacionamento.
diferença de ângulo máxima para considerar o veículo estacionado: 5 graus.

### Heurísticas Avaliadas

Neste estudo, a eficácia do *reward shaping* foi investigada através da comparação de três abordagens distintas para a definição da função de potencial.

1. **Baseline (Sem Heurística)**
   Nesta configuração de controle, o agente é submetido a um regime de recompensa esparsa, recebendo sinais de reforço exclusivamente nos eventos terminais (sucesso ou falha) e penalidades temporais. Esta abordagem serve como linha de base para avaliar o impacto da introdução de sinais de recompensa densos providos pelas heurísticas.

2. **Heurística Euclidiana**
   Esta abordagem emprega a distância Euclidiana ($L^2$ norm) entre o centro de massa do veículo e o centróide da vaga de destino como função de potencial. O *reward shaping* é formulado para recompensar o gradiente negativo da distância, incentivando a redução da distância linear a cada passo de tempo. Embora computacionalmente eficiente, esta heurística ignora a topologia do ambiente e restrições cinemáticas, tornando-a suscetível a mínimos locais em ambientes com obstáculos não convexos.

3. **Heurística Topológica (BFS)**
   Para incorporar a geometria dos obstáculos na função de recompensa, utiliza-se o algoritmo de busca em largura (*Breadth-First Search* - BFS). O ambiente é discretizado em uma grade de ocupação com resolução de $1m \times 1m$. O mapa de distâncias geodésicas (considerando obstáculos estáticos) é computado a partir do ponto alvo. A recompensa densa é derivada da redução da distância do menor caminho válido pelo BFS, guiando o agente através de trajetórias livres de colisão e mitigando o problema de mínimos locais.

   **Heurística Não Holonômica**
   Esta abordagem emprega a distância não holonômica entre o centro de massa do veículo e o centróide da vaga de destino como função de potencial. O *reward shaping* é formulado para recompensar o gradiente negativo da distância, incentivando a redução da distância não holonômica a cada passo de tempo. Esta heurística considera a geometria do veículo e o ambiente, tornando-a mais robusta em ambientes com obstáculos não convexos.












