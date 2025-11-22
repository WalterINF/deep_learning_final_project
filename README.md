# deep_learning_final_project
Repositório do projeto final de AGL10225 -  Aprendizado Por Reforço

## objetivo

Este projeto tem por objetivo implementar um algoritmo de aprendizado por reforço para um problema de planejamento de rotas de veículos articulados em espaço de estacionamento.

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


### Ambiente 

O ambiente consiste em um espaço de estacionamento retangular de 90x90 metros (8100 m^2) contendo vagas de estacionamento.
O agente deve posicionar o trailer sobre o objetivo, que é uma vaga de estacionamento, enquanto evita passar por cima das demais vagas.

### Agente:

#### Espaço de observação (o que o agente observa)

Estado: [theta, beta, alpha*, [r1, r2, ..., r14], [c1, c2, ..., c14], goal_proximity, goal_direction], onde:
* theta: ângulo de orientação do veiculo.
* beta: ângulo relativo entre o trator e o trailer.
* alpha*: ângulo de esterçamento do trator.
* r1, ..., r14: distâncias dos raycasts posicionados com origem no veiculo.
* c1, ..., c14: classes dos objetos detectados pelos raycasts (parede, vaga...)
* goal_proximity: proximidade do veículo ao objetivo, calculada como 1 / (1 + distância euclidiana).
* goal_direction: direção até o objetivo em radianos.
* angle_diff: diferença entre a orientação davaga de estacionamento e a orientação do veículo em radianos.
    
![veiculo com raycasts](veiculo_com_os_raycast.png)


#### Espaço de ação (o que o agente pode fazer)

Controle: [v, alpha], onde:
* v: velocidade do veiculo.	
* alpha: ângulo de esterçamento do trator.

#### Função de recompensa (o que o agente recebe)
* +50 por compleção do objetivo (estacionar na vaga de destino)
* +50 por alinhar o veículo na vaga corretamente ao estacionar
* -100 por colisão com obstáculos - incluindo outras vagas de estacionamento que não sejam a de origem ou destino - ou com o próprio veículo
* -100 por jackknife
* -50 por esgotar o tempo limite do episódio, divido entre os passos de tempo: (-20/MAX_STEPS) por passo
* -1*PUNISHMENT_TIME por passo de tempo por velocidade zero
* -3*PUNISHMENT_TIME por passo de tempo por invadir uma vaga

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
distancia minima para considerar o veículo estacionado: 2 metros.









