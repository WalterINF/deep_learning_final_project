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


### Ambiente 

O ambiente consiste em um espaço de estacionamento retangular de 90x90 metros (8100 m^2) contendo vagas de estacionamento.
O agente deve posicionar o trailer sobre o objetivo, que é uma vaga de estacionamento, enquanto evita passar por cima das demais vagas.

### Agente:

#### Espaço de observação (o que o agente observa)

Estado: [theta, beta, alpha*, [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14], goal_distance, goal_direction], onde:
* theta: ângulo de orientação do veiculo.
* beta: ângulo relativo entre o trator e o trailer.
* alpha*: ângulo de esterçamento do trator.
* r1, ..., r14: distâncias dos raycasts posicionados com origem no veiculo.
* goal_distance: distância euclidiana do veiculo ao objetivo.
* goal_direction: ângulo de orientação do veiculo em relação ao objetivo em radianos.
    
![veiculo com raycasts](veiculo_com_os_raycast.png)


#### Espaço de ação (o que o agente pode fazer)

Controle: [v, alpha], onde:
* v: velocidade do veiculo.	
* alpha: ângulo de esterçamento do trator.

#### Função de recompensa (o que o agente recebe)
* +100 por compleção do objetivo (estacionar na vaga de destino)
* -150 por colisão com obstáculos - incluindo outras vagas de estacionamento que não sejam a de origem ou destino - ou com o próprio veículo
* -150 por jackknife
* -0.01 por passo de tempo
* -0.05 por velocidade zero por passo de tempo

#### critérios de parada
* colisão com obstáculos - incluindo outras vagas de estacionamento que não sejam a de origem ou destino - com o próprio veículo ou paredes do ambiente
* tempo: 2 minutos
* objetivo atingido: veiculo estacionado no ponto de destino.


### Parâmetros de simulação (configurações do ambiente)

limites de angulo de esterçamento do trator: +-28 graus.
limites de taxa de esterçamento do trator: +-10 graus/s.
limites de velocidade do trator: 20km/h ou 5m/s.
angulo de canivete: 65 graus.
limite de visão do sensor de distância: 20 metros.


### Próximos passos

- [ ] Implementar a diferenciação de objetos no sensor de distância (mudar o espaço de observação)








