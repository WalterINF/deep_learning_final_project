# deep_learning_final_project
Repositório do projeto final de AGL10225 -  Aprendizado Por Reforço





##### Agente:

#### Espaço de observação (o que o agente observa)

Estado: [x, y, theta, beta, alpha*, [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14]]
    
![veiculo com raycasts](veiculo_com_os_raycast.png)

x, y: posição do veiculo no mapa.
theta: ângulo de orientação do veiculo.
beta: ângulo relativo entre o veiculo e o trailer.
rn: raycasts posicionados com origem no veiculo.


#### Espaço de ação (o que o agente pode fazer)


#### Função de recompensa (o que o agente recebe)
proporcional ao angulo de esterçamento do trator e velocidade do trator.
- (talvez) punir o trator por esterçar as rodas enquanto estacionário.
- 1 por compleção do objetivo, -1 por parada precoce (ver critérios de parada)

#### critérios de parada
- colisão com obstáculos ou com o próprio veículo
- tempo: 5 minutos
- esterçamento muito brusco: > 10 graus/s.
- objetivo atingido: veiculo estacionado no ponto de destino.

#### Política (o que o agente faz)


#### Valor (o que o agente aprende)

#### Ambiente (o que o agente interage)

#### Parâmetros de simulação (configurações do ambiente)

limites de angulo de esterçamento do trator: +-28 graus.
limites de taxa de esterçamento do trator: +-10 graus/s.
limites de velocidade do trator: 20km/h ou 5m/s.
angulo de canivete: 65 graus.
limite de visão do sensor de distância: 150 metros.



//todo:
definir a equação de atualização dos estados do veículo: taxa de variação do angulo de esterçamento do trator.










