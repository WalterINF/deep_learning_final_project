# deep_learning_final_project
Repositório do projeto final de AGL10225 -  Aprendizado Por Reforço

## Resumo

O planejamento de movimento para veículos articulados em ambientes complexos impõe desafios significativos devido a restrições não-holonômicas e à presença de obstáculos. Este trabalho investiga a eficácia do *Heuristic Guided Reinforcement Learning* (HuRL) na aceleração do treinamento de agentes de Aprendizado por Reforço para tarefas de estacionamento. Utilizando o algoritmo *Soft Actor-Critic* (SAC), avaliamos comparativamente três estratégias de *reward shaping*: ausência de heurística (baseline), distância Euclidiana e distância topológica via *Breadth-First Search* (BFS). Os resultados demonstram que a heurística baseada em BFS, ao incorporar a geometria dos obstáculos, supera significativamente as demais abordagens, permitindo a convergência robusta e altas taxas de sucesso, enquanto a heurística Euclidiana estagna em mínimos locais e o baseline falha em aprender a tarefa.

## objetivo

Este projeto tem por objetivo avaliar diferentes heurísticas aliadas ao HuRL - Heuristic Reinforcement Learning, para treinar um agente para planejamento de rotas de veículos articulados em espaço de estacionamento.

## introdução
O planejamento de movimento para veículos articulados (ex: uniciclo com reboque) em ambientes não estruturados enfrenta o desafio da complexidade cinemática local e topológica global.

Controladores baseados em otimização (MPC) resolvam a estabilização local, mas sofrem em navegação global com obstáculos (mínimos locais). 

Abordagens de Aprendizado por Reforço (RL) prometem robustez global, mas sofrem com o problema do horizonte longo

Nesse contexto, exploramos o potencial do Heuristic Guided Reinforcement Learning (HuRL)para aumentar a eficiência e velocidade treinamento de agente de RL em espaço de estacionamento.

## Metodologia
A metodologia consiste em três pilares: a modelagem do domínio físico, a definição da heurística física para reward shaping e a integração no framework HuRL.
Modelamos a entidade trator-trailer, seu ambiente físico, interações e dinâmicas em um ambiente da api Gymnasium customizado.
Instrumentamos o ambiente de aprendizado com duas heurísticas principais, que modelam a recompensa do agente a cada passo de forma distinta:
Distância euclidiana: Recompensa o agente por reduzir a distância em linha reta até o objetivo
Distância de menor caminho considerando obstáculos (BFS): Recompensa o agente por seguir o menor caminho válido até o objetivo, reduzindo a menor distância calculada pelo BFS.
Treinamos agentes de aprendizado por reforço por um número fixo de passos com cada heurística, usando o mesmo algoritmo
Comparamos o desempenho de treinamento dos agentes entre si e ao baseline, um agente treinado sem heurística.

Para investigar o impacto da qualidade da heurística na eficiência do aprendizado, foram estabelecidos três cenários experimentais distintos:
Baseline: Agente treinado sem informação heurística auxiliar (recompensas esparsas).
Heurística Euclidiana: Moldagem de recompensa baseada na redução da distância linear (line-of-sight) ao objetivo.
Heurística Topológica (BFS): Moldagem de recompensa baseada na redução da distância do menor caminho válido, computada via Breadth-First Search em um mapa de ocupação discretizado.


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
distancia minima para considerar o veículo estacionado: 2 metros do centro da vaga de estacionamento.

### Heurísticas Avaliadas

Neste estudo, a eficácia do *reward shaping* foi investigada através da comparação de três abordagens distintas para a definição da função de potencial.

1. **Baseline (Sem Heurística)**
   Nesta configuração de controle, o agente é submetido a um regime de recompensa esparsa, recebendo sinais de reforço exclusivamente nos eventos terminais (sucesso ou falha) e penalidades temporais. Esta abordagem serve como linha de base para avaliar o impacto da introdução de sinais de recompensa densos providos pelas heurísticas.

2. **Heurística Euclidiana**
   Esta abordagem emprega a distância Euclidiana ($L^2$ norm) entre o centro de massa do veículo e o centróide da vaga de destino como função de potencial. O *reward shaping* é formulado para recompensar o gradiente negativo da distância, incentivando a redução da distância linear a cada passo de tempo. Embora computacionalmente eficiente, esta heurística ignora a topologia do ambiente e restrições cinemáticas, tornando-a suscetível a mínimos locais em ambientes com obstáculos não convexos.

3. **Heurística Topológica (BFS)**
   Para incorporar a geometria dos obstáculos na função de recompensa, utiliza-se o algoritmo de busca em largura (*Breadth-First Search* - BFS). O ambiente é discretizado em uma grade de ocupação com resolução de $1m \times 1m$. O mapa de distâncias geodésicas (considerando obstáculos estáticos) é computado a partir do ponto alvo. A recompensa densa é derivada da redução da distância do menor caminho válido pelo BFS, guiando o agente através de trajetórias livres de colisão e mitigando o problema de mínimos locais.


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

## Resultados

A avaliação dos modelos baseou-se em duas métricas principais: recompensa média acumulada e taxa de sucesso por episódio.

### Recompensa Média
A análise das curvas de aprendizado evidencia a superioridade da heurística BFS no *reward shaping*. O modelo **SAC + BFS** apresentou convergência robusta para valores elevados de recompensa. Em contrapartida, o modelo **SAC + Euclidiano** estagnou em patamares inferiores, limitado por mínimos locais. O **Baseline** estabilizou em valores negativos, indicando o aprendizado de uma política de "sobrevivência" (evitar colisões) sem, contudo, solucionar a tarefa de navegação.

### Taxa de Sucesso
Definida como a conclusão efetiva da manobra de estacionamento, a taxa de sucesso corroborou a eficácia da informação topológica. A heurística **BFS** obteve desempenho significativamente superior, superando a complexidade dos obstáculos. A heurística **Euclidiana** demonstrou eficácia limitada, restringindo-se majoritariamente a cenários onde o vetor de direção ao objetivo não apresentava obstruções físicas.


### Conclusão

O HuRL apresenta um grande potencial de aumentar a eficiência do treinamento de agentes de RL em aplicações de planejamento de rotas. Em nosso caso, o aprendizado efetivo só foi possibilitado pelo uso de uma heurística.

O vantagem trazida  pelo reward shaping depende fortemente da heurística escolhida. Quanto mais a heurística se conforma à dinâmica real do ambiente (obstáculos), maior o ganho de eficiência no treinamento do agente.

O presente estudo demonstrou a eficácia do framework *Heuristic Guided Reinforcement Learning* (HuRL) aplicado ao planejamento de movimento de veículos articulados, um domínio caracterizado por dinâmica não-holonômica e horizontes longos. Os resultados indicam que a aplicação de *reward shaping* baseado em potencial é determinante para superar a esparsidade de recompensas, viabilizando a convergência do algoritmo SAC onde a abordagem *baseline* convergiu apenas para comportamentos de sobrevivência.

A análise comparativa evidenciou que o ganho de desempenho é estritamente correlacionado à fidelidade topológica da informação heurística. Enquanto a heurística Euclidiana sofreu com mínimos locais, a abordagem baseada em BFS mostrou-se superior ao incorporar a geometria dos obstáculos no cálculo do potencial, fornecendo gradientes de recompensa densos e consistentes. Conclui-se, portanto, que a hibridização de métodos de busca clássicos com Aprendizado por Reforço Profundo constitui uma estratégia robusta e necessária para a resolução eficaz de tarefas de navegação complexa.


### Trabalhos futuros

Com base nos resultados obtidos e nas limitações identificadas, propõem-se as seguintes direções para a continuidade da pesquisa:

1.  **Incorporação de Restrições Não-Holonômicas na Heurística**: A substituição do BFS (que assume movimento omnidirecional) por algoritmos como *Hybrid A** ou *RRT**, que consideram o raio mínimo de curvatura do veículo, poderia gerar potenciais de recompensa ainda mais informativos, especialmente em manobras de precisão.

2.  **Evolução para Modelagem Dinâmica**: A transição do modelo cinemático para um modelo dinâmico completo, incorporando forças de atrito, inércia, massa e escorregamento dos pneus, aumentaria a fidelidade da simulação e facilitaria a transferência para veículos reais (*Sim-to-Real*).

3.  **Ambientes Dinâmicos e Estocásticos**: Avaliar a robustez do agente em cenários com obstáculos móveis (outros veículos, pedestres) e incertezas sensoriais, aproximando o ambiente de situações de tráfego real.

4.  **Curriculum Learning**: Implementar estratégias de aprendizado curricular, onde a complexidade do ambiente (número de obstáculos, dificuldade da manobra) aumenta progressivamente, visando acelerar a convergência e estabilidade do treinamento.

5.  **Exploração de Outras Arquiteturas de RL**: Investigar o desempenho de outros algoritmos *off-policy* (ex: TD3) ou *on-policy* (ex: PPO), bem como abordagens baseadas em modelo (*Model-Based RL*), sob o paradigma HuRL.









