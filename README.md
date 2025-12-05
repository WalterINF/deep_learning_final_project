# deep_learning_final_project
Repositório do projeto final de AGL10225 -  Aprendizado Por Reforço

## objetivo

Este projeto tem por objetivo avaliar diferentes heurísticas aliadas ao HuRL - Heuristic Reinforcement Learning, para treinar um agente para planejamento de rotas de veículos articulados em espaço de estacionamento.

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

## Estrutura do repositório
**logs**: diretório para logs do TensorBoard

**models**: diretório para modelos treinados

**README.md**: este arquivo

**requirements.txt**: arquivo de dependências do projeto

**scripts**: diretório para scripts auxiliares uteis para treinamento

**plots**: diretório para plots dos resultados do treinamento

**scripts/clear_logs_models.py**: script para limpar logs e modelos treinados

**scripts/launch_tensorboard.py**: script para iniciar o TensorBoard

**src**: diretório para códigos que implementam o ambiente de simulação do Gymnasium

**train.py**: script para treinar o modelo SAC

**eval.py**: script para avaliar o modelo SAC treinado

**video**: diretório para vídeos de avaliação do agente SAC

Os demais diretórios contém códigos auxiliares para a nossa implementação e não são essenciais para a reprodução dos experimentos.



## Reproduzindo o treinamento

1. Criar um ambiente virtual para o projeto:
```bash
python -m venv venv
source venv/bin/activate
```
2. Instalar as dependências do projeto:
```bash
pip install -r requirements.txt
```
3. Treinar o modelo:
```bash
python train.py --heuristica <heuristica> --timesteps <timesteps> --salvar-cada <salvar-cada> --n-envs <n-envs>
```
Onde:
- heuristica é uma das seguintes: nao_holonomica, euclidiana, manhattan, reeds_shepp, nenhuma
- timesteps é o total de timesteps para treinar
- salvar-cada é o intervalo de timesteps para salvar o modelo
- n-envs é o número de ambientes paralelos para treinar

Alternativamente, pode-se treinar todos os modelos de uma vez:
```bash
bash train_all_heuristics.sh
```

4. (OPCIONAL) Acompanhar o treinamento no TensorBoard:
```bash
python scripts/launch_tensorboard.py
```
O TensorBoard será iniciado e você poderá acompanhar o treinamento no navegador.
Acesse http://localhost:6006/ para visualizar os logs.

5. Avaliar o modelo:
```bash
python eval.py --model-name <model-name>
```
Onde model-name é o nome do modelo treinado. Se nenhum nome for fornecido, o modelo mais recente será avaliado.




## Agente:

### Espaço de observação (o que o agente observa)

O vetor de coordenada privilegiada [z1, z2, z3, z4] + O_local: percepção local dos obstáculos - 14 sensores raycast posicionados ao redor do veículo
    
![veiculo com raycasts](veiculo_com_os_raycast.png)

### Espaço de ação (o que o agente pode fazer)

Controle: [v, alpha], onde:
* v: velocidade do veiculo.	
* alpha: ângulo de esterçamento do trator.

### Função de recompensa (o que o agente recebe)
* +100 por compleção do objetivo (estacionar na vaga de destino)
* +100 por alinhar o trailer na vaga corretamente ao estacionar (baseado na orientação do trailer)
* +1.0 * delta_distancia por progresso até o objetivo (onde delta_distancia é definido pela heurística)
* -100 por colisão com paredes ou outras vagas de estacionamento
* -100 por jackknife
* -20 distribuídos ao longo do episódio como penalidade por tempo: (-20/MAX_STEPS) por passo
* -0.1 por velocidade zero (penalidade por ficar parado)
* -0.02 por mudança brusca de esterçamento (smoothness penalty)

### critérios de parada
* colisão com obstáculos - incluindo outras vagas de estacionamento que não sejam a de origem ou destino - com o próprio veículo ou paredes do ambiente
* tempo: 90 segundos
* objetivo atingido: veiculo estacionado no ponto de destino.
* episódio terminado: tempo limite atingido.

## Heurísticas Avaliadas

Neste estudo, a eficácia do reward shaping foi investigada através da comparação de três abordagens distintas para a definição da função de potencial.

1. **Baseline (Sem Heurística)**
   Nesta configuração de controle, o agente é submetido a um regime de recompensa esparsa, recebendo sinais de reforço exclusivamente nos eventos terminais (sucesso ou falha) e penalidades temporais. Esta abordagem serve como linha de base para avaliar o impacto da introdução de sinais de recompensa densos providos pelas heurísticas.

2. **Heurística Euclidiana**
   Esta abordagem emprega a distância Euclidiana ($L^2$ norm) entre o centro de massa do veículo e o centróide da vaga de destino como função de potencial. O *reward shaping* é formulado para recompensar o gradiente negativo da distância, incentivando a redução da distância linear a cada passo de tempo. Embora computacionalmente eficiente, esta heurística ignora a topologia do ambiente e restrições cinemáticas, tornando-a suscetível a mínimos locais em ambientes com obstáculos não convexos.

3. **Heurística Topológica (BFS)**
   Para incorporar a geometria dos obstáculos na função de recompensa, utiliza-se o algoritmo de busca em largura (*Breadth-First Search* - BFS). O ambiente é discretizado em uma grade de ocupação com resolução de $1m \times 1m$. O mapa de distâncias geodésicas (considerando obstáculos estáticos) é computado a partir do ponto alvo. A recompensa densa é derivada da redução da distância do menor caminho válido pelo BFS, guiando o agente através de trajetórias livres de colisão e mitigando o problema de mínimos locais.

   **Heurística Não Holonômica**
   Esta abordagem emprega a distância não holonômica entre o centro de massa do veículo e o centróide da vaga de destino como função de potencial. O *reward shaping* é formulado para recompensar o gradiente negativo da distância, incentivando a redução da distância não holonômica a cada passo de tempo. Esta heurística considera a geometria do veículo e o ambiente, tornando-a mais robusta em ambientes com obstáculos não convexos.












