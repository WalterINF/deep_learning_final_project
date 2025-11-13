# Documentação do Código - Sistema de Simulação Trator-Trailer

Este documento descreve a estrutura e funcionamento do código legado no diretório `codigos/`, que implementa um sistema de simulação cinemática para veículos trator-trailer.

## Visão Geral do Sistema

O sistema implementa um simulador cinemático para veículos articulados (trator com trailer), incluindo:
- Modelos cinemáticos baseados em equações diferenciais
- Cálculo de geometria e vértices dos veículos
- Detecção de colisões com obstáculos e entre componentes do veículo
- Visualização de trajetórias e estados
- Gerenciamento de mapas binários

---

## Arquivos Principais

### 1. `parametros_trator_trailer.py` (494 linhas)

**Propósito**: Define os parâmetros geométricos dos veículos e calcula vértices para visualização e detecção de colisão.

#### Classes Principais:

##### `ParametrosTratorTrailer`
Gerencia parâmetros geométricos para um conjunto trator + trailer.

**Parâmetros do Trator**:
- `comprimento_trator`: comprimento total [m]
- `distancia_eixo_traseiro_quinta_roda`: distância do eixo traseiro à quinta roda [m]
- `distancia_eixo_dianteiro_quinta_roda`: distância do eixo dianteiro à quinta roda [m]
- `distancia_frente_quinta_roda`: distância da frente à quinta roda [m]
- `largura_trator`: largura total [m]

**Parâmetros do Trailer**:
- `comprimento_trailer`: comprimento total [m]
- `distancia_eixo_traseiro_trailer_quinta_roda`: distância do eixo traseiro ao pino-rei [m]
- `distancia_frente_trailer_quinta_roda`: distância da frente ao pino-rei [m]
- `largura_trailer`: largura total [m]

**Métodos Principais**:

1. **`calcula_vertices_em_metros(estado5: Estado5)`** (linhas 79-182)
   - Calcula os 4 vértices do trator e trailer em coordenadas métricas
   - Usa transformações trigonométricas para rotacionar os retângulos
   - Retorna: `{'trator': [4 pontos], 'trailer': [4 pontos]}`
   - Ordem dos vértices: traseira esquerda → traseira direita → dianteira direita → dianteira esquerda

2. **`calculate_vertices_pixel(estado5: Estado5, resolution: float)`** (linhas 184-210)
   - Converte vértices de metros para pixels
   - Usa fator de resolução (m/pixel)
   - Útil para visualização em mapas de grade

3. **`calcular_vertices_com_rodas(estado5: Estado5, steering_angle: float)`** (linhas 212-275)
   - Calcula vértices do veículo + rodas dianteiras
   - Considera o ângulo de esterçamento das rodas
   - Retorna dicionário com trator, trailer e 2 rodas dianteiras

##### `ParametrosTrator`
Classe similar à anterior, mas apenas para trator sem trailer (linhas 277-492).
- Trabalha com `Estado3` (x, y, theta) ao invés de `Estado5`
- Métodos equivalentes para cálculo de vértices

---

### 2. `modelo_cinematico_trator_trailer.py` (191 linhas)

**Propósito**: Implementa os modelos cinemáticos do veículo usando integração numérica com CasADi.

#### Classes Principais:

##### `TratorComUmTrailer` (linhas 7-110)
Modelo cinemático do trator-trailer em estado reduzido `[x1, y1, theta1, beta]`.

**Equações de Movimento**:
```
ẋ₁ = v·cos(θ₁)
ẏ₁ = v·sin(θ₁)
θ̇₁ = (v/D)·tan(α)
β̇ = -(v/L)·sin(β) - (v/D)·tan(α) + (a·v/(L·D))·tan(α)·cos(β)
```

Onde:
- `(x1, y1)`: posição do eixo traseiro do trator
- `θ1`: orientação do trator
- `β`: ângulo relativo entre trator e trailer
- `v`: velocidade longitudinal
- `α`: ângulo de esterçamento
- `D`: distância entre eixos do trator
- `L`: distância entre eixos do trailer
- `a`: distância da quinta roda ao eixo traseiro

**Métodos Principais**:

1. **`__init__(parametros: dict, dt: float)`** (linhas 8-61)
   - Configura equações diferenciais usando CasADi
   - Cria integrador numérico CVODES
   - Define passo de integração `dt`

2. **`atualizar_estado(estado: Estado4, controle: tuple)`** (linhas 63-74)
   - Integra o estado por um passo de tempo `dt`
   - Controle: `(velocidade, ângulo_esterçamento)`
   - Retorna novo `Estado4`

3. **`obter_estado5(estado_4: Estado4)`** (linhas 101-107)
   - Converte `Estado4` → `Estado5`
   - Calcula `θ2 = θ1 + β` (orientação do trailer)

4. **`obter_estado7(estado: Estado4)`** (linhas 91-99)
   - Converte para estado completo com posições de ambos os eixos
   - Calcula `(x2, y2)`: posição do eixo traseiro do trailer

##### `Trator` (linhas 112-191)
Modelo cinemático apenas do trator (sem trailer).
- Estado: `[x, y, theta]`
- Equações simplificadas (sem β)
- Métodos equivalentes ao modelo completo

---

### 3. `mapa.py` (149 linhas)

**Propósito**: Gerencia mapas ocupacionais binários (livre/ocupado) com padrão Singleton.

#### Classe Principal: `Mapa`

**Atributos**:
- `_matriz`: array NumPy com o mapa (1 = livre, 0 = ocupado)
- `resolucao`: pixels por metro
- `ponto_de_referencia_global`: coordenada de referência (origem)
- `ALTURA`, `LARGURA`: dimensões do mapa em pixels

**Métodos Principais**:

1. **`__new__(cls, definition)`** (linhas 9-21)
   - Implementa padrão Singleton por nome do mapa
   - Garante que cada mapa seja carregado apenas uma vez

2. **`__init__(definition: ListaDeMapas)`** (linhas 23-41)
   - Carrega mapa de arquivo `.npy`
   - Protege contra múltiplas inicializações

3. **`coordenadaGlobalParaPixel(ponto_global: tuple)`** (linhas 71-79)
   - Converte coordenadas do mundo real (metros) → pixels
   - Aplica transformação de referência e escala

4. **`linhaDeVisao(p1: tuple, p2: tuple)`** (linhas 46-69)
   - Verifica linha de visão entre dois pontos
   - Usa algoritmo de Bresenham
   - Retorna `False` se houver obstáculo no caminho

5. **`checarColisaoComObstaculos(vertices_mundo: list)`** (linhas 81-100)
   - Verifica colisão dos vértices do veículo com obstáculos
   - Converte vértices para pixels
   - Checa se algum vértice está em área ocupada ou fora do mapa

6. **`obterCentrosLivres(intervalo: int)`** (linhas 125-145)
   - Divide o mapa em blocos de tamanho `intervalo`
   - Retorna centros de blocos completamente livres
   - Útil para amostragem de pontos válidos

---

### 4. `colisao.py` (64 linhas)

**Propósito**: Centraliza toda a lógica de detecção de colisões.

#### Classe Principal: `Colisao`

**Inicialização**:
```python
def __init__(self, mapa, angulo_maximo_articulacao: float, diferenca_entre_angulos)
```
- `mapa`: objeto `Mapa` para checar colisões com obstáculos
- `angulo_maximo_articulacao`: limite físico do ângulo β
- `diferenca_entre_angulos`: objeto `Angulos` para cálculos angulares

**Métodos Principais**:

1. **`checarColisaoVeiculo(theta1: float, theta2: float)`** (linhas 16-29)
   - Verifica se o ângulo relativo β excede o limite permitido
   - β = diferença entre orientações do trator e trailer
   - Previne jackknife (dobramento excessivo)

2. **`checarColisaoComObstaculos(vertices: dict)`** (linhas 31-44)
   - Delega verificação ao objeto `mapa`
   - Converte dicionário de vértices em lista plana

3. **`checar_colisao(vertices: dict, theta1: float, theta2: float)`** (linhas 46-60)
   - **Método principal**: verifica ambos os tipos de colisão
   - Retorna `True` se houver qualquer colisão
   - Combina lógica de jackknife e obstáculos

---

### 5. `calculos_com_angulos.py` (70 linhas)

**Propósito**: Utilitários para manipulação de ângulos com normalização.

#### Classe Principal: `Angulos`

Todas as operações trabalham com ângulos em radianos no intervalo `[-π, π]`.

**Métodos Principais**:

1. **`normalize_angle(angle)`** (linhas 45-55)
   - Normaliza ângulo para intervalo `[-π, π]`
   - Fórmula: `(angle + π) % (2π) - π`
   - Usa cache LRU para otimização

2. **`normalize_if_needed(angle)`** (linhas 29-42)
   - Normaliza apenas se necessário (|angle| > π)
   - Evita cálculos desnecessários

3. **`menor_diferenca_entre_angulos(angulo1, angulo2, usarDirecao=False)`** (linhas 12-26)
   - Calcula menor diferença angular entre dois ângulos
   - `usarDirecao=True`: retorna diferença com sinal
   - `usarDirecao=False`: retorna valor absoluto

4. **`soma_angulo_normalizado(angulo1, angulo2)`** (linhas 57-70)
   - Soma dois ângulos e normaliza o resultado
   - Garante resultado em `[-π, π]`

**Otimização**: Todos os métodos usam `@lru_cache(maxsize=100)` para memorização.

---

### 6. `lista_veiculos.json` (55 linhas)

**Propósito**: Base de dados de configurações de veículos.

#### Estrutura:

```json
{
    "veiculos": [
        {
            "nome": "BUG1",
            "geometria_veiculo": {
                "comprimento_trator": 6.086,
                "distancia_eixo_traseiro_quinta_roda": 0.736,
                ...
                "angulo_maximo_articulacao": 0.78539
            }
        },
        ...
    ]
}
```

**Veículos Definidos**:
1. **BUG1**: Veículo completo com medidas reais
2. **VeiculoDesacoplado**: Mesmas medidas do BUG1
3. **BUG2-ainda-a-coletar**: Placeholder (valores zerados)

**Dimensões Típicas**:
- Trator: ~6m comprimento × 2.6m largura
- Trailer: ~9.6m comprimento × 2.6m largura
- Ângulo máximo articulação: ~0.785 rad (45°)

---

### 7. `mapaCTR_16px_metro_com_preto_preto_cropped.npy` (2.5MB)

**Propósito**: Mapa ocupacional binário do ambiente.

**Características**:
- Formato: Array NumPy 2D
- Resolução: 16 pixels/metro
- Valores: 0 (obstáculo) ou 1 (livre)
- Tamanho: ~2.5MB (estimativa: ~1000×1000 pixels ou ~62×62 metros)

**Uso**:
- Carregado pela classe `Mapa`
- Base para planejamento de trajetórias
- Detecção de colisões com ambiente

---

## Visualizações (`visualizacoes/`)

### 8. `v_mapa.py` (130 linhas)

**Classe**: `VMapa`

**Funcionalidade**: Visualização de mapas com coordenadas métricas.

**Métodos Principais**:
- `desenhar_mapa(ax, mapa)`: Plota mapa completo com eixos em metros
- `desenhar_caminho(ax, caminho_metros, mapa)`: Sobrepõe trajetória planejada
- `meters_to_pixels()` / `pixels_to_meters()`: Conversões de coordenadas

**Características**:
- Sistema de coordenadas global em metros
- Grade com marcações a cada 10m
- Origem (0,0) marcada em vermelho
- Suporta sobreposição de caminhos

---

### 9. `v_estado.py` (177 linhas)

**Classes**: `VEstado` e `VEstadoTrator`

**Funcionalidade**: Visualização de poses do veículo.

**Métodos Principais**:

1. **`desenhar_pose(ax, vertices: dict)`**
   - Desenha trator e trailer como polígonos
   - Cores padrão: darkturquoise (trator), steelblue (trailer)

2. **`desenhar_pose_com_rodas(ax, vertices: dict)`**
   - Adiciona rodas dianteiras à visualização
   - Rodas em dourado (gold)

3. **`desenhar2poses(ax, vertices1: dict, vertices2: dict)`**
   - Desenha dois veículos simultaneamente
   - Útil para comparar estados inicial/final

4. **`create_patches(vertices)`**
   - Cria patches de Matplotlib para animações
   - Retorna objetos `Polygon`

---

### 10. `v_trajetoria.py` (204 linhas)

**Classes**: `VisualizaTrajetoria` e `VisualizaTrajetoriaTrator`

**Funcionalidade**: Visualização e animação de trajetórias completas.

**Métodos Principais**:

1. **`desenhar_trajetoria(ax, parametros_veiculo, trajectory)`** (linhas 20-36)
   - Desenha múltiplas poses ao longo da trajetória
   - Usa transparência (alpha=0.4) para clareza visual
   - Cria coleção de patches para eficiência

2. **`desenhar_trajetoria_com_rodas(ax, parametros_veiculo, conjunto)`** (linhas 38-47)
   - Similar ao anterior, mas inclui rodas dianteiras
   - Mostra ângulo de esterçamento

3. **`animar_trajetoria(ax, parametros_veiculo, conjunto, nome_arquivo, folder)`** (linhas 49-107)
   - Cria animação MP4 da trajetória
   - Usa FFmpeg para exportação
   - **Pré-calcula vértices** para otimização
   - 60 FPS, limita a 3000 frames
   - Desenha linha do caminho percorrido
   - **Formato do conjunto**: `[(estado, angle, velocidade), ...]`

**Otimizações**:
- Pré-cálculo de todos os vértices antes da animação
- Uso de `blit=True` para renderização eficiente
- Limite de frames para arquivos gerenciáveis

---

### 11. `v_caminho.py` (21 linhas)

**Classe**: `VCaminho`

**Funcionalidade**: Visualização simples de caminhos 2D.

**Método Principal**:
- `desenhar_caminho(ax, path)`: Plota lista de pontos (x, y) como linha contínua

**Uso**: Visualização de trajetórias planejadas sobre mapas.

---

### 12. `v_seta.py` (20 linhas)

**Classe**: `VSeta`

**Funcionalidade**: Desenha setas indicando orientação do veículo.

**Método Principal**:
- `desenhar_seta(ax, start, length=2, width=2)`: 
  - Desenha seta a partir do estado
  - Direção baseada em θ
  - Tamanho e largura configuráveis

**Uso**: Indicação visual de orientação em diagramas de estado.

---

## Como o Sistema Funciona em Conjunto

### Fluxo Típico de Simulação

```
1. CONFIGURAÇÃO
   ├─ Carregar parâmetros do veículo (lista_veiculos.json)
   ├─ Criar ParametrosTratorTrailer
   ├─ Carregar mapa (Mapa.py → .npy)
   └─ Criar modelo cinemático (TratorComUmTrailer)

2. INICIALIZAÇÃO
   ├─ Definir estado inicial (Estado4 ou Estado5)
   ├─ Criar objeto Colisao
   └─ Calcular vértices iniciais

3. LOOP DE SIMULAÇÃO
   Para cada passo de tempo:
   ├─ Aplicar controle (v, α)
   ├─ Atualizar estado (modelo.atualizar_estado)
   ├─ Calcular novos vértices (parametros.calcula_vertices_em_metros)
   ├─ Verificar colisões (colisao.checar_colisao)
   ├─ Armazenar trajetória
   └─ Repetir ou parar se houver colisão

4. VISUALIZAÇÃO
   ├─ Desenhar mapa (VMapa.desenhar_mapa)
   ├─ Desenhar trajetória (VisualizaTrajetoria.desenhar_trajetoria)
   ├─ Ou criar animação (VisualizaTrajetoria.animar_trajetoria)
   └─ Salvar/exibir resultados
```

### Dependências entre Módulos

```
modelo_cinematico_trator_trailer.py
    ├─ Usa: calculos_com_angulos.py (Angulos)
    └─ Retorna: Estado4, Estado5, Estado7

parametros_trator_trailer.py
    ├─ Recebe: Estado5 (ou Estado3 para trator)
    └─ Calcula: vértices para visualização/colisão

mapa.py
    ├─ Carrega: .npy
    ├─ Recebe: vértices em metros
    └─ Detecta: colisões com obstáculos

colisao.py
    ├─ Usa: mapa.py (colisões com ambiente)
    ├─ Usa: calculos_com_angulos.py (ângulo β)
    └─ Integra: ambas as verificações

visualizacoes/
    ├─ v_mapa.py: desenha mapa base
    ├─ v_estado.py: desenha veículo (polígonos)
    ├─ v_trajetoria.py: animações
    ├─ v_caminho.py: linhas de trajetória
    └─ v_seta.py: indicadores de orientação
```

### Estados do Sistema

O sistema trabalha com diferentes representações de estado:

1. **Estado3**: `(x, y, θ)` - apenas trator
2. **Estado4**: `(x, y, θ, β)` - trator-trailer (estado reduzido)
3. **Estado5**: `(x, y, θ₁, β, θ₂)` - adiciona orientação do trailer
4. **Estado7**: `(x₁, y₁, θ₁, x₂, y₂, θ₂, β)` - estado completo com ambas as posições

### Sistema de Coordenadas

- **Global (metros)**: Sistema de referência do mundo real
  - Origem definida em `ponto_de_referencia_global`
  - Usado para cálculos de física e geometria

- **Pixels**: Sistema de coordenadas do mapa
  - 1 pixel = 0.0625 metros (resolução 16 px/m)
  - Origem no canto superior esquerdo da imagem
  - Eixo Y invertido (crescente para baixo)

### Detecção de Colisões

O sistema implementa **duas verificações independentes**:

1. **Colisão Jackknife** (colisao.py):
   - Verifica se |β| > `angulo_maximo_articulacao`
   - Previne dobramento físico impossível

2. **Colisão com Ambiente** (mapa.py):
   - Verifica cada vértice do veículo
   - Checa linha de visão entre vértices consecutivos
   - Detecta se está fora dos limites do mapa

---

## Tecnologias e Bibliotecas Utilizadas

- **CasADi**: Framework de otimização para equações diferenciais
  - Integrador CVODES para modelos cinemáticos
  - Programação simbólica de ODEs

- **NumPy**: Operações numéricas e manipulação de arrays
  - Mapas como arrays 2D
  - Cálculos vetoriais

- **Matplotlib**: Visualização e animações
  - Polígonos para representação de veículos
  - FFMpegWriter para exportação de vídeos

- **JSON**: Armazenamento de configurações de veículos

---

## Limitações e Considerações

1. **Modelo Cinemático**: Não considera dinâmica (forças, aceleração)
2. **Colisões Simplificadas**: Vértices + linha de visão (não detecção pixel-perfeita)
3. **Mapas Estáticos**: Não suporta obstáculos dinâmicos
4. **Integração Numérica**: Precisão depende do passo `dt`
5. **Performance**: Pré-cálculo de vértices essencial para animações fluidas

---

## Possíveis Aplicações

- Planejamento de trajetórias para veículos articulados
- Simulação de estacionamento de caminhões
- Treinamento de algoritmos de controle (RL)
- Validação de manobras antes de execução real
- Geração de datasets para aprendizado de máquina

---

## Referências

O modelo cinemático é baseado no artigo:
**DOI**: [10.1007/s40747-021-00330-z](https://doi.org/10.1007/s40747-021-00330-z)

Equações implementadas nas linhas 43-51 de `modelo_cinematico_trator_trailer.py`.

---

*Documentação gerada em: 2025-11-12*

