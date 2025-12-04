#!/bin/bash
# Script para treinar 4 modelos SAC, um para cada heurística disponível
# Executa sequencialmente com 8 ambientes, 5M timesteps, salvando a cada 1M

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configurações
TIMESTEPS=5000000
SAVE_EVERY=1000000
N_ENVS=8

# Heurísticas disponíveis
HEURISTICS=("nao_holonomica" "euclidiana" "manhattan" "nenhuma")

echo "=============================================="
echo "Treinamento de modelos com todas as heurísticas"
echo "=============================================="
echo "Timesteps por modelo: $TIMESTEPS"
echo "Salvar a cada: $SAVE_EVERY timesteps"
echo "Ambientes paralelos: $N_ENVS"
echo "Heurísticas: ${HEURISTICS[*]}"
echo "=============================================="
echo ""

for i in "${!HEURISTICS[@]}"; do
    HEURISTICA="${HEURISTICS[$i]}"
    NUM=$((i + 1))
    
    echo ""
    echo "=============================================="
    echo "[$NUM/4] Treinando modelo com heurística: $HEURISTICA"
    echo "=============================================="
    echo ""
    
    python train.py \
        --heuristica "$HEURISTICA" \
        --timesteps "$TIMESTEPS" \
        --salvar-cada "$SAVE_EVERY" \
        --n-envs "$N_ENVS"
    
    echo ""
    echo "[$NUM/4] Modelo com heurística '$HEURISTICA' concluído!"
    echo ""
done

echo ""
echo "=============================================="
echo "Treinamento de todos os modelos concluído!"
echo "=============================================="

