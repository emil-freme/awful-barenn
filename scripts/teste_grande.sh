#!/usr/bin/env bash
# Uso: scripts/teste_grande.sh [N]  (gera optdigits_xN.tes, teste repetido N vezes)
set -euo pipefail

n=${1:-50}
saida="optdigits_x${n}.tes"
for _ in $(seq "$n"); do
    cat optdigits.tes
done > "$saida"
wc -l "$saida"
