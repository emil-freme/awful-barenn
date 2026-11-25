#!/usr/bin/env bash
# Executa o binário N vezes e reporta média, desvio padrão, mínimo e máximo
# do "Tempo de treino" impresso pelo programa.
# Uso: ./bench.sh [binario] [N]
set -euo pipefail
export LC_ALL=C

BIN=${1:-./barenn}
N=${2:-10}

echo "Binario: $BIN | Execucoes: $N"
for _ in $(seq "$N"); do
    "$BIN" | awk '/Tempo de treino/ {print $4}'
done | sort -n | awk '
    { s += $1; q += $1 * $1; a[NR] = $1 }
    END {
        m = s / NR
        printf "Tempo de treino: media %.4f s | dp %.4f s | min %.4f s | max %.4f s\n",
               m, sqrt(q / NR - m * m), a[1], a[NR]
    }'
