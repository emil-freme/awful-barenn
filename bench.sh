#!/usr/bin/env bash
# Executa o binário N vezes e reporta média, desvio padrão, mínimo e máximo
# de cada tempo impresso pelo programa (treino, leitura e total).
# Uso: ./bench.sh [binario] [N]
set -euo pipefail
export LC_ALL=C

BIN=${1:-./barenn}
N=${2:-10}

echo "Binario: $BIN | Execucoes: $N | OMP_NUM_THREADS=${OMP_NUM_THREADS:-padrao}"
SAIDAS=$(for _ in $(seq "$N"); do "$BIN" | grep '^Tempo '; done)
for METRICA in "Tempo de treino" "Tempo de leitura" "Tempo total"; do
    echo "$SAIDAS" | awk -v m="$METRICA" 'index($0, m " ") == 1 {print $(NF-1)}' |
        sort -n | awk -v m="$METRICA" '
        { s += $1; q += $1 * $1; a[NR] = $1 }
        END {
            if (NR == 0) exit
            v = q / NR - (s / NR) ^ 2; if (v < 0) v = 0
            printf "%s: media %.4f s | dp %.4f s | min %.4f s | max %.4f s\n",
                   m, s / NR, sqrt(v), a[1], a[NR]
        }'
done
