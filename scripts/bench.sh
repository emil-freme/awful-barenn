#!/usr/bin/env bash
# Uso: scripts/bench.sh N comando [args...]  (média, desvio e mínimo de cada "Tempo de ...")
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Uso: $0 N comando [args...]" >&2
    exit 1
fi
n=$1
shift

for _ in $(seq "$n"); do
    "$@"
done | awk '
/^Tempo de / {
    k = $3
    if (!(k in cont)) ordem[++nk] = k
    v[k, ++cont[k]] = $4
}
END {
    for (o = 1; o <= nk; o++) {
        k = ordem[o]; m = cont[k]; s = 0; s2 = 0; mn = v[k, 1]; lista = ""
        for (i = 1; i <= m; i++) {
            x = v[k, i]; s += x; s2 += x * x
            if (x < mn) mn = x
            lista = lista " " x
        }
        med = s / m
        var = (m > 1) ? (s2 - m * med * med) / (m - 1) : 0
        if (var < 0) var = 0
        printf "Tempo de %-10s n=%d | media %.6f s | desvio %.6f s | min %.6f s | valores:%s\n",
               k, m, med, sqrt(var), mn, lista
    }
}'
