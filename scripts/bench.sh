#!/usr/bin/env bash
# Uso: scripts/bench.sh N comando [args...]  (média, desvio e mínimo dos tempos)
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
/^Tempo de treino/    { t[++nt] = $4 }
/^Tempo de avaliacao/ { a[++na] = $4 }
function resumo(nome, v, n,    i, s, s2, m, var, mn, lista) {
    if (n == 0) return
    s = 0; s2 = 0; mn = v[1]; lista = ""
    for (i = 1; i <= n; i++) {
        s += v[i]; s2 += v[i] * v[i]
        if (v[i] < mn) mn = v[i]
        lista = lista sprintf(" %s", v[i])
    }
    m = s / n
    var = (n > 1) ? (s2 - n * m * m) / (n - 1) : 0
    if (var < 0) var = 0
    printf "%-18s n=%d | media %.6f s | desvio %.6f s | min %.6f s | valores:%s\n",
           nome, n, m, sqrt(var), mn, lista
}
END {
    resumo("Tempo de treino", t, nt)
    resumo("Tempo de avaliacao", a, na)
}'
