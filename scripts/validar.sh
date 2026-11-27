#!/usr/bin/env bash
# Uso: scripts/validar.sh comando [args...]  (compara a saída com a do legado)
set -uo pipefail

ref="$(dirname "$0")/../referencia/saida_legado.txt"
if [ ! -f "$ref" ]; then
    echo "Referência ausente: $ref" >&2
    exit 1
fi

if "$@" | grep -v '^Tempo' | diff -u "$ref" -; then
    echo "VALIDACAO OK: saida identica a do legado (exceto linhas de tempo)"
else
    echo "VALIDACAO FALHOU: veja as diferencas acima" >&2
    exit 1
fi
