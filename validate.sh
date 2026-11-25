#!/usr/bin/env bash
# Compara a saída do binário com a saída de referência do código legado.
# A linha de tempo é removida porque varia entre execuções.
# Uso: ./validate.sh [binario]
set -euo pipefail

BIN=${1:-./barenn}
REF=ref_output.txt

if diff <("$BIN" | grep -v 'Tempo de treino') "$REF"; then
    echo "OK: saida identica a referencia ($REF)"
else
    echo "FALHA: saida difere da referencia ($REF)" >&2
    exit 1
fi
