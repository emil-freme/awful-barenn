#!/usr/bin/env bash
# Compara a saída do binário com a saída de referência do código legado.
# As linhas de tempo ("Tempo ...") são removidas porque variam entre execuções.
# Uso: ./validate.sh [binario]
set -euo pipefail

BIN=${1:-./barenn}
REF=ref_output.txt

if diff <("$BIN" | grep -v '^Tempo ') "$REF"; then
    echo "OK: saida identica a referencia ($REF)"
else
    echo "FALHA: saida difere da referencia ($REF)" >&2
    exit 1
fi
