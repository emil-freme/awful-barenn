#!/usr/bin/env bash
# Compila no nó de login do Franky e submete o job de benchmark.
# Uso (na raiz do repositório, no Franky): ./franky_submit.sh
set -euo pipefail
cd "$(dirname "$0")"
make -B
mkdir -p resultados
sbatch --export=ALL,GIT_COMMIT="$(git describe --tags --always)" franky.slurm
