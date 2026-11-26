#!/usr/bin/env bash
# Compila a versão CUDA no nó de login do Franky e submete o job de GPU.
# Uso (na raiz do repositório, no Franky): ./franky_submit_gpu.sh [N]
set -euo pipefail
cd "$(dirname "$0")"
module load cuda/12.8.1
make -B barenn_cuda
mkdir -p resultados
sbatch --export=ALL,N="${1:-5}",GIT_COMMIT="$(git describe --tags --always --dirty)" franky_gpu.slurm
