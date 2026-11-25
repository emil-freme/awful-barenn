#!/usr/bin/env bash
# Gera um relatório completo de uma versão: ambiente (hardware, compilador,
# flags), validação de corretude e benchmark. Roda igual na máquina local e
# no cluster (via job SLURM), para que os resultados sejam comparáveis.
# Uso: ./run_report.sh [N_execucoes]
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")"

N=${1:-10}

echo "===== AMBIENTE ====="
echo "Host:       $(hostname)"
echo "Data:       $(date -Iseconds)"
echo "Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo '?') $(git describe --tags --always 2>/dev/null || true)"
echo "Kernel:     $(uname -r)"
lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|L1d|L2|L3)' || true
echo "RAM:        $(free -h | awk '/^Mem/ {print $2}')"
echo "Compilador: $(${CXX:-g++} --version | head -1)"
if command -v nvidia-smi >/dev/null; then
    echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
fi
if command -v nvcc >/dev/null; then
    echo "CUDA:       $(nvcc --version | tail -1)"
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "SLURM:      job $SLURM_JOB_ID | particao ${SLURM_JOB_PARTITION:-?} | nos ${SLURM_JOB_NODELIST:-?} | cpus ${SLURM_CPUS_PER_TASK:-?}"
fi

echo
echo "===== BUILD ====="
make -B 2>&1

echo
echo "===== VALIDACAO ====="
make validate

echo
echo "===== BENCHMARK (N=$N) ====="
./bench.sh ./barenn "$N"
