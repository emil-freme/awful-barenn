#!/bin/bash
# Job SLURM da versão CUDA no Franky (nós compute20-23, NVIDIA RTX A1000).
# Nao submeta direto: use ./franky_submit_gpu.sh, que compila no no de login.
#SBATCH --job-name=barenn-gpu
#SBATCH --output=resultados/franky_gpu_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --partition=sunny_gpu
#SBATCH --gres=gpu:1

module load cuda/12.8.1

echo "===== AMBIENTE ====="
echo "Host:       $(hostname)"
echo "Commit:     ${GIT_COMMIT:-?}"
lscpu | grep -E '^Model name' || true
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader)"
echo "CUDA:       $(nvcc --version | tail -1)"
echo "Flags:      $(grep -E '^NVCCFLAGS' -A1 Makefile | tr -s ' \\\n' ' ')"
echo "SLURM:      job $SLURM_JOB_ID | particao $SLURM_JOB_PARTITION | nos $SLURM_JOB_NODELIST"

echo
echo "===== VALIDACAO ====="
./validate.sh ./barenn_cuda

echo
echo "===== BENCHMARK (N=${N:-5}) ====="
./bench.sh ./barenn_cuda "${N:-5}"

echo
echo "===== NSIGHT SYSTEMS ====="
mkdir -p profiling
nsys profile --force-overwrite=true --trace=cuda -o "profiling/barenn_cuda_$SLURM_JOB_ID" ./barenn_cuda > /dev/null
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    "profiling/barenn_cuda_$SLURM_JOB_ID.nsys-rep"
