#!/bin/bash
#SBATCH --job-name=barenn-cpu
#SBATCH --partition=merry_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=barenn-cpu-%j.out
# uso: make && sbatch slurm/cpu.slurm

echo "Job $SLURM_JOB_ID | nó $(hostname) | $(date)"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|L1d|L2|L3'
g++ --version | head -1

scripts/validar.sh ./barenn
scripts/bench.sh 5 ./barenn
