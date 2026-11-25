#!/bin/bash
# Job SLURM para rodar o relatório no cluster Franky.
# Nao submeta direto: use ./franky_submit.sh, que compila no no de login
# (os nos de computacao nao tem os headers da glibc) e entao chama o sbatch.
# Particoes CPU: merry_cpu (5 min), sunny_cpu (20 min), pluton_cpu (1 h).
# Particoes GPU: merry_gpu, sunny_gpu, pluton_gpu (usar com --gres=gpu:1).
#SBATCH --job-name=barenn
#SBATCH --output=resultados/franky_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=sunny_cpu
##SBATCH --gres=gpu:1

# module load gcc
# module load cuda

SKIP_BUILD=1 ./run_report.sh 10
