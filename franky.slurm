#!/bin/bash
# Job SLURM para rodar o relatório no cluster Franky.
# Submeta a partir da raiz do repositório:  sbatch franky.slurm
# Ajuste a partição e os módulos conforme o cluster.
#SBATCH --job-name=barenn
#SBATCH --output=resultados/franky_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
##SBATCH --partition=normal
##SBATCH --gres=gpu:1

# module load gcc
# module load cuda

./run_report.sh 10
