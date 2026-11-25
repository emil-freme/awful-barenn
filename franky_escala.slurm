#!/bin/bash
# Mede a escalabilidade com 1, 2, 4 e 8 threads OpenMP no Franky.
# Use depois de compilar no nó de login:  make -B && sbatch franky_escala.slurm
#SBATCH --job-name=barenn-escala
#SBATCH --output=resultados/franky_escala_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=sunny_cpu

echo "Host: $(hostname) | CPUs alocadas: $(taskset -cp $$ | awk -F: '{print $2}')"
for t in 1 2 4 8; do
    OMP_NUM_THREADS=$t ./bench.sh ./barenn 10
    echo
done
