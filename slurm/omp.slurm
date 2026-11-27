#!/bin/bash
#SBATCH --job-name=barenn-omp
#SBATCH --partition=sunny_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=barenn-omp-%j.out
# uso: make && scripts/teste_grande.sh 50 && sbatch slurm/omp.slurm

echo "Job $SLURM_JOB_ID | nó $(hostname) | $(date) | cpus-per-task=$SLURM_CPUS_PER_TASK"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|L1d|L2|L3'
g++ --version | head -1

for sch in static dynamic guided; do
    for t in 1 2 4 8 16; do
        export OMP_NUM_THREADS=$t OMP_SCHEDULE=$sch
        echo "===== $sch | $t thread(s)"
        scripts/validar.sh ./barenn | tail -1
        scripts/bench.sh 5 ./barenn
        echo "--- teste 50x (optdigits_x50.tes)"
        scripts/bench.sh 5 ./barenn optdigits.tra optdigits_x50.tes
    done
done
