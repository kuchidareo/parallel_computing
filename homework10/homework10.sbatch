#!/bin/bash
#SBATCH -J homework10
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2048
#SBATCH -t 00:10:00

module load py-mpi4py/3.1.3
module load py-numpy/1.22.4



hostname #prints the node's name 
echo '# nodes: ----------------'
printenv SLURM_NNODES
echo '# tasks per node: -------------'
printenv SLURM_NTASKS_PER_NODE


cd "$SLURM_SUBMIT_DIR"

python3 a10_collect.py
