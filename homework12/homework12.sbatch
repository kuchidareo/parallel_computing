#!/bin/bash
#SBATCH -N 1
#SBATCH -J pellespark
#SBATCH --mem=16000
#SBATCH --output=sparkjob-%j.out
#SBATCH -t 00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1


## --------------------------------------
## 0. Preparation
## --------------------------------------

# load the modules required for Spark 

module load python/3.9.12
module load py-virtualenv/20.22.0
module load openjdk/17.0.3_7

virtualenv --no-download sparkenv2
source sparkenv2/bin/activate

pip install pyspark 
pip install numpy 

rows=32000
cols=1000
partitions=100

## --------------------------------------
## Submit a task to the Spark cluster
## --------------------------------------

for cores in 1 2 4 8; do
    echo "Cores: $cores"
    spark-submit --master "local[$cores]" --executor-memory 1000M --driver-memory 6000M sparkMatVecMult.py $rows $cols $partitions || exit $?
done
