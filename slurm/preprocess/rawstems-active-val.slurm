#!/bin/bash
#SBATCH --array=0-50
#SBATCH -Jmoisesdb-preprocess          
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/Report-%A/%a.out                  # Combined output and error messages file
#SBATCH --mail-type=END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda


export PARALLEL_MODE=1
export INDEX=$SLURM_ARRAY_TASK_ID
export CHUNK_SIZE=1

srun python scripts/preprocess/rawstems.py -cn rawstems-active