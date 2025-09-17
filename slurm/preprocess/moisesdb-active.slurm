#!/bin/bash
#SBATCH -Jmoisesdb-preprocess          
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=24 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/Report-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

srun python scripts/preprocess/moisesdb.py -cn moisesdb-active