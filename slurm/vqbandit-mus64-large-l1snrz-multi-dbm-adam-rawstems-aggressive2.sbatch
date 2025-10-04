#!/bin/bash
#SBATCH -Jvqbandit-mus64-large-l1snrz-multi-dbm-adam-musdb18hq-moisesdb-aggressive-vdbo                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1 --constraint=A100-80GB|H200
#SBATCH --cpus-per-task=24 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/vqbandit-mus64-large-l1snrz-multi-dbm-adam-rawstems-aggressive2-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3+pace@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn vqbandit-mus64-large-l1snrz-multi-dbm-adam-rawstems-aggressive2

