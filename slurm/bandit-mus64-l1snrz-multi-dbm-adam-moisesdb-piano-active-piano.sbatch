#!/bin/bash
#SBATCH -Jbandit-mus64-l1snrz-multi-dbm-adam-moisesdb-piano-active-piano                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1 --constraint=A100-80GB|H100|H200
#SBATCH --cpus-per-task=24 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/bandit-mus64-l1snrz-multi-dbm-adam-moisesdb-piano-active-piano-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3+pace@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn bandit-mus64-l1snrz-multi-dbm-adam-moisesdb-piano-active-piano ++ckpt_path=/storage/ice1/4/1/kwatchar3/banda/banda/m6x93ix5/checkpoints/last.ckpt
