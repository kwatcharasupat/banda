#!/bin/bash
#SBATCH -Jbandit-moisesdb-nodbm                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:gpu-h100:1 # Request one GPU of any type
#SBATCH --cpus-per-task=16 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/Report-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn bandit-moisesdb-nodbm "++ckpt_path='./banda/wirnm2ms/checkpoints/epoch=53-step=54299.ckpt'"
