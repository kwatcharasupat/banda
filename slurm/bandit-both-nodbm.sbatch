#!/bin/bash
#SBATCH -Jbandit-both-nodbm                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1 # Request one GPU of any type
#SBATCH --constraint="gpu_mem>=32G"        
#SBATCH --cpus-per-task=16 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/Report-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn bandit-both-nodbm "++ckpt_path='./banda/bz83x3pb/checkpoints/epoch=53-step=55105.ckpt'"
