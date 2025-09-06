#!/bin/bash
#SBATCH -Jbandit-musdb18hq-nodbm-drums                    
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

srun python scripts/train.py -cn bandit-musdb18hq-nodbm-drums "++ckpt_path='./banda/wh1g4lg9/checkpoints/epoch=60-step=62386.ckpt'"
