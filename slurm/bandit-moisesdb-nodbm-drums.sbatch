#!/bin/bash
#SBATCH -Jbandit-moisesdb-nodbm-drums                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:gpu-h100:1 # Request one GPU of any type
#SBATCH --cpus-per-task=16 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/Report-%j.out   # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn bandit-moisesdb-nodbm-drums "++ckpt_path='./banda/l521n1f8/checkpoints/epoch=58-step=60416.ckpt'"
