#!/bin/bash
#SBATCH -Jbandit-musdb18hq-nodbm-other                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1 --constraint="V100-32GB|A40|A100-40GB|A100-80GB|H100|H200|L40S"
#SBATCH --cpus-per-task=16 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/Report-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn bandit-musdb18hq-nodbm-other "++ckpt_path='./banda/2ngt4sge/checkpoints/epoch=60-step=61616.ckpt'"
