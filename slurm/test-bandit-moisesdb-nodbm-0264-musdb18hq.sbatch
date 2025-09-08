#!/bin/bash
#SBATCH -Jtest-bandit-moisesdb-nodbm-0264-musdb18hq                    
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

srun python scripts/train.py -cn test-bandit-musdb18hq-vdbo.yaml ++ckpt_path="./banda/g39pf2r9/checkpoints/last.ckpt" ++wandb_name="test-bandit-moisesdb-nodbm-0264-musdb18hq" ++run_training=false ++run_evaluation=true
