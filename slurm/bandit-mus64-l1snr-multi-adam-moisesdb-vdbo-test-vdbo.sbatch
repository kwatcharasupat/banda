#!/bin/bash
#SBATCH -Jbandit-mus64-l1snr-multi-adam-moisesdb-vdbo-test-vdbo                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1 --constraint=V100-32GB|A40|A100-40GB|A100-80GB|H100|H200|L40S
#SBATCH --cpus-per-task=16 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/bandit-mus64-l1snr-multi-adam-moisesdb-vdbo-test-vdbo-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn bandit-mus64-l1snr-multi-adam-moisesdb-vdbo-test-vdbo
