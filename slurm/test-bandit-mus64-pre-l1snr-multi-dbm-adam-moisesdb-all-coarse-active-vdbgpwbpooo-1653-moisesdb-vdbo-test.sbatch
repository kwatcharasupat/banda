#!/bin/bash
#SBATCH -Jtest-bandit-mus64-pre-l1snr-multi-dbm-adam-moisesdb-all-coarse-active-vdbgpwbpooo-1653-moisesdb-vdbo-test                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1 --constraint=V100-32GB|A40|A100-40GB|A100-80GB|H100|H200|L40S
#SBATCH --cpus-per-task=24 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/test-bandit-mus64-pre-l1snr-multi-dbm-adam-moisesdb-all-coarse-active-vdbgpwbpooo-1653-moisesdb-vdbo-test-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3+pace@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn bandit-mus64-l1snrz-multi-dbm-adam-moisesdb-vdbo-test-vdbgpwbpooo ++ckpt_path=/storage/ice1/4/1/kwatchar3/banda/banda/pui49sm7/checkpoints/last.ckpt ++run_training=false ++run_evaluation=true
