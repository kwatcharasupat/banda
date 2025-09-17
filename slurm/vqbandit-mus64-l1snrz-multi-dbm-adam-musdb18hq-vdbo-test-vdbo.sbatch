#!/bin/bash
#SBATCH -Jvqbandit-mus64-l1snrz-multi-dbm-adam-musdb18hq-vdbo-test-vdbo                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1 --constraint=V100-32GB|A40|A100-40GB|A100-80GB|H100|H200|L40S
#SBATCH --cpus-per-task=24 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/vqbandit-mus64-l1snrz-multi-dbm-adam-musdb18hq-vdbo-test-vdbo-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3+pace@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn vqbandit-mus64-l1snrz-multi-dbm-adam-musdb18hq-vdbo-test-vdbo
