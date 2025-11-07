#!/bin/bash
#SBATCH -Jvqlocobandit-mus64-narrow-4k-l1snrz-multi-dbm-adam-musdbhq18-vdbo                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --qos=pace-grade
#SBATCH --gres=gpu:1 --constraint=A100-80GB|H100
#SBATCH --cpus-per-task=24 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/vqlocobandit-mus64-narrow-4k-l1snrz-multi-dbm-adam-musdbhq18-vdbo-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3+pace@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun python scripts/train.py -cn vqlocobandit-mus64-narrow-4k-l1snrz-multi-dbm-adam-musdbhq18-vdbo ++ckpt_path=/storage/ice1/4/1/kwatchar3/banda/banda/esp0ay2i/checkpoints/last.ckpt
