#!/bin/bash
#SBATCH -J{job_name}                    
#SBATCH -N1 --ntasks-per-node=1          
#SBATCH --cpu=16 --mem-per-cpu=16G                
#SBATCH -t16h                            
#SBATCH --gres=gpu:V100:1
#SBATCH -oReport-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications

module load mamba
mamba activate banda
cd scratch/banda

srun python $1