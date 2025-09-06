

import os
from pathlib import Path


def make_slurm_and_submit(config_name: str, overrides: str | None = None, job_name: str | None = None):
    
    command = f"python scripts/train.py -cn {config_name}"
    
    if overrides is not None:
        command += f" {overrides}"
 
 
    slurm_template =  """#!/bin/bash
    #SBATCH -J{job_name}                    
    #SBATCH -N1 --ntasks-per-node=1          
    #SBATCH --cpu=16 --mem-per-cpu=16G       
    #SBATCH --gres=gpu:V100:1 --mem-per-gpu=32G         
    #SBATCH -t16h                            
    #SBATCH -oReport-%j.out                  # Combined output and error messages file
    #SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
    #SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications

    module load mamba
    mamba activate banda
    cd scratch/banda

    srun {command}
    """
    
    job_name = config_name.replace("/", "-").replace(".yaml", "") if job_name is None else job_name
    slurm_script = slurm_template.format(job_name=job_name, command=command)
    
    script_path = Path(f"./slurm/{job_name}.sbatch").absolute()
    print(f"Writing slurm script to {script_path}")
    if os.path.exists(script_path):
        raise ValueError(f"Slurm script {script_path} already exists. Please remove it or choose a different job name.")
    
    with open(script_path, "w") as f:
        f.write(slurm_script)
        
    os.system(f"sbatch {script_path}")

if __name__ == "__main__":
    import fire
    fire.Fire(make_slurm_and_submit)