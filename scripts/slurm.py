import os
from pathlib import Path

import yaml


slurm_template = """#!/bin/bash
#SBATCH -J{job_name}                    
#SBATCH -N1 --ntasks-per-node=1     
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1 --constraint="V100-32GB|A40|A100-40GB|A100-80GB|H100|H200|L40S"
#SBATCH --cpus-per-task=16 --mem-per-cpu=16G     
#SBATCH --time=16:00:00                            
#SBATCH --output=./slurm-out/{job_name}.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=kwatchar3@gatech.edu # E-mail address for notifications
#SBATCH --signal=SIGTERM@120

cd $SLURM_SUBMIT_DIR

module load mamba
mamba activate banda

nvidia-smi

srun {command}
"""


def make_config(
    model: str,
    dataset: str,
    stems: list[str],
    logger: str = "wandb",
    loss: str = "l1snr-multi",
    metrics: str = "default",
    optimizer: str = "adam",
    trainer: str = "default",
    inference: str = "default",
    seed: int = 42,
    run_slurm: bool = False,
):
    defaults = [
        {
            "model": model,
        },
        {
            "data": dataset,
        },
        {
            "logger": logger,
        },
        {
            "loss": loss,
        },
        {
            "metrics": metrics,
        },
        {
            "optimizer": optimizer,
        },
        {
            "trainer": trainer,
        },
        {
            "inference": inference,
        },
        "_self_",
    ]

    config = {
        "defaults": defaults,
        "model": {
            "params": {
                "stems": stems,
            }
        },
        "metrics": {"stems": "${..model.params.stems}"},
        "seed": seed,
        "hydra": {
            "searchpath": [
                "file:///home/kwatchar3/projects/banda/configs",
                "file:///storage/ice1/4/1/kwatchar3/banda/configs",
            ]
        },
    }

    if len(stems) == 1:
        stem_code = stems[0]
    else:
        stem_code = "".join([s[0] for s in stems])

    config_name = f"{model}-{loss}-{optimizer}-{dataset}-{stem_code}"

    with open(f"experiments/{config_name}.yaml", "w") as f:
        yaml.dump(config, f)

    if run_slurm:
        make(
            config_name=config_name,
            job_name=config_name,
            submit=True,
        )

    return config_name


def make_configs(run_slurm: bool = False):
    datasets = ["musdb18hq", "moisesdb"]

    models = ["bandit-mus64", "vqbandit-mus64", "bandroformer-mus64", "bandmamba-mus64"]
    # models = ["bandit-mus64"]

    losses = ["l1snr-multi"]

    stems = [
        # ["vocals", "drums", "bass", "other"],
        ["vocals"],
        ["drums"],
        ["bass"],
        ["other"],
    ]

    for dataset in datasets:
        for model in models:
            for loss in losses:
                for stem in stems:
                    make_config(
                        model=model,
                        dataset=dataset,
                        stems=stem,
                        loss=loss,
                        run_slurm=run_slurm,
                    )


def multi_eval_vdbo(group_filter: str = "training runs - to be tested", max_epoch: int = 99, submit: bool = False):
    import pandas as pd

    import wandb

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("kwatcharasupat-gatech/banda")

    slurm_paths = []

    for run in runs:
        summary = run.summary
        epoch = summary.get("epoch", None)
        if epoch != max_epoch:
            continue

        if run.group != group_filter:
            continue

        print(f"Run: {run.name}, id: {run.id}, epoch: {epoch}")
        print(f"Group: {run.group}")

        ckpt_path = Path(f"./banda/{run.id}/checkpoints/last.ckpt").absolute()
        if not ckpt_path.exists():
            print(f"Checkpoint {ckpt_path} does not exist, skipping...")
            continue

        print(f"Checkpoint {ckpt_path} exists.")

        original_config_name = "-".join(run.name.split("-")[:-1]) + ".yaml"
        print(f"Original config name: {original_config_name}")

        for test_ds in ["musdb18hq-vdbo-test", "moisesdb-vdbo-test"]:

            test_job_name = f"test-{run.name}-{test_ds}"

            original_stems = run.config["model"]["params"]["stems"]
            print(f"Original stems: {original_stems}")

            model = "-".join(run.name.split("-")[:2])
            print(f"Model: {model}")
            
            config_name = make_config(
                model=model,
                dataset=test_ds,
                stems=original_stems,
                run_slurm=False,
            )

            print(f"New config name: {config_name}")

            print(f"Submitting test job: {test_job_name}")
            slurm_path = make(config_name=config_name,
                    ckpt_path=str(ckpt_path),
                    test_only=True,
                    job_name=test_job_name,
                    submit=submit)
            
            slurm_paths.append(slurm_path)

    sbatch_commands = "\n".join([f"sbatch {p}" for p in slurm_paths])
    print("To submit all jobs, run the following commands:")
    print(sbatch_commands)


def make(
    config_name: str,
    ckpt_path: str | None = None,
    overrides: str | None = None,
    job_name: str | None = None,
    test_only: bool = False,
    submit: bool = True,
):
    command = f"python scripts/train.py -cn {config_name}"

    if ckpt_path is not None:
        command += f" ++ckpt_path={ckpt_path}"

    if overrides is not None:
        command += f" {overrides}"

    if test_only:
        command += " ++run_training=false ++run_evaluation=true"

    job_name = (
        config_name.replace("/", "-").replace(".yaml", "")
        if job_name is None
        else job_name
    )

    if test_only and not job_name.startswith("test-"):
        job_name = "test-" + job_name

    slurm_script = slurm_template.format(job_name=job_name, command=command)

    script_path = Path(f"./slurm/{job_name}.sbatch").absolute()
    print(f"Writing slurm script to {script_path}")
    # if os.path.exists(script_path):
    #     raise ValueError(f"Slurm script {script_path} already exists. Please remove it or choose a different job name.")

    with open(script_path, "w") as f:
        f.write(slurm_script)

    print(slurm_script)

    if submit:
        os.system(f"sbatch {script_path}")


    return script_path

def continue_run(group_filter: str = "training runs - to be continued",
                run_name_regex: str = None, 
                 submit: bool = False):
    import wandb

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("kwatcharasupat-gatech/banda")

    slurm_paths = []

    for run in runs:

        if run.group != group_filter:
            continue

        if run_name_regex is not None:
            import re

            if not re.search(run_name_regex, run.name):
                continue

        if run.state != "finished":
            continue

        print(f"Run: {run.name}, id: {run.id}")
        ckpt_path = Path(f"./banda/{run.id}/checkpoints/last.ckpt").absolute()
        if not ckpt_path.exists():
            raise ValueError(f"Checkpoint {ckpt_path} does not exist.")

        print(f"Checkpoint {ckpt_path} exists.")

        config_name = "-".join(run.name.split("-")[:-1]) + ".yaml"
        print(f"Config name: {config_name}")


        slurm_path = make(
            config_name=config_name,
            ckpt_path=str(ckpt_path),
            submit=submit,
        )

        slurm_paths.append(slurm_path)

    sbatch_commands = "\n".join([f"sbatch {p}" for p in slurm_paths])
    print("To submit all jobs, run the following commands:")
    print(sbatch_commands)


if __name__ == "__main__":
    import fire

    fire.Fire()
