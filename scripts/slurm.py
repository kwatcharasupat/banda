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
#SBATCH --output=./slurm-out/Report-%j.out                  # Combined output and error messages file
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


# def eval_20250908(submit: bool = False):
#     import pandas as pd

#     import wandb

#     api = wandb.Api()

#     # Project is specified by <entity/project-name>
#     runs = api.runs("kwatcharasupat-gatech/banda")

#     commands = []

#     for run in runs:
#         summary = run.summary
#         epoch = summary.get("epoch", None)
#         if epoch != 99:
#             continue

#         print(run.name, run.id, epoch)

#         ckpt_path = f"./banda/{run.id}/checkpoints/last.ckpt"
#         print(ckpt_path)

#         print(run.config["model"]["params"]["stems"])

#         test_config_name = "test-{model_type}-{dataset}-{stem_suffix}.yaml"

#         names = run.name.split("-")
#         model_type = names[0]

#         if len(names) < 4:
#             continue

#         maybe_stem = names[3]

#         if maybe_stem in ["vocals", "drums", "bass", "other"]:
#             stem_suffix = f"{maybe_stem}"
#         else:
#             stem_suffix = "vdbo"

#         for test_ds in ["musdb18hq", "moisesdb"]:
#             command = {
#                 "config_name": test_config_name.format(
#                     model_type=model_type,
#                     dataset=test_ds,
#                     stem_suffix=stem_suffix,
#                 ),
#                 "overrides": f'++ckpt_path="{ckpt_path}" ++wandb_name="test-{run.name}-{test_ds}"',
#                 "test_only": True,
#                 "job_name": f"test-{run.name}-{test_ds}",
#             }
#             commands.append(command)

#     for command in commands:
#         print(command)
#         make(**command, submit=submit)


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


if __name__ == "__main__":
    import fire

    fire.Fire()
