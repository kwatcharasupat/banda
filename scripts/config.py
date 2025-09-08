import yaml


def make_config(
    model: str,
    dataset: str,
    stems: list[str],
    logger: str = "wandb",
    loss: str = "l1snr",
    metrics: str = "default",
    optimizer: str = "adam",
    trainer: str = "default",
    inference: str = "default",
    seed: int = 42,
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


def make_configs():
    datasets = ["musdb18hq", "moisesdb"]

    models = ["bandit", "vqbandit"]

    losses = ["l1snr"]

    stems = [
        ["vocals", "drums", "bass", "other"],
        ["vocals"],
        ["drums"],
        ["bass"],
        ["other"],
    ]

    for dataset in datasets:
        for model in models:
            for loss in losses:
                # Single stem
                for stem in stems:
                    make_config(
                        model=model,
                        dataset=dataset,
                        stems=stem,
                        loss=loss,
                    )


if __name__ == "__main__":
    import fire

    fire.Fire()
