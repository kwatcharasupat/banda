#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import random
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
import hydra.utils
from torch import nn
import structlog
from pytorch_lightning.utilities.seed import isolate_rng
from banda.callbacks.checkpoint import (
    ModelCheckpointWithAutoRestart,
)
from banda.data.base import DataConfig, SourceSeparationDataModule
from banda.inference.handler import InferenceHandler, InferenceHandlerParams
from banda.losses.handler import LossHandler, LossHandlerConfig
from banda.metrics.handler import MetricHandler, MetricHandlerParams
from banda.models.base import ModelRegistry
from banda.system.base import SourceSeparationSystem
from banda.utils import BaseConfig, WithClassConfig
from hydra.core.hydra_config import HydraConfig

import time

logger = structlog.get_logger(__name__)


torch.set_float32_matmul_precision("high")


class TrainingConfig(BaseConfig):
    seed: int

    data: DataConfig
    model: WithClassConfig[BaseConfig]
    loss: LossHandlerConfig
    metrics: MetricHandlerParams
    inference: InferenceHandlerParams = None

    trainer: BaseConfig

    ckpt_path: str | None = None
    run_training: bool = True
    run_evaluation: bool = False


def _build_model(config: WithClassConfig[BaseConfig]) -> nn.Module:
    cls_str = config.cls

    cls = ModelRegistry.get_registry().get(cls_str, None)

    if cls is None:
        raise ValueError(
            f"Unknown model class: {cls_str}. Available classes are: {list(ModelRegistry.get_registry().keys())}"
        )

    return cls(config=config.params)


@hydra.main(
    config_path="../experiments", version_base="1.3"
)  # Point to the top-level config.yaml
def train(config: DictConfig) -> None:
    logger.info("Config: ", config=config)

    config: TrainingConfig = TrainingConfig.model_validate(config)

    pl.seed_everything(config.seed, workers=True)

    datamodule = SourceSeparationDataModule(config=config.data)

    model = _build_model(config=config.model)

    loss = LossHandler(config=config.loss)

    metric = MetricHandler(config=config.metrics)

    inference_handler = InferenceHandler(config=config.inference)

    system = SourceSeparationSystem(
        model=model,
        loss_handler=loss,
        metric_handler=metric,
        inference_handler=inference_handler,
        optimizer_config=config.optimizer,
    )

    with isolate_rng(include_cuda=False):
        linux_time = int(time.time())
        random.seed(linux_time)
        random_number = random.randint(0, 10000)

    config_name = HydraConfig.get().job.config_name
    try:
        wandb_name = config.wandb_name
    except Exception:
        wandb_name = f"{config_name}-{random_number:04d}"

    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpointWithAutoRestart(
                config_name=config_name,
                should_trigger=config.run_training,  # only trigger slurm requeue on training runs
                monitor="val/loss",
                save_last=True,
                save_on_exception=True,
                every_n_epochs=1,
                save_top_k=3,
                mode="min",
            ),
            # pl_callbacks.EarlyStopping(monitor="val/loss", patience=5, verbose=True, check_finite=False),
            pl_callbacks.ModelSummary(max_depth=2),
        ],
        logger=WandbLogger(project="banda", log_model=True, name=wandb_name),
        **config.trainer.model_dump(),
    )

    trainer.logger.log_hyperparams(config.model_dump())
    trainer.logger.save()

    ckpt_path = config.ckpt_path

    if config.run_training:
        trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
        ckpt_path = "last"

    if config.run_evaluation:
        trainer.test(system, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    train()
