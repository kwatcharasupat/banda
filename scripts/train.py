#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
import hydra.utils
from typing import Dict, Any, List, Tuple, Optional
from torch import nn
import structlog

from banda.data.base import DataConfig, SourceSeparationDataModule
from banda.losses.handler import LossHandler, LossHandlerConfig
from banda.metrics.handler import MetricHandler
from banda.models.base import ModelRegistry
from banda.models.masking.dummy import DummyMaskingModel
from banda.system.base import SourceSeparationSystem
from banda.utils import BaseConfig, WithClassConfig


logger = structlog.get_logger(__name__)


torch.set_float32_matmul_precision('high')

class TrainingConfig(BaseConfig):
    seed: int 
    data: DataConfig
    model: WithClassConfig[BaseConfig]
    loss: LossHandlerConfig
    trainer: BaseConfig
    ckpt_path: str | None = None

def _build_model(config: WithClassConfig[BaseConfig]) -> nn.Module:
    cls_str = config.cls
    
    cls = ModelRegistry.get_registry().get(cls_str, None)

    if cls is None:
        raise ValueError(f"Unknown model class: {cls_str}. Available classes are: {list(ModelRegistry.get_registry().keys())}")

    return cls(config=config.params)

@hydra.main(config_path="../experiment", version_base="1.3") # Point to the top-level config.yaml
def train(config: DictConfig) -> None:
    
    logger.info(
        "Config: ", config=config
    )

    config : TrainingConfig = TrainingConfig.model_validate(config)

    pl.seed_everything(config.seed, workers=True)
    
    datamodule = SourceSeparationDataModule(config=config.data)

    model = _build_model(config=config.model)
    
    loss = LossHandler(config=config.loss)
    
    metric = MetricHandler(config=config.metrics)
    
    system = SourceSeparationSystem(
        model=model,
        loss_handler=loss,
        metric_handler=metric,
        optimizer_config=config.optimizer
    )

    trainer = pl.Trainer(
        callbacks=[
            pl_callbacks.ModelCheckpoint(monitor="val/loss", save_last=True, save_on_exception=True, every_n_epochs=1, save_top_k=3, mode="min"),
            # pl_callbacks.EarlyStopping(monitor="val/loss", patience=5, verbose=True, check_finite=False),
            pl_callbacks.ModelSummary(max_depth=2),
        ],
        logger=WandbLogger(project="banda", log_model=True),
        **config.trainer.model_dump()
    )
    
    trainer.logger.log_hyperparams(config.model_dump())
    trainer.logger.save()

    trainer.fit(system, datamodule=datamodule, ckpt_path=config.ckpt_path)

if __name__ == "__main__":
    train()