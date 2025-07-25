#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch

from banda.utils.logging import configure_logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from banda.utils.logging import configure_logging
from banda.tasks.separation import SeparationTask


@hydra.main(config_path="banda/configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    # Configure logging
    configure_logging(
        log_level=cfg.get("log_level", "INFO"),
        log_format=cfg.get("log_format", "console"),
        log_file=f"{cfg.paths.log_dir}/train.log"
    )
    logger = hydra.utils.log # Use Hydra's logger for consistency

    logger.info(f"Starting training with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)

    # 1. Instantiate DataModule
    datamodule = hydra.utils.instantiate(cfg.data)

    # 2. Instantiate Model
    model = hydra.utils.instantiate(cfg.model)

    # 3. Instantiate Loss Function
    loss_fn = hydra.utils.instantiate(cfg.loss)

    # 4. Instantiate Metrics
    metric_fn = hydra.utils.instantiate(cfg.metrics)

    # 5. Instantiate Lightning Task
    task = SeparationTask(
        model=model,
        loss_fn=loss_fn,
        optimizer_config=cfg.optimizer, # Optimizer config is passed directly
        metric_fn=metric_fn
    )

    # 6. Instantiate Trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=TensorBoardLogger(save_dir=cfg.paths.log_dir, name="tensorboard_logs"),
    )

    # 7. Train the model
    logger.info("Starting model training...")
    trainer.fit(task, datamodule=datamodule)
    logger.info("Model training completed.")


if __name__ == "__main__":
    train()