#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#


import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch

from banda.utils.logging import configure_logging
from pytorch_lightning.loggers import TensorBoardLogger
from hydra.core.hydra_config import HydraConfig
import os

from banda.tasks.separation import SeparationTask
from banda.data.datamodule import SourceSeparationDataModule
from banda.models.separator import Separator
from banda.losses.multi_resolution_l1_snr import MultiResolutionSTFTLoss


@hydra.main(config_path="configs", config_name="train_bandit_musdb18hq_test", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    # Configure logging
    # Ensure the output directory exists before configuring logging
    output_dir = HydraConfig.get().run.dir
    os.makedirs(output_dir, exist_ok=True)

    logger = hydra.utils.log # Use Hydra's logger for consistency
    configure_logging(
        log_level=cfg.get("log_level", "INFO"),
        log_format=cfg.get("log_format", "console"),
        log_file=os.path.join(output_dir, "train.log")
    )

    logger.info(f"Starting training with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)

    # 1. Instantiate DataModule
    datamodule = SourceSeparationDataModule.from_config(cfg.data)
    
    # 2. Instantiate Model
    model = Separator.from_config(cfg.model)

    # 3. Instantiate Loss Function
    loss_fn = MultiResolutionSTFTLoss.from_config(cfg.loss)

    # 4. Instantiate Metrics (handled within SeparationTask.from_config)

    # 5. Instantiate Lightning Task
    task = SeparationTask.from_config(cfg)

    # 6. Instantiate Trainer
    trainer_config = cfg.trainer
    trainer_kwargs = {k: v for k, v in trainer_config.items() if k != 'target_'}
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=output_dir, name="tensorboard_logs"),
        **trainer_kwargs
    )

    # 7. Train the model
    logger.info("Starting model training...")
    try:
        print(datamodule.__class__)
        trainer.fit(task, datamodule=datamodule)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        # Optionally, re-raise the exception if you want the script to fail
        raise e


if __name__ == "__main__":
    train()