#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch
import hydra.utils

from banda.utils.logging import configure_logging
from pytorch_lightning.loggers import WandbLogger
import wandb
from hydra.core.hydra_config import HydraConfig
import os

from banda.tasks.separation import SeparationTask
from banda.data.datamodule import SourceSeparationDataModule
from banda.models.separator import Separator
from banda.losses.multi_resolution_l1_snr import MultiResolutionSTFTLoss


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
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
    model = hydra.utils.instantiate(cfg.model) # Pass the entire config

    # 3. Instantiate Loss Function
    loss_fn = MultiResolutionSTFTLoss.from_config(cfg.loss)

    # 4. Instantiate Metrics (handled within SeparationTask.from_config)

    # 5. Instantiate Lightning Task
    task = SeparationTask.from_config(cfg) # Reverted to original call

    # 6. Instantiate Trainer
    trainer_config = cfg.trainer
    trainer_kwargs = {k: v for k, v in trainer_config.items() if k != 'target_'}
    wandb_logger = WandbLogger(project="banda", log_model="all", save_dir=output_dir)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=os.path.join(output_dir, "checkpoints"),
                filename="{epoch}-{step}-{val_loss:.4f}",
                save_on_train_epoch_end=False,
            )
        ],
        **trainer_kwargs
    )

    # Move task to the correct device before training
    # This is crucial for ensuring all model components and buffers (like STFT window)
    # are on the correct device before the sanity check and training begins.
    if trainer.accelerator == "mps":
        task.to(trainer.device)

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

    # 8. Test the model
    logger.info("Starting model testing...")
    try:
        trainer.test(task, datamodule=datamodule)
        logger.info("Model testing completed.")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
        raise e


wandb.finish()


if __name__ == "__main__":
    train()