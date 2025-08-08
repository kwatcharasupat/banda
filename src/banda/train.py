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
from typing import Dict, Any, List, Tuple

from banda.utils.logging import configure_logging
from pytorch_lightning.loggers import WandbLogger
from banda.tasks.separation import SeparationTask
from banda.models.core_models.banquet_separator import Banquet
from banda.data.datamodule import SourceSeparationDataModule
import wandb
from hydra.core.hydra_config import HydraConfig
import os

from banda.utils.registry import MODELS_REGISTRY, LOSSES_REGISTRY, DATASETS_REGISTRY, TASKS_REGISTRY
from banda.metrics.metric_handler import MetricHandler # Import MetricHandler


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """
    Main training function for the banda project.

    This function orchestrates the entire training process:
    1. Configures logging and sets up the output directory.
    2. Sets the random seed for reproducibility.
    3. Instantiates the DataModule, Model, Loss Function, and Lightning Task
       based on the provided Hydra configuration.
    4. Sets up the PyTorch Lightning Trainer with callbacks and logger.
     5. Initiates model training and testing.

    Args:
        cfg (DictConfig): The Hydra configuration object containing all
                          parameters for data, model, loss, metrics, and trainer.
    """
    # Configure logging
    # Ensure the output directory exists before configuring logging
    output_dir: str = HydraConfig.get().run.dir
    os.makedirs(output_dir, exist_ok=True)

    # Create a simplified config for wandb to avoid OmegaConf resolution issues
    wandb_config_dict = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "loss": OmegaConf.to_container(cfg.loss, resolve=True),
        "metrics": OmegaConf.to_container(cfg.metrics, resolve=True),
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
        "seed": cfg.seed,
        "log_level": cfg.log_level,
        "experiment_name": cfg.experiment_name,
        "paths": {
            "output_dir": output_dir,
            "log_dir": os.path.join(output_dir, "logs"),
            "data_dir": cfg.paths.data_dir # Directly assign, it's already resolved by Hydra
        }
    }

    # Initialize wandb
    wandb.init(project="banda", config=wandb_config_dict, dir=output_dir)

    logger = hydra.utils.log
    configure_logging(
        log_level=cfg.get("log_level", "INFO"),
        log_format=cfg.get("log_format", "console"),
        log_file=os.path.join(output_dir, "train.log")
    )

    logger.info(f"Starting training with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)

    # 1. Instantiate DataModule
    datamodule: pl.LightningDataModule = SourceSeparationDataModule.from_config(cfg.data)
    
    # 2. Instantiate Model
    model: pl.LightningModule = Banquet.from_config(cfg.model)

    # 3. Instantiate Loss Function
    loss_handler: torch.nn.Module = hydra.utils.instantiate(cfg.loss)

    # 4. Instantiate Metrics
    # Instantiate the actual metric (e.g., ScaleInvariantSignalNoiseRatio)
    raw_metric = hydra.utils.instantiate(cfg.metrics)
    # Wrap it in MetricHandler
    metric_handler: MetricHandler = MetricHandler(metric_fn=raw_metric)

    # 5. Instantiate Lightning Task
    task: pl.LightningModule = SeparationTask(
        model=model,
        loss_handler=loss_handler,
        metric_handler=metric_handler,
        optimizer=cfg.optimizer # Pass the optimizer config directly
    )

    # 6. Instantiate Trainer
    trainer_config: DictConfig = cfg.trainer
    trainer_kwargs: Dict[str, Any] = {k: v for k, v in trainer_config.items() if k != 'target_'}
    wandb_logger: WandbLogger = WandbLogger(project="banda", log_model="all", save_dir=output_dir)
    trainer: pl.Trainer = pl.Trainer(
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
        raise e
    finally:
        wandb.finish()

    # 8. Test the model
    logger.info("Starting model testing...")
    try:
        trainer.test(task, datamodule=datamodule)
        logger.info("Model testing completed.")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
        raise e


if __name__ == "__main__":
    train()