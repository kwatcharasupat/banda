#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch
import hydra.utils
from typing import Dict, Any, List, Tuple, Optional
import sys
import os

# Add the parent directory of 'banda' to sys.path to enable package-relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from banda.utils.logging import configure_logging
from pytorch_lightning.loggers import WandbLogger
from banda.tasks.separation import SeparationTask
from banda.models.core_models.banquet_separator import Banquet
from banda.models.core_models.bandit_separator import Bandit # Import Bandit
from banda.data.datamodule import SourceSeparationDataModule, DataModuleConfig # Corrected import
import wandb
from hydra.core.hydra_config import HydraConfig

from banda.utils.registry import MODELS_REGISTRY, LOSSES_REGISTRY, DATASETS_REGISTRY, TASKS_REGISTRY
from banda.metrics.metric_handler import MetricHandler # Import MetricHandler
from banda.configs.train_configs import TrainConfig, TrainerConfig, LoggerConfig, OptimizerConfig # Import Pydantic configs
from banda.metrics.metric_handler import MetricsConfig
from banda.losses.separation_loss_handler import LossConfig
from banda.data.datamodule import DatasetConfig # Import Pydantic configs
from banda.models.common_components.configs.common_configs import BaseSeparatorConfig, BanditSeparatorConfig, BanquetSeparatorConfig # Import BaseSeparatorConfig
from banda.models.common_components.time_frequency_models.tf_models import RNNTFModelConfig, TransformerTFModelConfig, MambaTFModelConfig # Import TFModelConfigs

@hydra.main(config_path="configs", config_name="config", version_base="1.3") # Point to the top-level config.yaml
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
    # Print the entire configuration as requested by the user
    print(f"Full Hydra Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Convert the DictConfig to a plain Python dictionary without resolving interpolations
    # This prevents Pydantic from trying to resolve interpolations prematurely
    cfg_dict_for_pydantic = OmegaConf.to_container(cfg, resolve=False)

    # Print the cfg_dict_for_pydantic after _group_ handling for inspection
    print(f"cfg_dict_for_pydantic after _group_ handling:\n{OmegaConf.to_yaml(OmegaConf.create(cfg_dict_for_pydantic))}")

    # Instantiate TrainConfig using model_validate with the modified dictionary
    train_config: TrainConfig = TrainConfig.model_validate(cfg_dict_for_pydantic)

    # Get output directory from HydraConfig after TrainConfig is instantiated
    # This ensures Hydra has set up the run directory
    output_dir: str = HydraConfig.get().run.dir
    os.makedirs(output_dir, exist_ok=True)
    log_dir: str = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Manually assign the resolved paths to the train_config object
    # Ensure train_config.paths is initialized as a mutable dictionary
    train_config.paths = {
        'output_dir': output_dir,
        'log_dir': log_dir,
        'data_root': cfg.paths.data_root # Get data_root directly from cfg
    }

    # Initialize wandb
    wandb.init(
        project=train_config.logger.project,
        config=train_config.model_dump(), # Use model_dump for wandb config
        dir=train_config.paths['output_dir'] # Use the resolved output_dir
    )

    configure_logging(log_level=train_config.log_level)

    logging.info(f"Starting training with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    pl.seed_everything(train_config.seed)

    # 1. Instantiate DataModule
    datamodule: pl.LightningDataModule = SourceSeparationDataModule.from_config(train_config.data)
    
    # 2. Instantiate Model
    # Use the appropriate model class based on the Pydantic config type
    if isinstance(train_config.model, BanquetSeparatorConfig):
        model: pl.LightningModule = Banquet.from_config(train_config.model)
    elif isinstance(train_config.model, BanditSeparatorConfig):
        model: pl.LightningModule = Bandit.from_config(train_config.model)
    else:
        # This allows for generalization to any new separator model that inherits from BaseSeparator
        # and has a corresponding Pydantic config inheriting from BaseSeparatorConfig.
        # The model's from_config method should be registered in MODELS_REGISTRY.
        model_cls = MODELS_REGISTRY.get(train_config.model.__class__.__name__.replace("SeparatorConfig", "").lower())
        if model_cls:
            model: pl.LightningModule = model_cls.from_config(train_config.model)
        else:
            raise ValueError(f"Unsupported model configuration type or model not registered: {type(train_config.model)}")

    # 3. Instantiate Loss Function
    loss_handler: LossConfig = LossConfig.from_config(train_config.loss) # Changed to LossConfig
    # 4. Instantiate Metrics
    metric_handler: MetricsConfig = MetricsConfig.from_config(train_config.metrics) # Changed to MetricsConfig

    # 5. Instantiate Lightning Task
    task: pl.LightningModule = SeparationTask.from_config(
        model=model,
        loss_config=train_config.loss,
        metrics_config=train_config.metrics,
        optimizer_config=train_config.optimizer
    )

    # 6. Instantiate Trainer
    trainer_kwargs: Dict[str, Any] = train_config.trainer.model_dump(exclude={"_target_"})
    wandb_logger: WandbLogger = WandbLogger(
        project=train_config.logger.project,
        log_model=train_config.logger.log_model,
        save_dir=train_config.paths['output_dir'] # Use the resolved output_dir
    )
    trainer: pl.Trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=os.path.join(train_config.paths['output_dir'], "checkpoints"), # Use the resolved output_dir
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
    logging.info("Starting model training...")
    try:
        print(datamodule.__class__)
        trainer.fit(task, datamodule=datamodule)
        logging.info("Model training completed.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise e
    finally:
        wandb.finish()

    # 8. Test the model
    logging.info("Starting model testing...")
    try:
        trainer.test(task, datamodule=datamodule)
        logging.info("Model testing completed.")
    except Exception as e:
        logging.error(f"An error occurred during testing: {e}")
        raise e


if __name__ == "__main__":
    train()