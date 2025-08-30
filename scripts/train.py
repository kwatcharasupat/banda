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
from typing import Dict, Any, List, Tuple, Optional

import structlog

from banda.data.base import SourceSeparationDataModule

logger = structlog.get_logger(__name__)

@hydra.main(config_path="../experiment", version_base="1.3") # Point to the top-level config.yaml
def train(config: DictConfig) -> None:
    
    logger.info(
        "Config: ", config=config
    )
    
    pl.seed_everything(config.seed, workers=True)
    
    datamodule = SourceSeparationDataModule(config=config.data)
    
    
    for item in datamodule.train_dataloader():
        print(item)
        break
    
    for item in datamodule.val_dataloader():
        print(item)
        break
    
if __name__ == "__main__":
    train()