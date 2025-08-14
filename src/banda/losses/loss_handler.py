#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import ast
from typing import Any, Dict, List, Optional, Union
from omegaconf import ListConfig
import torch
from torch import nn
from pydantic import BaseModel, Field, ConfigDict

from banda.data.batch_types import SeparationBatch
from banda.utils.registry import LossRegistry

# from banda.utils.registry import LOSSES_REGISTRY # Removed as per previous instructions

class LossConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_") # Changed _target_ to target_

    name: str
    weight: float = 1.0
    params: Optional[Dict[str, Any]] = None

class LossContribs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    contribs: Dict[str, torch.Tensor] 
    total: torch.Tensor



class LossHandler(nn.Module):
    def __init__(self, loss_configs: List[LossConfig]):
        super().__init__()
        self.loss_fns = nn.ModuleDict()
        self.loss_weights = {}
        self.loss_domains = {}

        for loss_cfg in loss_configs:
            loss_fn = LossRegistry.get(loss_cfg.target_)

            if loss_fn is None:
                raise ValueError(f"Loss {loss_cfg.name} is not registered.")
            self.loss_fns[loss_cfg.name] = loss_fn(**loss_cfg.params if loss_cfg.params else {})
            self.loss_weights[loss_cfg.name] = loss_cfg.weight


    def forward(self, batch: SeparationBatch) -> LossContribs:
        total_loss = 0.0
        losses = {}

        for loss_name, loss_fn in self.loss_fns.items():
            loss_weight = self.loss_weights[loss_name]
            loss_contribs : LossContribs = loss_fn(batch)
            total_loss += loss_weight * loss_contribs.total

            # for logging purposes
            with torch.no_grad():
                for key, subloss in loss_contribs.contribs.items():
                    losses[f"{loss_name}/{key}"] = subloss

        return LossContribs(contribs=losses, total=total_loss)
    
    @classmethod
    def from_config(cls, config: ListConfig):
        return cls(config)