#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
from pydantic import BaseModel, Field, ConfigDict

# from banda.utils.registry import LOSSES_REGISTRY # Removed as per previous instructions

class LossConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_") # Changed _target_ to target_
    name: str
    weight: float = 1.0
    params: Optional[Dict[str, Any]] = None

class SeparationLossHandler(nn.Module):
    def __init__(self, loss_configs: List[LossConfig]):
        super().__init__()
        self.loss_fns = nn.ModuleDict()
        self.loss_weights = {}

        for loss_cfg in loss_configs:
            loss_fn_class = self._get_loss_class(loss_cfg.name)
            loss_fn = loss_fn_class(**(loss_cfg.params or {}))
            self.loss_fns[loss_cfg.name] = loss_fn
            self.loss_weights[loss_cfg.name] = loss_cfg.weight

    def _get_loss_class(self, name: str):
        # This is a placeholder. In a real scenario, you'd map names to actual loss classes.
        # For example, using a registry or a dictionary lookup.
        if name == "mse":
            return nn.MSELoss
        elif name == "l1":
            return nn.L1Loss
        else:
            raise ValueError(f"Unknown loss function: {name}")

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        for loss_name, loss_fn in self.loss_fns.items():
            # Assuming predictions and targets dicts have keys matching loss names or a common key
            # This logic might need to be more sophisticated depending on your loss functions
            if loss_name in predictions and loss_name in targets:
                loss = loss_fn(predictions[loss_name], targets[loss_name])
            elif "mixture" in predictions and "mixture" in targets: # Example for a common case
                loss = loss_fn(predictions["mixture"], targets["mixture"])
            else:
                raise KeyError(f"Loss '{loss_name}' cannot find corresponding predictions/targets.")
            
            total_loss += loss * self.loss_weights[loss_name]
        return total_loss

    @classmethod
    def from_config(cls, config: Union[LossConfig, List[LossConfig]]) -> "SeparationLossHandler":
        if isinstance(config, list):
            return cls(config)
        else:
            return cls([config])

# No model_rebuild() calls needed here