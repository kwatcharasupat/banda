
from typing import Dict, List, Tuple
from omegaconf import DictConfig
import torch
from torch.nn.modules.loss import _Loss

from banda.data.item import SourceSeparationBatch

from banda.losses.base import BaseRegisteredLoss, LossConfig, LossDict, LossRegistry
from banda.utils import BaseConfig

class LossHandlerConfig(BaseConfig):
    losses: List[LossConfig]

class LossHandler(_Loss):
    def __init__(self, *, 
                 config: DictConfig):
        super().__init__()
        self.config = LossHandlerConfig.model_validate(config)
        
        self._build_losses()
        
        
    def forward(self, batch: SourceSeparationBatch) -> LossDict:

        loss_contribs = {}
        total_loss = 0.0

        for name, (loss, weight) in self.losses.items():
            loss_val : LossDict = loss(batch)
            total_loss += loss_val.total_loss * weight

            with torch.no_grad():
                loss_contribs[name] = loss_val.total_loss
                for k, v in loss_val.loss_contrib.items():
                    loss_contribs[f"{name}/{k}"] = v

        loss_dict = LossDict(
            total_loss=total_loss,
            loss_contrib=loss_contribs
        )

        return loss_dict

    def _build_losses(self):
        
        losses :  Dict[str, Tuple[BaseRegisteredLoss, float]] = {}

        for loss_config in self.config.losses:

            weight = loss_config.weight

            cls_str = loss_config.cls
            cls = LossRegistry.get_registry().get(cls_str, None)
            if cls is None:
                raise ValueError(f"Loss class '{cls_str}' not found in registry. \n Available classes: {LossRegistry.get_registry().keys()}")
            
            name = loss_config.name if loss_config.name is not None else cls_str

            loss = cls(config=loss_config.params)
            losses[name] = (loss, weight)

        self.losses = losses
