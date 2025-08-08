from typing import Dict, Union
from pydantic import BaseModel, Field
import torch.nn as nn
from omegaconf import DictConfig

class SingleLossConfig(BaseModel):
    """
    Configuration for a single loss function.
    """
    fn: DictConfig = Field(..., description="The Hydra configuration for the loss function module.")
    weight: float = Field(1.0, description="The weight for this loss function.")

class LossCollectionConfig(BaseModel):
    """
    Configuration for a collection of loss functions.
    """
    losses: Dict[str, SingleLossConfig] = Field(..., description="A dictionary of loss configurations.")