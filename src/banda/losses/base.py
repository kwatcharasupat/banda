


from typing import Dict
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict
import torch
from torch.nn.modules.loss import _Loss

from banda.utils import BaseConfig, WithClassConfig

class LossParams(BaseConfig):
    pass
class LossConfig(WithClassConfig[LossParams]):
    weight: float
    name: str | None = None
    
class LossRegistry(type):
    
    # from https://charlesreid1.github.io/python-patterns-the-registry.html

    LOSS_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.LOSS_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.LOSS_REGISTRY)

class BaseRegisteredLoss(_Loss, metaclass=LossRegistry):
    def __init__(self, *, 
                 config: DictConfig):
        super().__init__()
        
        self.config = LossParams.model_validate(config)
        

class LossDict(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    
    total_loss: torch.Tensor
    loss_contrib: Dict[str, torch.Tensor | float]  = {}
