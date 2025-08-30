from typing import Dict, Generic, TypeVar
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch


ArrayType = TypeVar("ArrayType", torch.Tensor, np.ndarray)

class MultiDomainSignal(Dict[str, ArrayType], Generic[ArrayType]):
    pass


class SourceSeparationItem(BaseModel, Generic[ArrayType]):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    mixture: MultiDomainSignal[ArrayType] | None
    sources: Dict[str, MultiDomainSignal[ArrayType]]
    estimates: Dict[str, MultiDomainSignal[ArrayType]] | None
    
    