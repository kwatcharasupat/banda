from typing import Annotated, Dict, Generic, TypeVar
import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict
import torch


ArrayType = TypeVar("ArrayType", torch.Tensor, np.ndarray)


class MultiDomainSignal(Dict[str, ArrayType], Generic[ArrayType]):
    pass


# Define a function to convert the input dictionary to MultiDomainSignal
def convert_to_multi_domain_signal(v):
    if isinstance(v, dict):
        return MultiDomainSignal(v)
    return v


# Create a custom type that applies the validator
ValidatedMultiDomainSignal = Annotated[
    MultiDomainSignal[ArrayType], BeforeValidator(convert_to_multi_domain_signal)
]


class SourceSeparationItem(BaseModel, Generic[ArrayType]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mixture: ValidatedMultiDomainSignal[ArrayType] | None
    sources: Dict[str, ValidatedMultiDomainSignal[ArrayType]]
    estimates: Dict[str, ValidatedMultiDomainSignal[ArrayType]] | None

    n_samples: torch.Tensor = torch.tensor(-1, dtype=torch.int64)


SourceSeparationBatch = SourceSeparationItem[torch.Tensor]
