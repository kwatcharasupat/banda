from typing import Generic, TypeVar
from pydantic import ConfigDict, BaseModel


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


_GenericConfig = TypeVar("GenericDictConfig", bound=BaseConfig)


class WithClassConfig(BaseConfig, Generic[_GenericConfig]):
    cls: str
    params: _GenericConfig
