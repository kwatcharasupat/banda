from typing import Dict, List
from pydantic import BaseModel
from torch.utils.data import Dataset

from banda.utils import WithClassConfig, BaseConfig


class DatasourceParams(BaseConfig):
    pass


class DatasourceConfig(WithClassConfig[DatasourceParams]):
    pass


class TrackIdentifier(BaseModel):
    full_path: str
    duration_samples: int | None = None
    sources: Dict[str, List[str]] | None = None


class DatasourceRegistry(type):
    # from https://charlesreid1.github.io/python-patterns-the-registry.html

    DATASOURCE_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.DATASOURCE_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.DATASOURCE_REGISTRY)


class BaseRegisteredDatasource(Dataset, metaclass=DatasourceRegistry):
    def __init__(self, *, config: DatasourceParams):
        super().__init__()
        self.config = config
