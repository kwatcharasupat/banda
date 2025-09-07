from typing import Dict
from omegaconf import DictConfig
import torch
import torchmetrics as tm

from banda.utils import BaseConfig, WithClassConfig


class MetricParams(BaseConfig):
    pass


class MetricConfig(WithClassConfig[MetricParams]):
    name: str | None = None


class MetricRegistry(type):
    # from https://charlesreid1.github.io/python-patterns-the-registry.html

    METRIC_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.METRIC_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.METRIC_REGISTRY)


class BaseRegisteredMetric(metaclass=MetricRegistry):
    def __init__(self, *, config: DictConfig):
        super().__init__()

        self.config = MetricParams.model_validate(config)


MetricDict = Dict[str, torch.Tensor | float]

MetricRegistry.METRIC_REGISTRY["SignalNoiseRatio"] = tm.SignalNoiseRatio
MetricRegistry.METRIC_REGISTRY["ScaleInvariantSignalNoiseRatio"] = tm.ScaleInvariantSignalNoiseRatio