
from typing import List
from audiomentations.core.transforms_interface import BaseWaveformTransform

from banda.utils import BaseConfig, WithClassConfig

class AugmentationParams(BaseConfig):
    p: float

class AugmentationConfig(WithClassConfig[AugmentationParams]):
    pass

class AugmentationRegistry(type):
    
    # from https://charlesreid1.github.io/python-patterns-the-registry.html

    AUGMENTATION_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.AUGMENTATION_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.AUGMENTATION_REGISTRY)
    
class BaseRegisteredAugmentation(AugmentationRegistry, BaseWaveformTransform):
    def __init__(self, *, config: AugmentationParams):
        super().__init__(p=self.config.p)
        self.config = config
