from typing import List
from audiomentations.core.composition import Compose as _Compose

from banda.data.augmentation.base import (
    AugmentationConfig,
    AugmentationRegistry,
    BaseRegisteredAugmentation,
)
from banda.utils import BaseConfig


class CompositionConfig(BaseConfig):
    augmentations: List[AugmentationConfig]
    num_transforms: int | tuple[int, int | None] | None = None
    p: float = 1.0
    shuffle: bool = True


class Compose(_Compose):
    def __init__(self, *, config: CompositionConfig):
        augmentations = self._build_augmentation(config)
        super().__init__(transforms=augmentations, p=config.p, shuffle=config.shuffle)

    @staticmethod
    def _build_augmentation(
        config: CompositionConfig,
    ) -> List[BaseRegisteredAugmentation]:
        augmentations = []

        for augmentation in config.augmentations:
            cls_str = augmentation.cls
            params = augmentation.params

            cls = AugmentationRegistry.get_registry().get(cls_str)
            if cls is not None:
                augmentations.append(cls(config=params))

        return augmentations
