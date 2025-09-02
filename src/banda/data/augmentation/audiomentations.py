from audiomentations.augmentations.gain import Gain as _Gain
from audiomentations.augmentations.shift import Shift as _Shift
from audiomentations.augmentations.polarity_inversion import PolarityInversion as _PolarityInversion

from banda.data.augmentation.base import AugmentationParams, BaseRegisteredAugmentation


class Gain(BaseRegisteredAugmentation, _Gain):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _Gain.__init__(self, **config.model_dump())

class Shift(BaseRegisteredAugmentation, _Shift):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _Shift.__init__(self, **config.model_dump())

class PolarityInversion(BaseRegisteredAugmentation, _PolarityInversion):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _PolarityInversion.__init__(self, **config.model_dump())
