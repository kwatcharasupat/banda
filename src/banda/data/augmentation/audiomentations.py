from audiomentations.augmentations.gain import Gain as _Gain
from audiomentations.augmentations.shift import Shift as _Shift
from audiomentations.augmentations.polarity_inversion import PolarityInversion as _PolarityInversion

from banda.data.augmentation.base import AugmentationParams, BaseRegisteredAugmentation


class Gain(BaseRegisteredAugmentation, _Gain):
    def __init__(self, *, config: AugmentationParams):
        super().__init__(config=config)

class Shift(BaseRegisteredAugmentation, _Shift):
    def __init__(self, *, config: AugmentationParams):
        super().__init__(config=config)

class PolarityInversion(BaseRegisteredAugmentation, _PolarityInversion):
    def __init__(self, *, config: AugmentationParams):
        super().__init__(config=config)
