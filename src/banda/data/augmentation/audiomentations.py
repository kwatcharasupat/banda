from audiomentations.augmentations.gain import Gain as _Gain
from audiomentations.augmentations.shift import Shift as _Shift
from audiomentations.augmentations.polarity_inversion import (
    PolarityInversion as _PolarityInversion,
)
from audiomentations.augmentations.seven_band_parametric_eq import SevenBandParametricEQ as _SevenBandParametricEQ
from audiomentations.augmentations.time_stretch import TimeStretch as _TimeStretch
from audiomentations.augmentations.pitch_shift import PitchShift as _PitchShift
from audiomentations.augmentations.limiter import Limiter as _Limiter
from audiomentations.augmentations.gain_transition import GainTransition as _GainTransition
from audiomentations.augmentations.clipping_distortion import ClippingDistortion as _ClippingDistortion
from audiomentations.augmentations.mp3_compression import Mp3Compression as _Mp3Compression
from audiomentations.augmentations.tanh_distortion import TanhDistortion as _TanhDistortion
from audiomentations.augmentations.repeat_part import RepeatPart as _RepeatPart
from audiomentations.augmentations.room_simulator import RoomSimulator as _RoomSimulator
from audiomentations.augmentations.time_mask import TimeMask as _TimeMask
import numpy as np

from banda.data.augmentation.base import AugmentationParams, BaseRegisteredAugmentation
from audiomentations.core.utils import convert_decibels_to_amplitude_ratio

class Gain(BaseRegisteredAugmentation, _Gain):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _Gain.__init__(self, **config.model_dump())

class MultichannelGain(BaseRegisteredAugmentation, _Gain):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _Gain.__init__(self, **config.model_dump())

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            n_channels = samples.shape[0]
            self.parameters["amplitude_ratio"] = convert_decibels_to_amplitude_ratio(
                np.random.uniform(self.min_gain_db, self.max_gain_db, size=(n_channels,))
            )

    def apply(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return samples * self.parameters["amplitude_ratio"]


class Shift(BaseRegisteredAugmentation, _Shift):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _Shift.__init__(self, **config.model_dump())


class PolarityInversion(BaseRegisteredAugmentation, _PolarityInversion):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _PolarityInversion.__init__(self, **config.model_dump())

class SevenBandParametricEQ(BaseRegisteredAugmentation, _SevenBandParametricEQ):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _SevenBandParametricEQ.__init__(self, **config.model_dump())

class TimeStretch(BaseRegisteredAugmentation, _TimeStretch):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _TimeStretch.__init__(self, **config.model_dump())

class PitchShift(BaseRegisteredAugmentation, _PitchShift):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _PitchShift.__init__(self, **config.model_dump())

class Limiter(BaseRegisteredAugmentation, _Limiter):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _Limiter.__init__(self, **config.model_dump())

class GainTransition(BaseRegisteredAugmentation, _GainTransition):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _GainTransition.__init__(self, **config.model_dump())

class ClippingDistortion(BaseRegisteredAugmentation, _ClippingDistortion):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _ClippingDistortion.__init__(self, **config.model_dump())

class Mp3Compression(BaseRegisteredAugmentation, _Mp3Compression):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _Mp3Compression.__init__(self, **config.model_dump())

class TanhDistortion(BaseRegisteredAugmentation, _TanhDistortion):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _TanhDistortion.__init__(self, **config.model_dump())

class RepeatPart(BaseRegisteredAugmentation, _RepeatPart):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _RepeatPart.__init__(self, **config.model_dump())

class RoomSimulator(BaseRegisteredAugmentation, _RoomSimulator):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _RoomSimulator.__init__(self, **config.model_dump())

class TimeMask(BaseRegisteredAugmentation, _TimeMask):
    def __init__(self, *, config: AugmentationParams):
        BaseRegisteredAugmentation.__init__(self, config=config)
        _TimeMask.__init__(self, **config.model_dump())

        