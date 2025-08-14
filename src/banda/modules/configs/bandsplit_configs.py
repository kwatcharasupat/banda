from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from omegaconf import DictConfig # Keep for now, will remove later if possible

class BandDefinitionConfig(BaseModel):
    start_hz: float
    end_hz: Optional[float] = None
    bandwidth_hz: Optional[float] = None

class FixedBandsplitSpecsConfig(BaseModel):
    bands: List[BandDefinitionConfig]

class VocalBandsplitSpecsConfig(FixedBandsplitSpecsConfig):
    version: str = "7" # Default version

# Add other fixed bandsplit configs if they have specific parameters
# For now, they can just use FixedBandsplitSpecsConfig

class PerceptualBandsplitSpecsConfig(BaseModel):
    n_bands: int
    f_min: float = 0.0
    f_max: Optional[float] = None

class MusicalBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
    pass # No additional fields

class MelBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
    pass

class BarkBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
    pass

class TriangularBarkBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
    pass

class MiniBarkBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
    pass

class EquivalentRectangularBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
    pass

# Union type for all bandsplit configs
BandsplitConfig = Union[
    FixedBandsplitSpecsConfig,
    VocalBandsplitSpecsConfig,
    PerceptualBandsplitSpecsConfig,
    MusicalBandsplitSpecsConfig,
    MelBandsplitSpecsConfig,
    BarkBandsplitSpecsConfig,
    TriangularBarkBandsplitSpecsConfig,
    MiniBarkBandsplitSpecsConfig,
    EquivalentRectangularBandsplitSpecsConfig,
]
from enum import StrEnum

class BandsplitType(StrEnum):
    VOCAL = "vocal"
    BASS = "bass"
    DRUM = "drum"
    OTHER = "other"
    MUSICAL = "musical"
    MEL = "mel"
    BARK = "bark"
    TRIBARK = "tribark"
    ERB = "erb"
    MINIBARK = "minibark"

BANDSPLIT_CONFIG_MAP = {
    BandsplitType.VOCAL: VocalBandsplitSpecsConfig,
    BandsplitType.BASS: FixedBandsplitSpecsConfig,
    BandsplitType.DRUM: FixedBandsplitSpecsConfig,
    BandsplitType.OTHER: FixedBandsplitSpecsConfig,
    BandsplitType.MUSICAL: MusicalBandsplitSpecsConfig,
    BandsplitType.MEL: MelBandsplitSpecsConfig,
    BandsplitType.BARK: BarkBandsplitSpecsConfig,
    BandsplitType.TRIBARK: TriangularBarkBandsplitSpecsConfig,
    BandsplitType.ERB: EquivalentRectangularBandsplitSpecsConfig,
    BandsplitType.MINIBARK: MiniBarkBandsplitSpecsConfig,
}