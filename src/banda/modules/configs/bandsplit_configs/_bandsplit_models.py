from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import StrEnum

class BandDefinitionConfig(BaseModel):
    start_hz: float
    end_hz: Optional[float] = None
    bandwidth_hz: Optional[float] = None

class FixedBandsplitSpecsConfig(BaseModel):
    bands: List[BandDefinitionConfig]

class VocalBandsplitSpecsConfig(BaseModel):
    version: str = "7"

class PerceptualBandsplitSpecsConfig(BaseModel):
    n_bands: int
    f_min: Optional[float] = None # Changed to Optional and default to None
    f_max: Optional[float] = None

class MusicalBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
    pass

class MelBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
    pass

class TriangularBarkBandsplitSpecsConfig(PerceptualBandsplitSpecsConfig):
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
    TriangularBarkBandsplitSpecsConfig,
    EquivalentRectangularBandsplitSpecsConfig,
]

class BandsplitType(StrEnum):
    FIXED = "fixed" # Added FIXED to BandsplitType
    VOCAL = "vocal"
    VOCAL_V7 = "vocal_v7"
    BASS = "bass"
    DRUM = "drum"
    OTHER = "other"
    MUSICAL = "musical"
    MUSIC = "music"
    MEL = "mel"
    TRIBARK = "tribark"
    ERB = "erb"

BANDSPLIT_CONFIG_MAP = {
    BandsplitType.FIXED: FixedBandsplitSpecsConfig, # Added this line
    BandsplitType.VOCAL: VocalBandsplitSpecsConfig,
    BandsplitType.VOCAL_V7: VocalBandsplitSpecsConfig,
    BandsplitType.BASS: FixedBandsplitSpecsConfig,
    BandsplitType.DRUM: FixedBandsplitSpecsConfig,
    BandsplitType.OTHER: FixedBandsplitSpecsConfig,
    BandsplitType.MUSICAL: MusicalBandsplitSpecsConfig,
    BandsplitType.MUSIC: MusicalBandsplitSpecsConfig,
    BandsplitType.MEL: MelBandsplitSpecsConfig,
    BandsplitType.TRIBARK: TriangularBarkBandsplitSpecsConfig,
    BandsplitType.ERB: EquivalentRectangularBandsplitSpecsConfig,
}