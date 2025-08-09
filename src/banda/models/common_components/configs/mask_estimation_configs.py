from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel, Field, ConfigDict

class MaskEstimationType(str, Enum):
    OVERLAPPING = "overlapping"
    NON_OVERLAPPING = "non_overlapping"

class BaseMaskEstimationSpecsConfig(BaseModel):
    """
    Base class for mask estimation specification configurations.
    """
    mlp_dim: int
    hidden_activation: str
    hidden_activation_kwargs: Optional[Dict] = None
    complex_mask: bool
    norm_mlp_cls: str = "banda.models.common_components.mask_estimation.mask_estimation_modules.NormMLP" # Added norm_mlp_cls

class OverlappingMaskEstimationSpecsConfig(BaseMaskEstimationSpecsConfig):
    """
    Configuration for OverlappingMaskEstimationModule.
    """
    mask_estimation_type: MaskEstimationType = Field(MaskEstimationType.OVERLAPPING)
    n_freq: int

class NonOverlappingMaskEstimationSpecsConfig(BaseMaskEstimationSpecsConfig):
    """
    Configuration for MaskEstimationModule (non-overlapping).
    """
    mask_estimation_type: MaskEstimationType = Field(MaskEstimationType.NON_OVERLAPPING)

MaskEstimationConfig = Union[OverlappingMaskEstimationSpecsConfig, NonOverlappingMaskEstimationSpecsConfig]

MASK_ESTIMATION_CONFIG_MAP: Dict[MaskEstimationType, Type[MaskEstimationConfig]] = {
    MaskEstimationType.OVERLAPPING: OverlappingMaskEstimationSpecsConfig,
    MaskEstimationType.NON_OVERLAPPING: NonOverlappingMaskEstimationSpecsConfig,
}