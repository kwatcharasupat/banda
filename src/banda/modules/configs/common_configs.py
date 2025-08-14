from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union, Dict, Any, List

from banda.models.common_components.spectral_components.spectral_base import STFTConfig
from banda.models.common_components.spectral_components.bandsplit import BandsplitModuleConfig
from banda.models.common_components.mask_estimation.mask_estimation_modules import MaskEstimationConfig
from banda.models.common_components.time_frequency_models.tf_models import BaseTFModelConfig, RNNTFModelConfig, TransformerTFModelConfig, MambaTFModelConfig

class BaseSeparatorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_")
    stft_config: STFTConfig
    bandsplit_module_config: BandsplitModuleConfig
    mask_estimation_config: MaskEstimationConfig
    time_frequency_model_config: Union[RNNTFModelConfig, TransformerTFModelConfig, MambaTFModelConfig, BaseTFModelConfig]

class BanditSeparatorConfig(BaseSeparatorConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Bandit-specific fields, if any, would go here
    # For now, it inherits all fields from BaseSeparatorConfig

class BanquetSeparatorConfig(BaseSeparatorConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Banquet-specific fields, if any, would go here
    # For now, it inherits all fields from BaseSeparatorConfig

# No model_rebuild() calls needed here, as they will be handled by the top-level TrainConfig.
