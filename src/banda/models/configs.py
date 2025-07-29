from pydantic import BaseModel, ConfigDict # Import ConfigDict
from typing import Dict, List, Optional, Union

from banda.models.spectral import BandsplitSpecification # Import BandsplitSpecification (now in models folder)

# Pydantic models for Separator configuration

class STFTConfig(BaseModel):
    n_fft: int
    win_length: Optional[int] = None
    hop_length: int
    window_fn: str
    wkwargs: Optional[Dict] = None
    power: Optional[int] = None
    center: bool
    normalized: bool
    pad_mode: str
    onesided: bool

class BandsplitModuleConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) # Allow arbitrary types for BandsplitSpecification
    # band_specs will be an instantiated BandsplitSpecification object
    band_specs: BandsplitSpecification # This will be instantiated by Hydra
    require_no_overlap: bool
    require_no_gap: bool
    normalize_channel_independently: bool
    treat_channel_as_feature: bool

class TFModelConfig(BaseModel):
    tf_model_type: str
    n_tf_modules: int
    rnn_dim: int
    bidirectional: bool
    rnn_type: str
    tf_dropout: float

class MaskEstimationConfig(BaseModel):
    mlp_dim: int
    hidden_activation: str
    hidden_activation_kwargs: Optional[Dict] = None
    complex_mask: bool
    # use_freq_weights: bool # Deprecated

class SeparatorConfig(BaseModel):
    stems: List[str]
    fs: int
    emb_dim: int
    in_channel: int
    stft_config: STFTConfig
    bandsplit_config: BandsplitModuleConfig
    tfmodel_config: TFModelConfig
    mask_estim_config: MaskEstimationConfig