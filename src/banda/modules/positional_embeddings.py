import torch
from torch import nn
import math

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, dim, max_seq_len_bands, max_seq_len_time):
        super().__init__()
        # For bands dimension
        inv_freq_bands = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) # Changed dim // 2 to dim
        self.register_buffer("inv_freq_bands", inv_freq_bands)
        # For time dimension
        inv_freq_time = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) # Changed dim // 2 to dim
        self.register_buffer("inv_freq_time", inv_freq_time)

        self.max_seq_len_bands = max_seq_len_bands
        self.max_seq_len_time = max_seq_len_time
        self.dim = dim

    def forward(self, x, num_band_patches, num_time_patches):
        # x: (batch, num_patches, emb_dim)
        # num_band_patches: actual number of patches along band dimension (int)
        # num_time_patches: actual number of patches along time dimension (int)

        # Generate coordinates for each element in the flattened sequence
        # Assuming the flattening was done row-major (time varies fastest, then bands)
        # The coordinates should be based on the actual number of patches, not max_seq_len
        band_coords = torch.arange(0, num_band_patches, device=x.device).repeat_interleave(num_time_patches)
        time_coords = torch.arange(0, num_time_patches, device=x.device).repeat(num_band_patches)
        
        # Generate frequencies for band and time dimensions
        # Use pre-computed inv_freq_bands and inv_freq_time, but apply to current coords
        freqs_bands = torch.einsum("i,j->ij", band_coords.float(), self.inv_freq_bands) # Removed slicing
        freqs_time = torch.einsum("i,j->ij", time_coords.float(), self.inv_freq_time) # Removed slicing

        # Combine frequencies for 2D rotation
        freqs_2d = torch.cat((freqs_bands, freqs_time), dim=-1)
        
        # Apply 2D RoPE to the input tensor
        return apply_rotary_pos_emb(x, freqs_2d.unsqueeze(0))