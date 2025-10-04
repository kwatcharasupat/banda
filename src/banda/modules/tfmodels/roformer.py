from typing import List
from omegaconf import DictConfig
import torch

from banda.modules.tfmodels.tfmodel import TimeFrequencyModellingModule
from torch import nn

from banda.modules.utils import Transpose

from torch.utils.checkpoint import checkpoint_sequential


from torch.nn.utils.parametrizations import weight_norm
from torch.nn import functional as F
# from torchtune.modules import RotaryPositionalEmbeddings

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional


from torch import Tensor


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [bsz, seq_len, num_heads, head_dim]
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, n_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not. When
        # input_pos is provided, we're in inference mode
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, n_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [1, s, 1, n_d // 2, 2]
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, n_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, n_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

class RoPETransformerParams(DictConfig):
    n_modules: int = 6
    emb_dim: int = 128
    hidden_dim: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 4096


class RoFormerTFModel(TimeFrequencyModellingModule):
    def __init__(
        self,
        *,
        config: DictConfig,
    ) -> None:
        super().__init__(config=config)

        seqband: List[nn.Module] = []
        for _ in range(2 * self.config.n_modules):
            seqband += [
                ResidualRoPETransformer(
                    emb_dim=self.config.emb_dim,
                    hidden_dim=self.config.hidden_dim,
                    n_heads=self.config.n_heads,
                    dropout=self.config.dropout,
                    max_seq_len=self.config.max_seq_len,
                ),
                Transpose(dim0=1, dim1=2),
            ]

        self.seqband = nn.Sequential(*seqband)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SeqBandModellingModule.

        Args:
            z (torch.Tensor): Input tensor. Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            torch.Tensor: Output tensor. Shape: (batch, n_bands, n_time, emb_dim)
        """

        z = checkpoint_sequential(
            self.seqband, self.config.n_modules, z, use_reentrant=False
        )
        return z  # (batch, n_bands, n_time, emb_dim)


class NormFF(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.combined = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Sequential(
                weight_norm(nn.Linear(emb_dim, emb_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                weight_norm(nn.Linear(emb_dim, emb_dim)),
                nn.Dropout(dropout),
            ),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for NormFF.

        Args:
            xb (torch.Tensor): Subband spectrogram. Shape: (batch, n_time, in_chan * bw * 2)

        Returns:
            torch.Tensor: Subband embedding. Shape: (batch, n_time, emb_dim)
        """
        return xb + self.combined(xb)


class ResidualRoPETransformer(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.rope_attention: nn.Module = RoPEAttentionModule(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.fc: nn.Module = NormFF(emb_dim=emb_dim, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.rope_attention(z)
        z = self.fc(z)
        return z


class RoPEAttentionModule(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)

        self.proj = weight_norm(nn.Linear(emb_dim, hidden_dim * 3))

        self.n_heads = n_heads
        self.emb_dim_per_head = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0, (
            f"emb_dim must be divisible by n_heads, got {hidden_dim} and {n_heads}"
        )
        self.dropout = dropout

        self.ff = weight_norm(nn.Linear(hidden_dim, emb_dim))

        self.rope = RotaryPositionalEmbeddings(
            dim=self.emb_dim_per_head, max_seq_len=max_seq_len
        )

        self.hidden_dim = hidden_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, n_uncrossed, n_across, emb_dim = z.shape

        z_ = self.norm(z)
        # (batch, n_uncrossed, n_across, emb_dim)

        z_ = z_.reshape(-1, n_across, emb_dim)
        # (batch * n_uncrossed, n_across, emb_dim)

        q, k, v = (
            self.proj(z_)
            .view(
                batch * n_uncrossed, n_across, self.n_heads, self.emb_dim_per_head * 3
            )
            .chunk(3, dim=-1)
        )
        # each is (batch * n_uncrossed, n_across, n_heads, head_dim)

        qhe = self.rope(q)
        khe = self.rope(k)

        qhe = qhe.transpose(1, 2).contiguous()
        khe = khe.transpose(1, 2).contiguous()
        vh = v.transpose(1, 2).contiguous()

        attn = F.scaled_dot_product_attention(
            qhe,
            khe,
            vh,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (batch', n_heads, n_across, head_dim)

        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch * n_uncrossed, n_across, self.hidden_dim)
        )  # (batch', n_across, hidden_dim)

        attn = self.ff(attn)
        attn = attn.view(batch, n_uncrossed, n_across, emb_dim)

        z = z + attn

        return z
