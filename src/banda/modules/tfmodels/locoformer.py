# from https://github.com/merlresearch/tf-locoformer/blob/main/standalone/tflocoformer_separator.py

# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, List, Union

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V

from banda.modules.tfmodels.tfmodel import TimeFrequencyModellingModule

from torch.utils.checkpoint import checkpoint_sequential

from banda.utils import BaseConfig
is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")

class LocoFormerParams(BaseConfig):
    n_modules: int = 8
    emb_dim: int = 128

    pe_freq : Any = None
    pe_time : Any = None

    # general setup
    emb_dim: int = 128
    norm_type: str = "rmsgroupnorm"
    num_groups: int = 4
    tf_order: str = "ft"

    # self-attention related
    n_heads: int = 4
    flash_attention: bool = False
    attention_dim: int = 128

    # ffn related
    ffn_type: str | List[str] = ["swiglu_conv1d", "swiglu_conv1d"]
    ffn_hidden_dim: int | List[int] = [256, 256]
    conv1d_kernel: int = 4
    conv1d_shift: int = 1
    dropout: float = 0.0
    eps: float = 1.0e-5

class LocoFormerTFModel(TimeFrequencyModellingModule):
    def __init__(self, *, config):
        config = LocoFormerParams.model_validate(config)
        super().__init__(config=config)

        self.config : LocoFormerParams
        print("Building LocoFormer TF model with config:", self.config)
        print("n_modules:", self.config.n_modules)


        seqband: List[nn.Module] = []
        # no need to *2, as we are doing both freq and time in each block
        for _ in range(self.config.n_modules):
            seqband += [
                TFLocoformerBlock(
                    pe_freq=self.config.pe_freq,
                    pe_time=self.config.pe_time,
                    # general setup
                    emb_dim=self.config.emb_dim,
                    norm_type=self.config.norm_type,
                    num_groups=self.config.num_groups,
                    # self-attention related
                    n_heads=self.config.n_heads,
                    flash_attention=self.config.flash_attention,
                    attention_dim=self.config.attention_dim,
                    # ffn related
                    ffn_type=self.config.ffn_type,
                    ffn_hidden_dim=self.config.ffn_hidden_dim,
                    conv1d_kernel=self.config.conv1d_kernel,
                    conv1d_shift=self.config.conv1d_shift,
                    dropout=self.config.dropout,
                    eps=self.config.eps,
                ),
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

        # tf locoformer expects (batch, emb_dim, n_bands, n_time)
        z = z.permute(0, 3, 1, 2)  # (batch, emb_dim, n_bands, n_time)

        z = checkpoint_sequential(
            self.seqband, self.config.n_modules, z, use_reentrant=False
        )

        # now it is (batch, emb_dim, n_bands, n_time)
        z = z.permute(0, 2, 3, 1)  # back to (batch, n_bands, n_time, emb_dim)

        return z  # (batch, n_bands, n_time, emb_dim)


class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        pe_freq=None,
        pe_time=None,
        # general setup
        emb_dim=128,
        norm_type="rmsgrouporm",
        num_groups=4,
        tf_order="ft",
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

        self.freq_path = LocoformerBlock(
            pe_freq,
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = LocoformerBlock(
            pe_time,
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, input):
        """TF-Locoformer forward.

        input: torch.Tensor
            Input tensor, (n_batch, channel, n_frame, n_freq)
        """

        if self.tf_order == "ft":
            output = self.freq_frame_process(input)
        else:
            output = self.frame_freq_process(input)

        return output

    def freq_frame_process(self, input):
        output = input.movedim(1, -1)  # (B, T, Q_old, H)
        output = self.freq_path(output)

        output = output.transpose(1, 2)  # (B, F, T, H)
        output = self.frame_path(output)
        return output.transpose(-1, 1)

    def frame_freq_process(self, input):
        # Input tensor, (n_batch, hidden, n_frame, n_freq)
        output = input.transpose(1, -1)  # (B, F, T, H)
        output = self.frame_path(output)

        output = output.transpose(1, 2)  # (B, T, F, H)
        output = self.freq_path(output)
        return output.movedim(-1, 1)


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        pe=None,
        # general setup
        emb_dim=128,
        norm_type="rmsgrouporm",
        num_groups=4,
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        FFN = {"swiglu_conv1d": SwiGLUConvDeconv1d}
        Norm = {"layernorm": nn.LayerNorm, "rmsgroupnorm": RMSGroupNorm}
        assert norm_type in Norm, norm_type

        self.macaron_style = isinstance(ffn_type, list) and len(ffn_type) == 2
        if self.macaron_style:
            assert (
                isinstance(ffn_hidden_dim, list) and len(ffn_hidden_dim) == 2
            ), "Two FFNs required when using Macaron-style model"
        else:
            if not isinstance(ffn_type, list):
                ffn_type = [ffn_type]
            if not isinstance(ffn_hidden_dim, list):
                ffn_hidden_dim = [ffn_hidden_dim]

        # initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type[::-1], ffn_hidden_dim[::-1]):
            assert f_type in FFN, f_type
            if norm_type == "rmsgroupnorm":
                self.ffn_norm.append(Norm[norm_type](num_groups, emb_dim, eps=eps))
            else:
                self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
            self.ffn.append(
                FFN[f_type](
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                )
            )

        # initialize self-attention
        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            pe=pe,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        B, T, F, C = x.shape

        if self.macaron_style:
            # FFN before self-attention
            # Note that this implementation does not include the 1/2 factor described in the paper.
            # Experiments in the paper did use the 1/2 factor, but we removed it by mistake in this
            # implementation. We found that the 1/2 factor does not impact final performance, and
            # thus decided to keep the current implementation for consistency with the pre-trained
            # models that we provide.
            input_ = x
            output = self.ffn_norm[-1](x)  # [B, T, F, C]
            output = self.ffn[-1](output)  # [B, T, F, C]
            output = output + input_
        else:
            output = x

        # Self-attention
        input_ = output
        output = self.attn_norm(output)
        output = output.reshape([B * T, F, C])
        output = self.attn(output)
        output = output.reshape([B, T, F, C]) + input_

        # FFN after self-attention
        input_ = output
        output = self.ffn_norm[0](output)  # [B, T, F, C]
        output = self.ffn[0](output)  # [B, T, F, C]
        output = output + input_

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        pe=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = pe
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        if self.rope is not None:
            raise NotImplementedError("Rotary positional encoding is not implemented in this version.")

        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        # with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
        # with torch.nn.attention.sdpa_kernel(**self.flash_attention_config):
        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.amp.autocast("cuda", enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision, RMSGroupNorm
        is applied to each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                Number of dimensions
            eps: float
                Small constant to avoid division by zero.
            bias: bool
                Whether to add a bias term. RMSNorm does not use bias.

        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output