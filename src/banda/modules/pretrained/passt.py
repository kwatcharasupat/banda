import torch
import torchaudio as ta
from hear21passt.base import get_basic_model
from torch import nn


class Passt(nn.Module):
    _PASST_EMB_DIM: int = 768
    _PASST_FS: int = 32000
    _PASST_MAX_TIME_FRAMES: int = 998

    def __init__(
        self,
        emb_dim: int,
        fs: int = 44100,
    ):
        super().__init__()

        self.passt = torch.compile(
            get_basic_model(mode="embed_only", arch="openmic").eval()
        )
        self.resample = ta.transforms.Resample(
            orig_freq=fs, new_freq=self._PASST_FS
        ).eval()

        for p in self.passt.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(self._PASST_EMB_DIM, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = torch.mean(x, dim=1)
            x = self.resample(x)

            specs = self.passt.mel(x)[..., : self._PASST_MAX_TIME_FRAMES]
            specs = specs[:, None, ...]
            _, z = self.passt.net(specs)

        z = self.proj(z)
        return z
