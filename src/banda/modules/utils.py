from torch import nn
import torch


class Transpose(nn.Module):
    def __init__(self, *, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0: int = dim0
        self.dim1: int = dim1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.transpose(self.dim0, self.dim1)


# class IdentityNArgs(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x, *args, **kwargs):
#         return x


# class Constant(nn.Module):
#     def __init__(self, value) -> None:
#         super().__init__()
#         self.value = value

#     def forward(self, *args, **kwargs):
#         return self.value
