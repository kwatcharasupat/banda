#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.

import torch
import torch.nn as nn
from banda.utils.registry import QUERY_PROCESSORS_REGISTRY

@QUERY_PROCESSORS_REGISTRY.register("linear")
class LinearQueryProcessor(nn.Module):
    def __init__(self, query_features: int, emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(query_features, emb_dim)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.linear(query)

@QUERY_PROCESSORS_REGISTRY.register("mlp")
class MLPQueryProcessor(nn.Module):
    def __init__(self, query_features: int, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(query_features, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.mlp(query)