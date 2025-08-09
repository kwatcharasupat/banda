#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.

import torch
import torch.nn as nn
from banda.utils.registry import QUERY_PROCESSORS_REGISTRY
from banda.data.batch_types import QuerySignal # Import QuerySignal

@QUERY_PROCESSORS_REGISTRY.register("linear")
class LinearQueryProcessor(nn.Module):
    def __init__(self, query_features: int, emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(query_features, emb_dim)

    def forward(self, query_signal: QuerySignal) -> torch.Tensor:
        """
        Processes a QuerySignal object to extract the relevant tensor and apply linear transformation.
        """
        if query_signal.audio is not None and query_signal.audio.audio is not None:
            # Assuming audio query needs to be flattened or pooled to match query_features
            # This is a placeholder; actual processing might be more complex
            processed_query = query_signal.audio.audio.mean(dim=[-1, -2]) # Example: mean over channels and samples
        elif query_signal.embedding is not None:
            processed_query = query_signal.embedding
        elif query_signal.class_label is not None:
            processed_query = query_signal.class_label
        elif query_signal.one_hot is not None:
            processed_query = query_signal.one_hot
        elif query_signal.data_dict is not None:
            # This case might need more specific handling depending on the query_processor
            # For now, assuming it takes a single tensor from the dict if applicable
            first_key = next(iter(query_signal.data_dict))
            processed_query = query_signal.data_dict[first_key]
        else:
            raise ValueError("Unsupported query type in QuerySignal for LinearQueryProcessor.")
        
        return self.linear(processed_query)

@QUERY_PROCESSORS_REGISTRY.register("mlp")
class MLPQueryProcessor(nn.Module):
    def __init__(self, query_features: int, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(query_features, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

    def forward(self, query_signal: QuerySignal) -> torch.Tensor:
        """
        Processes a QuerySignal object to extract the relevant tensor and apply MLP transformation.
        """
        if query_signal.audio is not None and query_signal.audio.audio is not None:
            # Assuming audio query needs to be flattened or pooled to match query_features
            # This is a placeholder; actual processing might be more complex
            processed_query = query_signal.audio.audio.mean(dim=[-1, -2]) # Example: mean over channels and samples
        elif query_signal.embedding is not None:
            processed_query = query_signal.embedding
        elif query_signal.class_label is not None:
            processed_query = query_signal.class_label
        elif query_signal.one_hot is not None:
            processed_query = query_signal.one_hot
        elif query_signal.data_dict is not None:
            # This case might need more specific handling depending on the query_processor
            # For now, assuming it takes a single tensor from the dict if applicable
            first_key = next(iter(query_signal.data_dict))
            processed_query = query_signal.data_dict[first_key]
        else:
            raise ValueError("Unsupported query type in QuerySignal for MLPQueryProcessor.")
        
        return self.mlp(processed_query)