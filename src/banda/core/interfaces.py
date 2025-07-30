"""
This module defines Abstract Base Classes (ABCs) for core components in the banda codebase.
These ABCs enforce consistent APIs across different implementations, promoting modularity and extensibility.
"""

from abc import ABC, abstractmethod
import torch
from typing import Any
import torch.nn as nn
from torchmetrics import Metric

class BaseModel(ABC, nn.Module):
    """
    Abstract Base Class for all models in the banda codebase.

    All concrete model implementations should inherit from this class.
    It provides a common interface for neural network models.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseModel.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the forward pass of the model.

        Concrete implementations must override this method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The output of the model's forward pass.
        """
        pass

class BaseDataset(ABC):
    """
    Abstract Base Class for all datasets in the banda codebase.

    All concrete dataset implementations should inherit from this class.
    It provides a common interface for data loading.
    """
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Concrete implementations must override this method.

        Returns:
            The number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: # Type hint for Any will be replaced by Pydantic models later
        """
        Retrieves a sample from the dataset at the given index.

        Concrete implementations must override this method.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A single data sample.
        """
        pass

class BaseLoss(ABC, nn.Module):
    """
    Abstract Base Class for all loss functions in the banda codebase.

    All concrete loss implementations should inherit from this class.
    It provides a common interface for calculating loss values.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseLoss.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Calculates the loss.

        Concrete implementations must override this method.

        Args:
            *args: Variable length argument list, typically containing predictions and targets.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The calculated loss tensor.
        """
        pass

class BaseMetric(Metric):
    """
    Abstract Base Class for all metrics in the banda codebase.

    All concrete metric implementations should inherit from this class.
    It provides a common interface for computing evaluation metrics.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the BaseMetric.

        Args:
            *args: Variable length argument list for metric initialization.
            **kwargs: Arbitrary keyword arguments for metric initialization.
        """
        super().__init__(*args, **kwargs)

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Updates the metric's internal state with new data.

        Concrete implementations must override this method.

        Args:
            *args: Variable length argument list, typically containing predictions and targets.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """
        Computes the final metric value from the internal state.

        Concrete implementations must override this method.

        Returns:
            The computed metric tensor.
        """
        pass

class BaseQueryModel(ABC, nn.Module):
    """
    Abstract Base Class for all query encoding models.

    All concrete query model implementations should inherit from this class.
    It provides a common interface for processing query inputs.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseQueryModel.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the forward pass for the query model.

        Concrete implementations must override this method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The output of the query model's forward pass.
        """
        pass

class BaseEncoder(ABC, nn.Module):
    """
    Abstract Base Class for model encoders.

    All concrete encoder implementations should inherit from this class.
    It provides a common interface for encoding input data.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseEncoder.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the forward pass for the encoder.

        Concrete implementations must override this method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The encoded representation.
        """
        pass

class BaseDecoder(ABC, nn.Module):
    """
    Abstract Base Class for model decoders.

    All concrete decoder implementations should inherit from this class.
    It provides a common interface for decoding latent representations.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseDecoder.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the forward pass for the decoder.

        Concrete implementations must override this method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The decoded output.
        """
        pass

class BaseMaskingHead(ABC, nn.Module):
    """
    Abstract Base Class for mask estimation heads.

    All concrete masking head implementations should inherit from this class.
    It provides a common interface for predicting masks.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseMaskingHead.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the forward pass for the masking head.

        Concrete implementations must override this method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The predicted mask.
        """
        pass

class BaseTimeFrequencyModel(ABC, nn.Module):
    """
    Abstract Base Class for time-frequency processing modules.

    All concrete time-frequency model implementations should inherit from this class.
    It provides a common interface for operations in the time-frequency domain.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseTimeFrequencyModel.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the forward pass for the time-frequency model.

        Concrete implementations must override this method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The output of the time-frequency model's forward pass.
        """
        pass