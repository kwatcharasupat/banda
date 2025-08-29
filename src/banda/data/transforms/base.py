from abc import abstractmethod
from typing import List

from ..typing import Identifier, NumPySourceDict, TorchInputAudioDict


class Transform:
    """
    Base class for data transformation operations.
    """


class PreMixTransform(Transform):
    """
    Abstract base class for pre-mix transformations applied to source data.

    Methods:
        __call__: Applies the transformation to the source data.

    Raises:
        NotImplementedError: If the `__call__` method is not implemented.
    """

    @abstractmethod
    def __call__(
        self, *, sources: NumPySourceDict, identifier: Identifier
    ) -> NumPySourceDict:
        """
        Apply the transformation to the source data.

        Args:
            sources (NumPySourceDict): Dictionary containing source data.
            identifier (Identifier): Identifier for the source data.

        Returns:
            NumPySourceDict: Transformed source data.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__ method."
        )


class PostMixTransform(Transform):
    """
    Abstract base class for post-mix transformations applied to audio data.

    Methods:
        __call__: Applies the transformation to the audio data.

    Raises:
        NotImplementedError: If the `__call__` method is not implemented.
    """

    @abstractmethod
    def __call__(
        self, *, audio_dict: TorchInputAudioDict, identifier: Identifier
    ) -> TorchInputAudioDict:
        """
        Apply the transformation to the audio data.

        Args:
            audio_dict (TorchInputAudioDict): Dictionary containing audio data.
            identifier (Identifier): Identifier for the audio data.

        Returns:
            TorchInputAudioDict: Transformed audio data.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__ method."
        )


class IdentityPreMixTransform(PreMixTransform):
    """
    Identity transformation for pre-mix operations.
    """

    def __call__(
        self, *, sources: NumPySourceDict, identifier: Identifier
    ) -> NumPySourceDict:
        """
        Return the input sources unchanged.

        Args:
            sources (NumPySourceDict): Dictionary containing source data.
            _ (Identifier): Unused identifier.

        Returns:
            NumPySourceDict: Unchanged source data.
        """
        return sources


class IdentityPostMixTransform(PostMixTransform):
    """
    Identity transformation for post-mix operations.
    """

    def __call__(
        self, *, audio_dict: TorchInputAudioDict, identifier: Identifier
    ) -> TorchInputAudioDict:
        """
        Return the input audio dictionary unchanged.

        Args:
            audio_dict (TorchInputAudioDict): Dictionary containing audio data.
            _ (Identifier): Unused identifier.

        Returns:
            TorchInputAudioDict: Unchanged audio data.
        """
        return audio_dict


class ComposePreMixTransforms(PreMixTransform):
    """
    Compose multiple pre-mix transformations into a single transformation.

    Args:
        transforms (List[PreMixTransform]): List of pre-mix transformations to compose.
    """

    def __init__(self, transforms: List[PreMixTransform]) -> None:
        self.transforms = transforms

    def __call__(
        self, *, sources: NumPySourceDict, identifier: Identifier
    ) -> NumPySourceDict:
        """
        Apply all composed transformations sequentially.

        Args:
            sources (NumPySourceDict): Dictionary containing source data.
            identifier (Identifier): Identifier for the source data.

        Returns:
            NumPySourceDict: Transformed source data.
        """
        for transform in self.transforms:
            sources = transform(sources=sources, identifier=identifier)
        return sources


class ComposePostMixTransforms(PostMixTransform):
    """
    Compose multiple post-mix transformations into a single transformation.

    Args:
        transforms (List[PostMixTransform]): List of post-mix transformations to compose.
    """

    def __init__(self, transforms: List[PostMixTransform]) -> None:
        self.transforms = transforms

    def __call__(
        self, *, audio_dict: TorchInputAudioDict, identifier: Identifier
    ) -> TorchInputAudioDict:
        """
        Apply all composed transformations sequentially.

        Args:
            audio_dict (TorchInputAudioDict): Dictionary containing audio data.
            identifier (Identifier): Identifier for the audio data.

        Returns:
            TorchInputAudioDict: Transformed audio data.
        """
        for transform in self.transforms:
            audio_dict = transform(audio_dict=audio_dict, identifier=identifier)
        return audio_dict
