#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

from abc import abstractmethod
from typing import List, Union

from pydantic import BaseModel, Extra

from banda.data.types import Identifier, NumPySourceDict, TorchInputAudioDict
from banda.utils.config import ConfigWithTarget


class _TransformConfig(BaseModel, extra=Extra.allow):
    pass


class TransformConfig(ConfigWithTarget):
    config: _TransformConfig


class Transform:
    """
    Base class for data transformation operations.
    """

    @classmethod
    def from_config(cls, config: _TransformConfig):
        """
        Create a Transform instance from a configuration.

        Args:
            config (_TransformConfig): Configuration for the transform.

        Returns:
            Transform: An instance of the Transform class.
        """
        return cls(**config.model_dump())  # noqa


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
        transforms (List[Union[PreMixTransform, TransformConfig]]): List of pre-mix transformations to compose.
    """

    def __init__(self, transforms: List[Union[PreMixTransform, TransformConfig]]) -> None:
        instantiated_transforms = []
        for transform in transforms:
            if isinstance(transform, TransformConfig):
                transform_cls = transform.target_
                transform = transform_cls.from_config(
                    transform.config
                )
            instantiated_transforms.append(transform)
        self.transforms = instantiated_transforms

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
        transforms (List[Union[PostMixTransform, TransformConfig]]): List of post-mix transformations to compose.
    """

    def __init__(self, transforms: List[Union[PostMixTransform, TransformConfig]]) -> None:
        instantiated_transforms = []
        for transform in transforms:
            if isinstance(transform, TransformConfig):
                transform_cls = transform.target_
                transform = transform_cls.from_config(
                    transform.config
                )
            instantiated_transforms.append(transform)
        self.transforms = instantiated_transforms

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