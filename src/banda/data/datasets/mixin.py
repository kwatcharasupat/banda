import inspect

from typing import Type, TypeVar, Generic, Protocol, List, Dict, Any
from ...typing.batch import UnbatchedInput
from ...typing.signal import Signal, SignalDict
from ..typing import GenericIdentifier, TorchInputAudioDict
from .base import SourceSeparationDataset


from structlog import get_logger

logger = get_logger(__name__)


class FixedStemMixin:
    """
    Mixin class for datasets with fixed stems.
    """

    def __init__(
        self,
        *,
        allowed_stems: List[str],
    ) -> None:
        """
        Initializes the FixedStemMixin.

        Args:
            allowed_stems (List[str]): List of allowed stems.
        """
        self.allowed_stems = allowed_stems

    def __getitem__(self, index: int) -> UnbatchedInput:
        """
        Retrieves the dataset item at the specified index.

        Args:
            index (int): The index of the dataset item.

        Returns:
            UnbatchedInput: The dataset item containing mixture, sources, and metadata.
        """
        if not isinstance(self, SourceSeparationDataset):
            raise TypeError(
                "FixedStemMixin must be used with a SourceSeparationDataset"
            )

        identifier = self.get_identifier(index)
        audio_dict: TorchInputAudioDict = self.get_audio_dict(
            stems=self.allowed_stems, identifier=identifier
        )
        return UnbatchedInput(
            mixture=Signal(audio=audio_dict.mixture),
            sources=SignalDict(
                {
                    stem: Signal(audio=source)
                    for stem, source in audio_dict.sources.items()
                }
            ),
            metadata=identifier,
        )

    @classmethod
    def from_dataset(
        cls: Type["FixedStemMixin"],
        dataset: Type[SourceSeparationDataset],
        allowed_stems: List[str],
        **kwargs: Any,
    ) -> Type[SourceSeparationDataset]:
        """
        Creates a new dataset class with FixedStemMixin mixed in.

        Args:
            dataset (Type[SourceSeparationDataset]): The dataset class to wrap.
            allowed_stems (List[str]): List of allowed stems.
            **kwargs (Any): Keyword arguments to pass to the dataset's __init__ method.

        Returns:
            Type[SourceSeparationDataset]: A new dataset class with FixedStemMixin mixed in.
        """
        dataset_kwargs = set(inspect.signature(dataset.__init__).parameters.keys())
        # Remove 'self' and 'kwargs' from the set of valid arguments
        dataset_kwargs.discard("self")
        dataset_kwargs.discard("kwargs")

        fwd_kwargs: Dict[str, Any] = {}
        invalid_kwargs: List[str] = []

        for kwarg in kwargs:
            if kwarg not in dataset_kwargs:
                invalid_kwargs.append(kwarg)
                continue
            fwd_kwargs[kwarg] = kwargs[kwarg]

        if invalid_kwargs:
            msg = (
                f"Keyword arguments {invalid_kwargs} are not valid for the dataset. "
                f"Valid arguments are: {dataset_kwargs}"
            )
            raise ValueError(msg)

        class MixedDataset(cls, dataset):  # type: ignore
            def __init__(self, **kwargs: Any) -> None:
                FixedStemMixin.__init__(self, allowed_stems=allowed_stems)
                dataset.__init__(self, **kwargs)

        MixedDataset.__name__ = f"{dataset.__name__}WithFixedStems"
        return MixedDataset(allowed_stems=allowed_stems, **fwd_kwargs)
