#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")


class ConfigWithTarget(BaseModel):
    """
    A Pydantic model that includes a `_target_` field for Hydra-style instantiation.

    This class is designed to be inherited by configuration models that need to specify
    a Python path to a class or function that Hydra should instantiate. It also
    provides a mechanism to validate that the `_target_` points to a subclass of
    a specified `_superclass_`.

    Attributes:
        _target_ (str): The fully qualified Python path to the target class or function.
        _partial_ (bool): If True, Hydra will return a partially instantiated object.
        _recursive_ (bool): If True, Hydra will recursively instantiate nested objects.
        _convert_ (str): Controls how Hydra converts objects (e.g., "all", "partial", "none").
        _superclass_ (Type[T], optional): An optional superclass type for validation.
                                          If provided, `_target_` must point to a subclass of this type.
        config (Dict[str, Any]): A dictionary of configuration parameters for the target.
    """

    _target_: str = Field(..., alias="_target_")
    _partial_: bool = Field(False, alias="_partial_")
    _recursive_: bool = Field(True, alias="_recursive_")
    _convert_: str = Field("all", alias="_convert_")
    _superclass_: Type[T] = Field(None, exclude=True)
    config: Dict[str, Any] = Field({}, alias="config")

    @model_validator(mode="after")
    def validate_target_is_subclass(self) -> "ConfigWithTarget":
        """
        Validates that the `_target_` points to a subclass of `_superclass_` if specified.
        """
        if self._superclass_ is not None:
            try:
                # Dynamically import the target class
                module_name, class_name = self._target_.rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                target_cls = getattr(module, class_name)

                if not issubclass(target_cls, self._superclass_):
                    raise ValueError(
                        f"Target class '{self._target_}' is not a subclass of "
                        f"'{self._superclass_.__module__}.{self._superclass_.__name__}'"
                    )
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not import target class '{self._target_}': {e}")
        return self

    @property
    def target_(self) -> Any:
        """
        Dynamically imports and returns the target class or function.
        """
        module_name, class_name = self._target_.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
