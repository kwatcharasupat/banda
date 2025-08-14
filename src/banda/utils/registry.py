"""
This module implements a generic Registry class for dynamic component registration and retrieval.
"""

from typing import Any, Dict, Type, Callable

import structlog

logger = structlog.get_logger(__name__)

# Decorator-based model registration
class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, name=None):
        def decorator(model_class):
            nonlocal name
            if name is None:
                name = model_class.__name__
            self._registry[name] = model_class

            logger.info(
                f"Registered model '{name}'"
                f" with class {model_class.__name__}"
                f" on registry '{self.name}'"
            )

            return model_class
        return decorator

    def get(self, name):
        model_class = self._registry.get(name)
        if model_class:
            return model_class # Changed to return the class, not an instance
        else:
            raise ValueError(f"'{name}' not registered with registry '{self.name}'. "
                             f"Available models: {', '.join(self._registry.keys())}"
                             )


# Instantiate global registries
ModelRegistry = Registry("model")
DatasetRegistry = Registry("dataset")
LossRegistry = Registry("loss")
MetricRegistry = Registry("metric")
OptimizerRegistry = Registry("optimizer")