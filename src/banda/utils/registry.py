"""
This module implements a generic Registry class for dynamic component registration and retrieval.
"""

from typing import Any, Dict, Type, Callable

class Registry:
    """
    A generic registry class to store and retrieve components by name.
    """

    def __init__(self, name: str):
        """
        Initializes the Registry with a given name.

        Args:
            name: The name of the registry.
        """
        self._name = name
        self._data: Dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        """
        A decorator for registering a class with the registry.

        Args:
            name: The name to register the class under.

        Returns:
            A decorator function that registers the class.

        Raises:
            ValueError: If a component with the same name is already registered.
        """
        def decorator(cls: Type) -> Type:
            """
            The actual decorator function that registers the class.

            Args:
                cls: The class to be registered.

            Returns:
                The registered class.

            Raises:
                ValueError: If a component with the same name is already registered.
            """
            if name in self._data:
                raise ValueError(f"Component '{name}' already registered in {self._name} registry.")
            self._data[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Any:
        """
        Retrieves a registered component by its name.

        Args:
            name: The name of the component to retrieve.
            
        Returns:
            The registered component (class or function).

        Raises:
            KeyError: If no component with the given name is registered.
        """
        if name not in self._data:
            raise KeyError(f"Component '{name}' not found in {self._name} registry.")
        return self._data[name]

    def __contains__(self, name: str) -> bool:
        """
        Checks if a component is registered in the registry.

        Args:
            name: The name of the component to check.

        Returns:
            True if the component is registered, False otherwise.
        """
        return name in self._data

    def __repr__(self) -> str:
        """
        Returns a string representation of the registry.
        """
        return f"<Registry '{self._name}' with {len(self._data)} components>"

# Instantiate global registries
MODELS_REGISTRY = Registry("Models")
DATASETS_REGISTRY = Registry("Datasets")
LOSSES_REGISTRY = Registry("Losses")
METRICS_REGISTRY = Registry("Metrics")
OPTIMIZERS_REGISTRY = Registry("Optimizers")
QUERY_MODELS_REGISTRY = Registry("Query Models")
COLLATE_FUNCTIONS_REGISTRY = Registry("Collate Functions")
TASKS_REGISTRY = Registry("Tasks")
QUERY_PROCESSORS_REGISTRY = Registry("Query Processors") # New registry for query processors