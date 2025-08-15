from typing import Dict, Type, TYPE_CHECKING
from banda.models.modules.configs.bandsplit_configs._bandsplit_models import BandsplitType

if TYPE_CHECKING:
    from banda.models.modules.spectral_components.spectral_base import BandsplitSpecification

class BandsplitRegistry:
    _registry: Dict[BandsplitType, Type["BandsplitSpecification"]] = {}

    @classmethod
    def register(cls, bandsplit_type: BandsplitType, bandsplit_class: Type["BandsplitSpecification"]):
        if bandsplit_type in cls._registry:
            raise ValueError(f"BandsplitType {bandsplit_type} already registered.")
        cls._registry[bandsplit_type] = bandsplit_class

    @classmethod
    def get(cls, bandsplit_type: BandsplitType) -> Type["BandsplitSpecification"]:
        bandsplit_class = cls._registry.get(bandsplit_type)
        if bandsplit_class is None:
            raise ValueError(f"BandsplitType {bandsplit_type} not found in registry.")
        return bandsplit_class

# Instantiate the registry
bandsplit_registry = BandsplitRegistry()