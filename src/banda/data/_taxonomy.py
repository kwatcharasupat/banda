from pydantic import BaseModel
from omegaconf import OmegaConf
from itertools import chain
from typing import Dict, List, Set


class TaxonomySchema(BaseModel):
    instruments: Dict[str, List[str]]


class BaseTaxonomy:
    def __init__(self, *, taxonomy_path: str) -> None:
        taxonomy_data = OmegaConf.load(taxonomy_path)
        self.taxonomy = TaxonomySchema.model_validate(taxonomy_data)

    @property
    def coarse_level_instruments(self) -> List[str]:
        return list(self.taxonomy.instruments.keys())

    @property
    def fine_level_instruments(self) -> List[str]:
        return list(chain(*self.taxonomy.instruments.values()))

    @property
    def coarse_to_fine(self) -> Dict[str, List[str]]:
        return self.taxonomy.instruments

    @property
    def fine_to_coarse(self) -> Dict[str, str]:
        return {k: kk for kk, v in self.coarse_to_fine.items() for k in v}

    @property
    def all_level_instruments(self) -> List[str]:
        return list(
            self.coarse_level_instruments_set.union(self.fine_level_instruments_set)
        )

    @property
    def coarse_level_instruments_set(self) -> Set[str]:
        return set(self.coarse_level_instruments)

    @property
    def fine_level_instruments_set(self) -> Set[str]:
        return set(self.fine_level_instruments)
