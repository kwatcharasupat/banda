import os

from ..utils import BaseStemAlias


class MedleyDBStemAlias(BaseStemAlias):
    """MedleyDB stem alias class."""

    def get(self, stem_name: str) -> str:
        """Get the stem alias for a given stem name.

        Args:
            stem_name (str): The name of the stem.

        Returns:
            str: The stem alias.
        """
        return super().get(stem_name, stem_name.lower().replace(" ", "_"))


medleydb_stem_alias = MedleyDBStemAlias(
    yaml_path=os.path.join(os.path.dirname(__file__), "stem_alias.yaml")
)
