import os

from ..utils import BaseStemAlias


moisesdb_stem_alias = BaseStemAlias(
    yaml_path=os.path.join(os.path.dirname(__file__), "stem_alias.yaml")
)
