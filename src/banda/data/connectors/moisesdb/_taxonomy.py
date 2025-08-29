

import os

from .._taxonomy import BaseTaxonomy

moisesdb_taxonomy = BaseTaxonomy(
    taxonomy_path=os.path.join(os.path.dirname(__file__), "taxonomy.yaml")
)
