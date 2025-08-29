import os
from typing import List
import yaml


with open(os.path.join(os.path.dirname(__file__), "validation_files.yaml")) as f:
    validation_files: List[str] = yaml.safe_load(f)
