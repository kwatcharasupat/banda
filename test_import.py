import sys
import os

# Add the project root to the Python path to ensure modules are discoverable
# Assuming the script is run from the project root or a subdirectory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import banda.models.modules.configs.bandsplit_configs._bandsplit_models as bandsplit_models
    print("Successfully imported _bandsplit_models module!")
    if hasattr(bandsplit_models, 'ERBBandsplitSpecsConfig'):
        print("ERBBandsplitSpecsConfig found within _bandsplit_models module!")
    else:
        print("ERBBandsplitSpecsConfig NOT found within _bandsplit_models module.")
        print(f"Available attributes in _bandsplit_models: {dir(bandsplit_models)}")
except ImportError as e:
    print(f"Failed to import _bandsplit_models: {e}")
    print("sys.path:")
    for p in sys.path:
        print(f"- {p}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
