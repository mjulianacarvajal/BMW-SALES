"""Legacy entry point; implementation lives in bmw-data-processor/src."""

import importlib.util
import sys
from pathlib import Path


PROCESSOR_ROOT = Path(__file__).resolve().parent / "bmw-data-processor"
if str(PROCESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSOR_ROOT))


def _load_main():
    main_path = PROCESSOR_ROOT / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("bmw_data_processor_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load main module from {main_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


main = _load_main()


if __name__ == "__main__":
    main()
