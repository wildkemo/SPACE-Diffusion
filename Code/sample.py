"""Standard entry point for sampling floorplans: `python sample.py ...`"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from condiffplan.sample import main

if __name__ == "__main__":
    main()
