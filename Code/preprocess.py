"""Standard entry point for preprocessing ResPlan.pkl: `python preprocess.py ...`"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from condiffplan.data.preprocess import main

if __name__ == "__main__":
    main()
