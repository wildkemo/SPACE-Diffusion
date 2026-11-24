"""Standard entry point for training ConDiffPlan: `python train.py ...`"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from condiffplan.train import main

if __name__ == "__main__":
    main()
