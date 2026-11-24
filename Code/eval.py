"""Standard entry point for evaluating ConDiffPlan: `python eval.py ...`"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from condiffplan.eval import main

if __name__ == "__main__":
    main()
