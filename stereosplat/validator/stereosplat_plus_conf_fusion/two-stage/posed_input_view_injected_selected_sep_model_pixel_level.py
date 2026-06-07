"""Legacy path → eval/run.py (stage2 pixel_fusion separated)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.run import main

if __name__ == "__main__":
    main(
        defaults={
            "training_stage": "stage2",
            "eval_mode": "pixel_fusion",
            "architecture": "separated",
        }
    )
