"""Legacy dev path → eval/run.py (pose-injection whole, no pixel fusion)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.run import main

if __name__ == "__main__":
    main(
        defaults={
            "training_stage": "stage1",
            "eval_mode": "pixel_fusion",
            "architecture": "whole",
        }
    )
