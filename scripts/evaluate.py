"""Run the same greedy tool-enabled validation used during SIPO training."""
import sys
from train import main

if __name__ == "__main__":
    sys.exit(main([*sys.argv[1:], "--eval-only"]))
