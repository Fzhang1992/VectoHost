#!/usr/bin/env python3
"""Command-line entry point for five-fold model evaluation (not feature extraction)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from vh3.train import main
if __name__ == '__main__':
    main()
