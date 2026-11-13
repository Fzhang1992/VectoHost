#!/usr/bin/env python3
"""Predict new viral sequences with a newly trained and saved classifier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from vh3.predict import main
if __name__ == '__main__':
    main()
