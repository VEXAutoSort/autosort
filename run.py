#!/usr/bin/env python3
"""AutoSort entry point.

Edit config.yaml, then run:
    python run.py                 # uses config.yaml
    python run.py --dry-run       # simulate everything, no hardware
"""
from autosort.pipeline import main

if __name__ == "__main__":
    main()
