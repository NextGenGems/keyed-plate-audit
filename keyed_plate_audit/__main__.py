"""Entry point: python -m keyed_plate_audit ..."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
