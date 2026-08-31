"""
Make the repository root importable so tests can use the same package
imports the application does (`backend.app...`, `data_pipeline...`).
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
