# conftest.py  —  Root-level pytest configuration
"""
Shared pytest fixtures and configuration for the CICIoT23 test suite.

Adds the project root to sys.path so every test module can import
graph_builder, graph_partition, and spectral_features directly without
needing an installed package.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path for all test modules
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
