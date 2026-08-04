"""
Unit test runner under tests/unit/ for kimipy.
"""

import sys
from pathlib import Path

# Add project root to python path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tests.test_kimipy import (
    TestExpertLRUCache,
    TestKimiConfig,
    TestMXFP4Dequantization,
    TestSafetensorsReader,
    TestTrunkStreamer,
)
import unittest

if __name__ == "__main__":
    unittest.main()
