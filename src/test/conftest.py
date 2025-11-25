import os
import sys

# Get the directory of this conftest.py file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to project root: src/test -> src -> root
PROJECT_ROOT = os.path.dirname(os.path.dirname(TEST_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
