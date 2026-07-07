# Root conftest so `eval` resolves as a namespace package when running
# pytest from the repo root (e.g. `uv run --with pytest pytest eval/pit/tests`).
# Adding the repo root to sys.path lets Python's implicit namespace-package
# machinery find `eval.pit.config`.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
