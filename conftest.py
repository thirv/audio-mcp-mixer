"""Pytest configuration (repository root).

Ensures the repository root is importable so test modules can import project
modules (e.g. ``ear_direct_renderer``) no matter which directory pytest is
invoked from.

Why this is needed: pytest's default (``prepend``) import mode inserts the
first directory above a test file that lacks an ``__init__.py`` -- which is
``tests/`` -- at the front of ``sys.path``, *not* the repository root. So a
test that does ``from ear_direct_renderer import ...`` only works when the CWD
happens to be the repo root. This conftest anchors the repo root on ``sys.path``
explicitly, making the whole suite robust to the current working directory.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
