"""
Regression tests guarding the tools.py lazy-loading refactor.

The whole point of the refactor ("Tools lazy-loading refactor" commit) was to
keep the heavy dependencies -- ``torch``, ``transformers`` and ``database_hf``
-- OUT of the module import path so that ``import tools`` stays fast and free
of side effects (CLAP model download, HF cache access, torch init, etc.).

To be deterministic, each test runs ``import tools`` in a fresh, isolated
interpreter (a subprocess). This guarantees pytest's own imports cannot pollute
``sys.modules`` and lets us measure import time in a clean process -- the
accurate metric for the original "< 2s" goal.

We assert:
  1. importing ``tools`` does NOT pull any heavy dep into ``sys.modules``
     (the primary guard -- hard fail), and
  2. the import completes quickly (a generous cap, kept well above the real
     target so this does not become a flaky timing test in CI).
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HEAVY_MODULES = ("torch", "transformers", "database_hf")

# NOTE: built with string concatenation (NOT .format) so that the `{dt:.4f}`
# placeholder below stays a literal for the child process to evaluate as its
# own f-string. A .format() over the whole template would try to resolve `dt`
# as a format key and raise KeyError.
_CHILD = textwrap.dedent(
    """
    import sys, time

    t0 = time.perf_counter()
    import tools  # noqa: F401  (target of the regression guard)
    dt = time.perf_counter() - t0

    heavy = [m for m in """ + repr(list(HEAVY_MODULES)) + """ if m in sys.modules]
    if heavy:
        # Exit with a distinctive, non-zero code + machine-readable marker.
        print("HEAVY_LOADED=" + ",".join(heavy))
        sys.exit(2)

    print(f"IMPORT_TIME={dt:.4f}")
    print("OK")
    """
)


def _run_child():
    """Import ``tools`` in a clean interpreter and capture the outcome."""
    env = dict(os.environ)
    # Ensure the repo root is importable regardless of how pytest is invoked.
    env["PYTHONPATH"] = REPO_ROOT.as_posix() + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )


def test_import_tools_does_not_load_heavy_deps():
    """Primary guard: `import tools` must not import torch/transformers/database_hf."""
    res = _run_child()
    assert "OK" in res.stdout and res.returncode == 0, (
        "import tools pulled heavy dependencies into sys.modules as a side "
        f"effect.\nstdout={res.stdout!r}\nstderr={res.stderr!r}"
    )
    assert "HEAVY_LOADED" not in res.stdout


def test_import_tools_is_fast():
    """Secondary guard: `import tools` stays fast (target < 2s, cap 3s to avoid flakiness)."""
    res = _run_child()
    assert res.returncode == 0, (
        f"import tools failed in a clean interpreter.\n"
        f"stdout={res.stdout!r}\nstderr={res.stderr!r}"
    )
    line = next(
        (ln for ln in res.stdout.splitlines() if ln.startswith("IMPORT_TIME=")),
        None,
    )
    assert line is not None, f"missing IMPORT_TIME marker in stdout={res.stdout!r}"
    dt = float(line.split("=", 1)[1])
    assert dt < 3.0, f"import tools took {dt:.3f}s (expected < 3.0s; target < 2s)"
