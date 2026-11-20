"""Run every module's synthetic self-test: python tests/run_all.py (from the repo root).

Each test prints PASS/FAIL lines; a stub that is not implemented yet shows as TODO.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imple import lsq, perturbation, svd, tikhonov  # noqa: E402

ok = True
for mod in (svd, lsq, tikhonov, perturbation):
    print(f"\n== {mod.__name__}")
    try:
        ok &= mod._self_test()
    except NotImplementedError:
        print("TODO  not implemented yet")
        ok = False
sys.exit(0 if ok else 1)
