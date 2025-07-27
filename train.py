#!/usr/bin/env python3
"""Wrapper script in repo root to delegate to real trainer in tdmpc2/train.py.
Needed because Docker alias sets workdir to repo root so `python train.py`
works just like `python batch_train.py`.
"""
import sys, pathlib, runpy

repo_root = pathlib.Path(__file__).resolve().parent
real_script = repo_root / "tdmpc2" / "train.py"
if not real_script.exists():
    raise FileNotFoundError(f"Expected trainer at {real_script}")

# Put repo root and tdmpc2 on PYTHONPATH so intra-package imports resolve
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(real_script.parent))

if __name__ == "__main__":
    # Preserve argv; just adjust the script name shown in errors/usage
    sys.argv[0] = "tdmpc2/train.py"
    runpy.run_path(str(real_script), run_name="__main__") 