"""Click-to-run entry point: open this file and press Run.

Running ``cartpole/cli.py`` directly does not work. Python puts the *script's own
folder* first on ``sys.path``, so from inside ``cartpole/`` the statement
``import cartpole`` finds nothing and the program dies with a ModuleNotFoundError.

This wrapper lives at the project root, so the package is importable either way,
and it defaults to the benchmark when no arguments are given.

    python run.py               # same as: python -m cartpole.cli bench
    python run.py all           # benchmark + robustness sweep + swing-up GIF
    python run.py robustness    # model-mismatch study only
    python run.py animate       # swing-up GIF only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cartpole.cli import main  # noqa: E402  (import must follow the sys.path fix)

if __name__ == "__main__":
    main(sys.argv[1:] or ["bench"])
