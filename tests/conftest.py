"""Every test runs against a throwaway library, never the user's `.rca/`.

`rca.api.main` builds a default app when it is imported, and `Settings()` falls back to
`./.rca`, so without this any test that imported the API module opened the real library.
Set here, before any test module is imported; environment variables win over `.env`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_LIBRARY = Path(tempfile.mkdtemp(prefix="rca-test-library-"))

for _name, _path in {
    "RCA_DATA_DIR": _LIBRARY,
    "RCA_GRAPH_DB_PATH": _LIBRARY / "graph.sqlite3",
    "RCA_VECTOR_DIR": _LIBRARY / "vectors",
    "RCA_TELEMETRY_LOG_PATH": _LIBRARY / "telemetry.jsonl",
    "RCA_EXPERIMENT_DB_PATH": _LIBRARY / "experiments.sqlite3",
}.items():
    os.environ[_name] = str(_path)


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    shutil.rmtree(_LIBRARY, ignore_errors=True)
