from pathlib import Path

from rca.config.settings import Settings

REPO = Path(__file__).resolve().parents[2]


def test_tests_never_use_the_real_library() -> None:
    """Importing rca.api.main builds a default app from Settings(); under pytest that must not
    point at the user's ./.rca (tests/conftest.py redirects it)."""
    settings = Settings()
    real = (REPO / ".rca").resolve()
    for path in (
        settings.data_dir,
        settings.graph_db_path,
        settings.vector_dir,
        settings.telemetry_log_path,
        settings.experiment_db_path,
    ):
        assert not path.resolve().is_relative_to(real), path
