import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "mlflow.db"


def test_sample_database_has_no_paths_from_the_old_project() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        locations = [
            row[0]
            for row in connection.execute(
                "SELECT artifact_location FROM experiments "
                "UNION ALL SELECT artifact_uri FROM runs"
            )
        ]

    assert locations
    assert all("/mnt/d/DTSC/mlflow-analyzer/" not in location for location in locations)
    assert all("/data/artifacts/" in location for location in locations)
