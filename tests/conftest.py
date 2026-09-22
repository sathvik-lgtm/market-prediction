import sqlite3
from pathlib import Path

import pytest

from market_pred.db.schema import init_db


@pytest.fixture
def db_conn(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()
