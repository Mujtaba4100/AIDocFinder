from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  folder TEXT NOT NULL,
  path TEXT NOT NULL UNIQUE,
  rel_path TEXT NOT NULL,
  ext TEXT NOT NULL,
  mtime INTEGER NOT NULL,
  size INTEGER NOT NULL,
  sha256 TEXT,
  extracted_text TEXT,
  embedding BLOB,
  embedding_dim INTEGER,
  image_embedding BLOB,
  image_embedding_dim INTEGER,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_folder ON files(folder);
"""


def open_db(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA_SQL)
    _ensure_columns(con)
    return con


def _ensure_columns(con: sqlite3.Connection) -> None:
  """Best-effort schema migration for existing DBs."""
  try:
    cols = {row[1] for row in con.execute("PRAGMA table_info(files)").fetchall()}
    if "image_embedding" not in cols:
      con.execute("ALTER TABLE files ADD COLUMN image_embedding BLOB")
    if "image_embedding_dim" not in cols:
      con.execute("ALTER TABLE files ADD COLUMN image_embedding_dim INTEGER")
    con.commit()
  except Exception:
    # Non-fatal: app can still run; image search will just be unavailable.
    pass


def now_ts() -> int:
    return int(time.time())
