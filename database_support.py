"""Small adapter that lets the existing app use SQLite locally or Postgres on Render."""

from __future__ import annotations

import os
import re
import sqlite3


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

IS_POSTGRES = bool(DATABASE_URL)

if IS_POSTGRES and not DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg://")):
    raise RuntimeError("DATABASE_URL must be a PostgreSQL connection URL.")

if IS_POSTGRES:
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError(
            "DATABASE_URL is set, but psycopg is not installed. Install project requirements."
        ) from error

    DBIntegrityError = psycopg.IntegrityError
else:
    psycopg = None
    DBIntegrityError = sqlite3.IntegrityError


class _HybridRow(dict):
    """Support both SQLite-style name and numeric row access."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(self.values())[key]
        return super().__getitem__(key)


def _hybrid_row(cursor):
    names = [column.name for column in cursor.description]

    def make_row(values):
        return _HybridRow(zip(names, values))

    return make_row


class _EmptyCursor:
    rowcount = 0

    @staticmethod
    def fetchone():
        return None

    @staticmethod
    def fetchall():
        return []


def _postgres_sql(statement):
    """Translate the small set of SQLite-only statements used by app.py."""
    match = re.fullmatch(r"\s*PRAGMA\s+table_info\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;?\s*", statement, re.IGNORECASE)
    if match:
        return (
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "ORDER BY ordinal_position",
            (match.group(1),),
        )

    if re.match(r"\s*PRAGMA\b", statement, re.IGNORECASE):
        return None, ()

    statement = re.sub(
        r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b",
        "SERIAL PRIMARY KEY",
        statement,
        flags=re.IGNORECASE,
    )
    statement = re.sub(r"\bBEGIN\s+IMMEDIATE\b", "BEGIN", statement, flags=re.IGNORECASE)
    statement = re.sub(
        r"\bALTER\s+TABLE(\s+[A-Za-z_][A-Za-z0-9_]*\s+)ADD\s+COLUMN\b(?!\s+IF\s+NOT\s+EXISTS)",
        r"ALTER TABLE\1ADD COLUMN IF NOT EXISTS",
        statement,
        flags=re.IGNORECASE,
    )

    def replace_datetime(match):
        interval = match.group(1)
        return f"CURRENT_TIMESTAMP + INTERVAL '{interval}'"

    statement = re.sub(
        r"datetime\s*\(\s*'now'\s*,\s*'([+-]\d+\s+(?:seconds?|minutes?|hours?|days?))'\s*\)",
        replace_datetime,
        statement,
        flags=re.IGNORECASE,
    )
    statement = statement.replace("?", "%s")
    return statement, None


class PostgresConnection:
    """Expose the subset of sqlite3.Connection used by this Flask app."""

    def __init__(self, connection):
        self._connection = connection

    def execute(self, statement, parameters=()):
        statement, replacement_parameters = _postgres_sql(statement)
        if statement is None:
            return _EmptyCursor()
        if replacement_parameters is not None:
            parameters = replacement_parameters
        return self._connection.execute(statement, parameters)

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


def get_db(database_path="road_guardian.db"):
    """Open the configured database; default to the existing local SQLite file."""
    if IS_POSTGRES:
        return PostgresConnection(
            psycopg.connect(
                DATABASE_URL,
                row_factory=_hybrid_row,
                connect_timeout=15,
            )
        )

    connection = sqlite3.connect(database_path, timeout=30)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.row_factory = sqlite3.Row
    return connection
