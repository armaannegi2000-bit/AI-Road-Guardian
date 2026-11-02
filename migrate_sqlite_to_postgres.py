"""Copy a local Road Guardian SQLite database into an empty Postgres database."""

from __future__ import annotations

import os
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
if not os.getenv("DATABASE_URL"):
    raise SystemExit("DATABASE_URL is required. Set it to the destination Postgres URL.")

# Importing the Flask app creates any missing destination tables. Keep migration
# isolated from AI work so queued reports are not processed during the copy.
os.environ["AI_ANALYSIS_WORKER_MODE"] = "external"
from app import DATABASE, get_db  # noqa: E402


TABLES = (
    "complaints",
    "reverse_geocoding_cache",
    "ai_analysis_jobs",
    "complaint_status_history",
    "repair_verifications",
    "engineer_work_updates",
    "notifications",
    "engineers",
)


def _sqlite_path(source_db):
    path = Path(source_db)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _postgres_columns(connection, table_name):
    return {
        row["column_name"]: row["data_type"]
        for row in connection.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = ?
            ORDER BY ordinal_position
            """,
            (table_name,),
        ).fetchall()
    }


def _convert_value(value, data_type):
    if value is None or not isinstance(value, str):
        return value
    if data_type.startswith("timestamp"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-db",
        default=DATABASE,
        help="SQLite database to copy (default: road_guardian.db)",
    )
    args = parser.parse_args()
    source_path = _sqlite_path(args.source_db)
    if not source_path.is_file():
        raise SystemExit(f"SQLite database not found: {source_path}")

    source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    target = get_db()

    try:
        existing_tables = {
            row[0]
            for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        target_counts = {}
        for table in TABLES:
            count = target.execute(f'SELECT COUNT(*) AS count FROM "{table}"').fetchone()["count"]
            target_counts[table] = count
        if any(target_counts.values()):
            raise RuntimeError(
                "Destination Postgres is not empty. Migration stopped to avoid overwriting data."
            )

        copied_counts = {}
        for table in TABLES:
            if table not in existing_tables:
                copied_counts[table] = 0
                continue

            columns = [
                row["name"]
                for row in source.execute(f'PRAGMA table_info("{table}")').fetchall()
            ]
            target_types = _postgres_columns(target, table)
            missing_columns = [name for name in columns if name not in target_types]
            if missing_columns:
                raise RuntimeError(
                    f"Destination table {table} lacks source columns: {', '.join(missing_columns)}"
                )

            placeholders = ", ".join("%s" for _ in columns)
            column_sql = ", ".join(f'"{name}"' for name in columns)
            insert_sql = f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})'
            rows = source.execute(f'SELECT * FROM "{table}"').fetchall()
            for row in rows:
                values = tuple(
                    _convert_value(row[name], target_types[name])
                    for name in columns
                )
                target.execute(insert_sql, values)

            copied_counts[table] = len(rows)

        # Restore PostgreSQL-generated integer IDs after inserting their old values.
        for table in TABLES:
            target_types = _postgres_columns(target, table)
            if "id" in target_types:
                target.execute(
                    f"""
                    SELECT setval(
                        pg_get_serial_sequence('public.{table}', 'id'),
                        GREATEST(COALESCE(MAX(id), 1), 1),
                        COUNT(*) > 0
                    )
                    FROM "{table}"
                    """
                )

        target.commit()
        print("SQLite to Postgres copy finished. Rows copied by table:")
        for table, count in copied_counts.items():
            print(f"  {table}: {count}")
    except Exception:
        target.rollback()
        raise
    finally:
        source.close()
        target.close()


if __name__ == "__main__":
    main()
