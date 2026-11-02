"""Upload locally referenced Road Guardian photos to the configured R2 bucket."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
from photo_storage import PHOTO_STORAGE_MODE, migrate_local_photo_file  # noqa: E402


PHOTO_TABLES = ("complaints", "engineer_work_updates", "repair_verifications")


def main():
    if PHOTO_STORAGE_MODE != "r2":
        raise SystemExit("Set PHOTO_STORAGE_MODE=r2 and configure all R2 credentials first.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="static/uploads",
        help="Folder containing the local uploaded images (default: static/uploads)",
    )
    parser.add_argument(
        "--database",
        default="road_guardian.db",
        help="Local SQLite database used to find referenced photos",
    )
    args = parser.parse_args()
    source_dir = Path(args.source_dir).resolve()

    database_path = Path(args.database)
    if not database_path.is_absolute():
        database_path = Path.cwd() / database_path
    if not database_path.is_file():
        raise SystemExit(f"SQLite database not found: {database_path}")

    database = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        photo_names = set()
        for table in PHOTO_TABLES:
            if table not in tables:
                continue
            columns = {
                row[1]
                for row in database.execute(f'PRAGMA table_info("{table}")').fetchall()
            }
            if "photo" not in columns:
                continue
            photo_names.update(
                row[0]
                for row in database.execute(
                    f'SELECT DISTINCT photo FROM "{table}" WHERE photo IS NOT NULL AND photo <> \'\''
                ).fetchall()
            )
    finally:
        database.close()

    if not photo_names:
        print("No referenced photos were found in the local SQLite database.")
        return

    missing_files = sorted(
        name for name in photo_names if not (source_dir / name).is_file()
    )
    if missing_files:
        raise SystemExit(
            f"Stopped before upload: {len(missing_files)} referenced photo(s) are missing from {source_dir}."
        )

    total_bytes = 0
    for filename in sorted(photo_names):
        path = source_dir / filename
        migrate_local_photo_file(str(path), filename)
        total_bytes += path.stat().st_size

    print(
        f"Uploaded and verified {len(photo_names)} referenced photo(s) "
        f"({total_bytes} bytes). Local originals were kept."
    )


if __name__ == "__main__":
    main()
