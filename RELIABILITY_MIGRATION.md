# AI Road Guardian storage and worker migration

## Current behavior

- With `DATABASE_URL` unset, the app continues to use `road_guardian.db` (SQLite).
- With `PHOTO_STORAGE_MODE=local`, uploads continue to use `static/uploads`.
- The existing web pages and report submission flow are unchanged by default.
- `worker.py` can run the existing persistent AI job queue as a separate process.

## Before a live cutover

1. Back up the live Render database and its photos. A copy of the local files on a development computer is not a backup of data stored inside Render.
2. Create an empty Postgres database and configure `DATABASE_URL` with its connection URL.
3. Create an R2 bucket and its S3-compatible access key and secret. These are separate from a general Cloudflare API token.
4. Set `PHOTO_STORAGE_MODE=r2`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_BUCKET` in the environment used by the migration scripts and both Render services.
5. Copy the source SQLite database and its matching `static/uploads` folder to the machine where the migration scripts will run. Keep this source backup unchanged until cutover is verified.
6. Run `python migrate_local_photos_to_r2.py --source-dir "<backup>\static\uploads" --database "<backup>\road_guardian.db"` to upload and verify referenced photos. The script does not delete local originals.
7. Run `python migrate_sqlite_to_postgres.py --source-db "<backup>\road_guardian.db"` to copy the SQLite rows. It creates missing tables, requires the destination tables to be empty, and stops if a source column is missing from Postgres.
8. Before switching traffic, verify report counts, photo links, report submission, status updates, and AI job completion against the new services.

## Render service settings after data is ready

- Web service: set `DATABASE_URL`, `PHOTO_STORAGE_MODE=r2`, the R2 values above, and `AI_ANALYSIS_WORKER_MODE=external`. Keep the existing Gunicorn start command.
- Background worker: create a Render Background Worker for this same repository. Set the same `DATABASE_URL`, `PHOTO_STORAGE_MODE=r2`, R2 values, `GEMINI_API_KEY`, and `GROQ_API_KEY` on it. Use start command `python worker.py`.
- The web service and worker must use the same Postgres database and R2 bucket. Render services do not share their local files, so local photo mode cannot pass a submitted image from the web service to the worker.
- The worker reads queued reports from Postgres, downloads the image from R2 for analysis, and saves the result and retry state back to Postgres.
- Transient provider/network failures, rate limits, server errors, and malformed AI output are retried up to three job attempts. Missing provider keys and permanent provider errors stop promptly and save a visible unavailable result, so the page does not wait forever.

Do not point the live site at a new empty Postgres database before its current Render data has been copied. The migration scripts read local SQLite files; they do not connect to or export a Render service's local SQLite file.
