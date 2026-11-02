"""Local photo storage for development and private Cloudflare R2 storage for deployment."""

from __future__ import annotations

import mimetypes
import os
import tempfile
from contextlib import contextmanager
from functools import lru_cache

from flask import Response, abort, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename


PHOTO_STORAGE_MODE = os.getenv("PHOTO_STORAGE_MODE", "local").strip().lower()
if PHOTO_STORAGE_MODE not in {"local", "r2"}:
    raise RuntimeError("PHOTO_STORAGE_MODE must be 'local' or 'r2'.")

if PHOTO_STORAGE_MODE == "r2":
    _missing_r2_values = [
        name
        for name in (
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
        )
        if not os.getenv(name)
    ]
    if _missing_r2_values:
        raise RuntimeError(
            "PHOTO_STORAGE_MODE is r2, but required R2 settings are missing: "
            + ", ".join(_missing_r2_values)
        )


def is_object_storage_enabled():
    return PHOTO_STORAGE_MODE == "r2"


def _safe_photo_name(filename):
    if not filename or "/" in filename or "\\" in filename:
        return None
    safe_name = secure_filename(filename)
    return safe_name if safe_name == filename else None


def _r2_key(filename):
    safe_name = _safe_photo_name(filename)
    if not safe_name:
        raise ValueError("Invalid uploaded-photo filename.")
    return f"uploads/{safe_name}"


@lru_cache(maxsize=1)
def _r2_client():
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError(
            "R2 storage is enabled, but boto3 is not installed. Install project requirements."
        ) from error

    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def save_uploaded_photo(upload, filename, local_folder):
    """Save an uploaded image locally or to the configured R2 bucket."""
    safe_name = _safe_photo_name(filename)
    if not safe_name:
        raise ValueError("Invalid uploaded-photo filename.")

    if is_object_storage_enabled():
        content_type = upload.mimetype or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        _r2_client().upload_fileobj(
            upload.stream,
            os.environ["R2_BUCKET"],
            _r2_key(safe_name),
            ExtraArgs={"ContentType": content_type},
        )
        return f"r2://{_r2_key(safe_name)}"

    os.makedirs(local_folder, exist_ok=True)
    local_path = os.path.join(local_folder, safe_name)
    upload.save(local_path)
    return local_path


def migrate_local_photo_file(local_path, filename):
    """Upload a preserved local image to R2 and confirm the object exists."""
    if not is_object_storage_enabled():
        raise RuntimeError("Set PHOTO_STORAGE_MODE=r2 before migrating photos.")

    safe_name = _safe_photo_name(filename)
    if not safe_name or not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local photo is missing or has an invalid name: {filename}")

    content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    client = _r2_client()
    client.upload_file(
        local_path,
        os.environ["R2_BUCKET"],
        _r2_key(safe_name),
        ExtraArgs={"ContentType": content_type},
    )
    client.head_object(Bucket=os.environ["R2_BUCKET"], Key=_r2_key(safe_name))


def delete_uploaded_photo(filename, local_folder):
    """Remove an orphaned upload after its database transaction fails."""
    safe_name = _safe_photo_name(filename)
    if not safe_name:
        return

    local_path = os.path.join(local_folder, safe_name)
    if os.path.isfile(local_path):
        try:
            os.remove(local_path)
        except OSError as error:
            print("Could not remove local orphaned photo:", error)

    if is_object_storage_enabled():
        try:
            _r2_client().delete_object(
                Bucket=os.environ["R2_BUCKET"],
                Key=_r2_key(safe_name),
            )
        except Exception as error:
            print("Could not remove orphaned R2 photo:", error)


@contextmanager
def local_photo_path(filename, local_folder):
    """Yield a local path for image providers, downloading R2 objects temporarily when needed."""
    safe_name = _safe_photo_name(filename)
    if not safe_name:
        raise ValueError("Invalid uploaded-photo filename.")

    local_path = os.path.join(local_folder, safe_name)
    if os.path.isfile(local_path) or not is_object_storage_enabled():
        yield local_path
        return

    suffix = os.path.splitext(safe_name)[1]
    file_handle = tempfile.NamedTemporaryFile(prefix="road-guardian-", suffix=suffix, delete=False)
    temporary_path = file_handle.name
    file_handle.close()
    try:
        _r2_client().download_file(
            os.environ["R2_BUCKET"],
            _r2_key(safe_name),
            temporary_path,
        )
        yield temporary_path
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass


def serve_uploaded_photo(filename, local_folder):
    """Serve existing local photos or stream private R2 objects through Flask."""
    safe_name = _safe_photo_name(filename)
    if not safe_name:
        abort(404)

    local_path = os.path.join(local_folder, safe_name)
    if os.path.isfile(local_path):
        return send_from_directory(local_folder, safe_name, conditional=True)

    if not is_object_storage_enabled():
        abort(404)

    try:
        stored_object = _r2_client().get_object(
            Bucket=os.environ["R2_BUCKET"],
            Key=_r2_key(safe_name),
        )
    except Exception as error:
        response = getattr(error, "response", {})
        error_code = str(response.get("Error", {}).get("Code", ""))
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            abort(404)
        print("Could not read photo from R2:", error)
        abort(503)

    body = stored_object["Body"]

    def photo_chunks():
        try:
            for chunk in body.iter_chunks(chunk_size=64 * 1024):
                if chunk:
                    yield chunk
        finally:
            body.close()

    response = Response(
        stream_with_context(photo_chunks()),
        mimetype=stored_object.get("ContentType") or mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
        direct_passthrough=True,
    )
    content_length = stored_object.get("ContentLength")
    if content_length is not None:
        response.content_length = content_length
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response
