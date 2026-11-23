"""
Postgres (Supabase) and R2 (S3 API) helpers.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import boto3
import psycopg
from botocore.config import Config
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


# ---------- Postgres ----------

def connect():
    return psycopg.connect(os.environ["DB_URL"], autocommit=True,
                           prepare_threshold=None, row_factory=dict_row)


def query(sql, params=None):
    """Run a statement and return all rows as dicts (empty list if none)."""
    with connect() as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall() if cur.description else []


def query_one(sql, params=None):
    rows = query(sql, params)
    return rows[0] if rows else None


def jsonb(value):
    return Jsonb(value)


def current_model():
    """Latest promoted model version: {'model_version', 'model_key', 'challenger_metrics'}."""
    return query_one("""select model_version, model_key, challenger_metrics, finished_at
                        from training_runs where promoted
                        order by model_version desc limit 1""")


# ---------- R2 ----------

@lru_cache
def _r2():
    url = urlparse(os.environ["S3_URL"])
    client = boto3.client(
        "s3",
        endpoint_url=f"{url.scheme}://{url.netloc}",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", max_pool_connections=32),
    )
    return client, url.path.strip("/")


def put_file(path, key, content_type=None):
    client, bucket = _r2()
    extra = {"ContentType": content_type} if content_type else None
    client.upload_file(str(path), bucket, key, ExtraArgs=extra)


def download_file(key, path):
    client, bucket = _r2()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(path))


def download_many(pairs, workers=32):
    """Download [(key, local_path), ...] in parallel. Returns the list of keys that failed."""
    def one(pair):
        try:
            download_file(*pair)
            return None
        except Exception:
            return pair[0]
    with ThreadPoolExecutor(workers) as ex:
        return [k for k in ex.map(one, pairs) if k]


def head(key):
    """Object metadata (size in 'ContentLength'), or None if it doesn't exist."""
    client, bucket = _r2()
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError:
        return None


def delete(key):
    client, bucket = _r2()
    client.delete_object(Bucket=bucket, Key=key)


def list_keys(prefix):
    client, bucket = _r2()
    keys = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.update(obj["Key"] for obj in page.get("Contents", []))
    return keys


def presign_put(key, content_type, expires=600):
    """URL the browser can PUT one file to. The Content-Type header must match."""
    client, bucket = _r2()
    return client.generate_presigned_url(
        "put_object", Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expires)
