"""One-off: upload the full dataset to R2 as dataset/<split>/<class>/<file>.

Reads models/splits.csv and the local images in data/<split>/<class>/<file>.
Already-uploaded files are skipped, so it is safe to re-run.

    uv run --env-file .env python -m scripts.upload_dataset
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src import database as db

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "data"

splits = pd.read_csv(ROOT / "models" / "splits.csv")
splits["key"] = "dataset/" + splits.split + "/" + splits.file
splits["path"] = [LOCAL / s / f for s, f in zip(splits.split, splits.file)]

missing = [p for p in splits.path if not p.exists()]
if missing:
    raise SystemExit(f"{len(missing)} files listed in splits.csv are missing locally, e.g. {missing[0]}")

existing = db.list_keys("dataset/")
todo = splits[~splits.key.isin(existing)]
print(f"{len(splits)} files, {len(existing)} already in R2, uploading {len(todo)}")


def upload(row):
    db.put_file(row[0], row[1], "image/jpeg")


with ThreadPoolExecutor(32) as ex:
    for i, _ in enumerate(ex.map(upload, zip(todo.path, todo.key)), 1):
        if i % 1000 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")
print("Done")
