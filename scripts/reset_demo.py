"""Reset the app to the original model (v1): deletes all retraining runs after v1, all uploads,
all logged predictions, and their files in R2 (uploads/ and models/). The dataset in R2 is kept.

    uv run --env-file .env python -m scripts.reset_demo
"""
import shutil

from src import database as db
from src.prediction import CACHE_DIR

keys = db.list_keys("uploads/") | db.list_keys("models/")
for key in keys:
    db.delete(key)

db.query("delete from uploads")
db.query("delete from predictions")
db.query("delete from training_runs where model_version is distinct from 1")
db.query("select setval('training_runs_id_seq', (select max(id) from training_runs))")
shutil.rmtree(CACHE_DIR, ignore_errors=True)

print(f"Deleted {len(keys)} files from R2. Runs left:",
      db.query("select id, model_version from training_runs"))
