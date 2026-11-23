"""
FastAPI app: prediction, presigned uploads, retraining, metrics.
"""
import json
import logging
import os
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import psycopg
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import database as db
from src import model as retraining
from src import prediction
from src.preprocessing import MAX_BYTES, MODELS_DIR, check_upload_paths, load_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

STARTED = time.time()
DATASET_STATS = json.loads((MODELS_DIR / "dataset_stats.json").read_text())
NOTEBOOK_METRICS = json.loads((MODELS_DIR / "metrics.json").read_text())

@asynccontextmanager
async def lifespan(app):
    prediction.get_model()          # load the model before serving
    yield


app = FastAPI(title="Tomato leaf disease API", lifespan=lifespan)
origins = [o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "instance": socket.gethostname(),   # which container answered
            "uptime_s": round(time.time() - STARTED),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(STARTED)),
            "model_version": prediction._state["version"]}


# ---------- prediction ----------

def log_prediction(result):
    try:
        db.query("insert into predictions (label, confidence, latency_ms, model_version) values (%s, %s, %s, %s)",
                 (result["label"], result["confidence"], result["latency_ms"], result["model_version"]))
    except Exception as e:
        log.warning("could not log prediction: %s", e)


@app.post("/predict")
def predict(background: BackgroundTasks, file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    try:
        image = load_image(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    result = prediction.predict_image(image)
    background.add_task(log_prediction, result)
    return result


# ---------- uploads (browser -> R2 via presigned URLs) ----------

class FileIn(BaseModel):
    relative_path: str
    content_type: str


class PresignIn(BaseModel):
    files: list[FileIn]


class CompleteIn(BaseModel):
    ids: list[uuid.UUID]


@app.post("/uploads/presign")
def presign(body: PresignIn):
    """Check the folder structure and return one PUT URL per image. All-or-nothing."""
    accepted, errors = check_upload_paths([f.model_dump() for f in body.files])
    if errors:
        raise HTTPException(422, {"errors": errors})
    batch_id = uuid.uuid4()
    rows, out = [], []
    for f in accepted:
        uid = uuid.uuid4()
        key = f"uploads/{batch_id}/{uid}{f['ext']}"
        rows.append((uid, batch_id, f["relative_path"], f["label"], key, f["content_type"]))
        out.append({"id": str(uid), "relative_path": f["relative_path"], "label": f["label"],
                    "content_type": f["content_type"], "url": db.presign_put(key, f["content_type"])})
    with db.connect() as conn:
        conn.cursor().executemany("""insert into uploads (id, batch_id, relative_path, label, r2_key, content_type)
                                     values (%s, %s, %s, %s, %s, %s)""", rows)
    return {"batch_id": str(batch_id), "files": out}


@app.post("/uploads/complete")
def complete(body: CompleteIn):
    """Confirm uploaded files: check they exist in R2 and are not too large."""
    rows = db.query("select id, r2_key from uploads where id = any(%s) and status = 'requested'",
                    ([str(i) for i in body.ids],))
    def check(r):
        meta = db.head(r["r2_key"])
        if meta is None:
            return "rejected", "not uploaded", None, r["id"]
        if meta["ContentLength"] > MAX_BYTES:
            db.delete(r["r2_key"])
            return "rejected", "file too large", meta["ContentLength"], r["id"]
        return "pending", None, meta["ContentLength"], r["id"]

    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(check, rows))
    with db.connect() as conn:
        conn.cursor().executemany("update uploads set status = %s, reason = %s, size_bytes = %s where id = %s",
                                  results)
    statuses = [r[0] for r in results]
    return {"pending": statuses.count("pending"), "rejected": statuses.count("rejected")}


@app.get("/uploads/summary")
def uploads_summary():
    by_status = db.query("select status, count(*)::int as n from uploads group by status")
    pending = db.query("""select label, count(*)::int as n from uploads where status = 'pending'
                          group by label order by label""")
    return {"by_status": {r["status"]: r["n"] for r in by_status},
            "pending_by_label": {r["label"]: r["n"] for r in pending}}


# ---------- retraining ----------

@app.post("/retrain", status_code=202)
def start_retrain(background: BackgroundTasks):
    # a run stuck in 'running' (e.g. container restarted) for over 2 hours is marked failed
    db.query("""update training_runs set status = 'failed', error = 'timed out', finished_at = now()
                where status = 'running' and started_at < now() - interval '2 hours'""")
    if not db.query_one("select 1 from uploads where status = 'pending' limit 1"):
        raise HTTPException(400, "no pending uploads to retrain on")
    try:
        run = db.query_one("insert into training_runs (status) values ('running') returning id")
    except psycopg.errors.UniqueViolation:
        raise HTTPException(409, "a retraining run is already in progress")
    background.add_task(retraining.retrain, run["id"])
    return {"run_id": run["id"], "status": "running"}


RUN_COLUMNS = """id, status, n_new, n_rejected, n_replay, champion_metrics, challenger_metrics,
                 promoted, model_version, error, started_at, finished_at"""


@app.get("/retrain")
def list_runs():
    return db.query(f"select {RUN_COLUMNS} from training_runs order by id desc limit 50")


@app.get("/retrain/{run_id}")
def get_run(run_id: int):
    run = db.query_one(f"select {RUN_COLUMNS} from training_runs where id = %s", (run_id,))
    if not run:
        raise HTTPException(404, "run not found")
    return run


# ---------- dashboard data ----------

@app.get("/metrics")
def metrics():
    current = db.current_model()
    counts = db.query_one("""select count(*)::int as total,
                                    count(*) filter (where created_at > now() - interval '24 hours')::int as last_24h,
                                    avg(latency_ms)::float as avg_latency_ms
                             from predictions""")
    by_label = db.query("select label, count(*)::int as n from predictions group by label order by n desc")
    return {"model_version": current["model_version"],
            "trained_at": current["finished_at"],
            "test_metrics": current["challenger_metrics"],
            "notebook": {"hyperparameters": NOTEBOOK_METRICS["hyperparameters"],
                         "history": NOTEBOOK_METRICS["history"]},
            "predictions": {**counts, "by_label": {r["label"]: r["n"] for r in by_label}}}


@app.get("/stats")
def stats():
    return DATASET_STATS
