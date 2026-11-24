# Tomato Leaf Doctor: an end-to-end ML pipeline

Tomato is one of the most widely grown vegetables in Rwanda and across Africa, and leaf diseases such as
blights, leaf mold and viral infections can destroy a large part of a harvest when they are caught late.
This project lets a farmer or extension officer **photograph a tomato leaf and get an instant
diagnosis** (9 diseases or healthy), and lets new labelled photos be **uploaded to retrain the model**.

It covers the full ML cycle: data acquisition, preprocessing, model creation and evaluation (notebook),
an API, a web UI, retraining with new data, Docker, load testing and cloud deployment.

| | |
|---|---|
| **Video demo** | https://youtu.be/yzbg0IQp2SA |
| **Live app** | http://89.116.24.136 |
| **API docs (Swagger)** | http://89.116.24.136/api/docs |
| **Dataset** | [PlantVillage (Mendeley Data)](https://data.mendeley.com/datasets/tywbtsjrjv/1) |

## Features

- **Predict**: upload one leaf photo and get the disease, the confidence and the top 3 classes.
- **Dashboard**: API status and uptime, current model version and its test metrics, prediction counts.
- **Insights**: dataset visualisations with interpretations (class balance, lesion coverage,
  brightness and contrast, texture).
- **Upload & retrain**: upload a folder of labelled images (files go straight from the browser to
  cloud storage), then press a button to retrain. The retrained model replaces the current one only if
  it performs at least as well.
- **Training history**: every retraining run, with old vs new metrics and whether it was promoted.

## Architecture

```
                       ┌──────────────── VPS (docker compose) ────────────────┐
 Browser ── :80 ──► nginx ──/api/*──► API containers (FastAPI + TensorFlow) ×1–4 ──► Postgres (Supabase)
                    │      round robin            │                                  uploads, runs,
                    │                             │                                  predictions
                    └── / ──► UI (Streamlit) ─────┘
                                                                                ──► Cloudflare R2
 Browser ── presigned PUT (uploaded images go straight to R2, not through the API) ─►  dataset/, uploads/,
                                                                                      models/v<N>.h5
```

- **API** (`src/api.py`): prediction, presigned uploads, retraining in the background, metrics.
- **UI** (`ui/app.py`): talks only to the API; it holds no database or storage credentials.
- **Postgres**: `training_runs` (one row per run; a promoted run is a model version), `uploads`
  (one row per uploaded image), `predictions` (monitoring).
- **R2**: the full dataset, uploaded images and every promoted model version.
- Every API container checks the current model version every 30 s, so a model promoted by one container
  is picked up by all of them.

## Repository structure

```
├── README.md
├── notebook/crop_disease_pipeline.ipynb   # EDA, preprocessing, training, evaluation, retraining
├── src/
│   ├── preprocessing.py   # image validation/resizing, upload folder rules
│   ├── model.py           # model building, evaluation, retraining (champion vs challenger)
│   ├── prediction.py      # model loading (current version) and single-image prediction
│   ├── database.py        # Postgres and R2 helpers
│   └── api.py             # FastAPI app
├── ui/                    # Streamlit app + folder upload widget
├── data/                  # train / validation / test images (Git LFS)
├── models/                # crop_disease_mobilenetv2.h5 (v1), class names, metrics, splits.csv
├── migrations/            # SQL to create the database tables
├── scripts/               # one-off dataset upload to R2, demo reset
├── locustfile.py, loadtest/   # load test and results
├── Dockerfile, docker-compose.yml, nginx/
└── pyproject.toml, uv.lock
```

## Data

**Source.** PlantVillage dataset, *without augmentation* version
([Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)). Only the 10 tomato classes (folders
starting with `Tomato___`) are used.

**Raw archive (`data/raw/Tomato___.zip`, not in git).** The 10 `Tomato___*` class folders from the
PlantVillage download, zipped together: 18,160 images, one folder per class. It was also published as the
Kaggle dataset `etiennekagaba/plantvillage-tomato`, which is what the notebook reads. To recreate it:

1. Download the PlantVillage dataset (without augmentation) from the Mendeley link above and unzip it.
2. Zip the 10 `Tomato___*` folders: `zip -r data/raw/Tomato___.zip Tomato___*`

**Splits (`data/train`, `data/validation`, `data/test`, stored with Git LFS).** Created by the notebook
(sections 1–2): corrupt files and exact duplicates (MD5) are removed, leaving 18,145 images, which are
split 70/15/15, stratified by class (seed 42). The assignment of every file is recorded in
`models/splits.csv`.

| Split | Images |
|---|---|
| train | 12,701 |
| validation | 2,722 |
| test | 2,722 |

**Cloud copy.** The same splits are stored in Cloudflare R2 under `dataset/<split>/<class>/<file>`
(uploaded with `uv run --env-file .env python -m scripts.upload_dataset`). Retraining samples its
replay, validation and test images from there using `models/splits.csv`.

## Model

Built and evaluated in [`notebook/crop_disease_pipeline.ipynb`](notebook/crop_disease_pipeline.ipynb)
(run on Kaggle, GPU T4).

- **Preprocessing:** decode as RGB, resize to 224×224, normalise to [-1, 1] inside the model, augment
  the training set only (flips, rotation, zoom, translation, brightness, contrast; no hue shifts,
  because colour carries disease information), class weights for the imbalance.
- **Baseline:** a small CNN from scratch, about 43% validation accuracy.
- **Optimised model:** MobileNetV2 pretrained on ImageNet plus a new head, with Adam, L2
  regularisation, dropout, batch normalisation, label smoothing, early stopping, learning-rate
  reduction on plateau, a hyperparameter search (learning rate, dropout) and fine-tuning of the top 30
  backbone layers.

**Test set results (2,722 images, v1):**

| Accuracy | Loss | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | Top-3 accuracy | ROC-AUC (macro, OvR) |
|---|---|---|---|---|---|---|---|
| 0.909 | 0.367 | 0.884 | 0.894 | 0.886 | 0.909 | 0.989 | 0.994 |

Macro F1 is the main metric, because the classes are imbalanced and every disease matters equally.

## Retraining

1. **Upload.** The UI asks the API for one presigned URL per image and the browser uploads each file
   straight to R2. Each image is saved as a row in the `uploads` table. The folder must be laid out as
   `<folder>/<class>/<image>` (`.jpg`, `.jpeg`, `.png`, at most 200 per batch); the label is the
   folder name. The browser and the API both check this, and a batch with any invalid path is
   rejected as a whole.
2. **Trigger.** The "Retrain model" button calls `POST /retrain`. Only one run can be active at a time.
3. **Preprocessing.** Pending images are downloaded, decoded and validated, de-duplicated (MD5) and
   resized to 224×224. Invalid or duplicate images are marked `rejected` with a reason.
4. **Training.** The current model is used as the pretrained starting point (*challenger*) and
   fine-tuned for 3 epochs (Adam, learning rate 1e-5, augmentation, early stopping) on the new images
   plus 50 original training images per class (to avoid forgetting old classes), with 20 validation
   images per class.
5. **Evaluation and promotion.** The current model (*champion*) and the challenger are evaluated on the
   same fixed sample of 30 test images per class. The challenger is promoted only if its macro F1 is at
   least as high. A promoted model is saved to R2 as `models/v<N>.h5` and becomes the new version.

A run takes about 3 minutes on CPU and needs about 1.5 GB of RAM.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, uptime, model version, which container answered |
| POST | `/predict` | one image (multipart `file`) → label, confidence, top 3 |
| POST | `/uploads/presign` | `{files: [{relative_path, content_type}]}` → one upload URL per image |
| POST | `/uploads/complete` | `{ids: [...]}` → checks the files arrived, marks them pending |
| GET | `/uploads/summary` | upload counts by status and pending images by class |
| POST | `/retrain` | start a retraining run in the background |
| GET | `/retrain`, `/retrain/{id}` | list runs / status of one run |
| GET | `/metrics` | current model, its test metrics, training history, prediction counts |
| GET | `/stats` | dataset statistics for the Insights page |

Interactive docs: `/docs` (or `/api/docs` behind nginx).

## Setup

### Requirements

- [Docker](https://docs.docker.com/get-docker/) with Compose, [uv](https://docs.astral.sh/uv/) and
  [Git LFS](https://git-lfs.com/)
- A Postgres database (this project uses Supabase) and a Cloudflare R2 bucket (or any S3-compatible storage)

### 1. Clone and configure

```bash
git clone https://github.com/kagaba-etienne/ml-pipeline-summative.git
cd ml-pipeline-summative
git lfs pull                 # the images in data/
cp .env.example .env         # then fill in the values
uv sync                      # local Python environment (also needed for Locust and the scripts)
```

`.env` holds the database URLs and R2 keys. It is never committed or copied into the Docker image;
docker compose passes the values to the API container at runtime.

### 2. Database

```bash
psql "$(grep ^DIRECT_DB_URL .env | cut -d= -f2-)" -f migrations/001_init.sql
psql "$(grep ^DIRECT_DB_URL .env | cut -d= -f2-)" -f migrations/002_seed_v1.sql
```

### 3. Storage

Upload the dataset to R2 (once, skips files already there):

```bash
uv run --env-file .env python -m scripts.upload_dataset
```

Add a CORS rule to the bucket so browsers can upload directly (R2 → bucket → Settings → CORS policy).
List every address the app is opened from:

```json
[
  {
    "AllowedOrigins": ["http://<vps-ip>", "http://localhost:8501", "http://localhost:8080"],
    "AllowedMethods": ["PUT"],
    "AllowedHeaders": ["Content-Type"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

### 4. Run

```bash
docker compose up -d --build              # app on http://localhost (HTTP_PORT=8080 to use another port)
docker compose up -d --scale api=4        # 4 API containers behind nginx
docker compose logs -f api
docker compose down
```

Without Docker (two terminals; set `CORS_ORIGINS=http://localhost:8501` in `.env`):

```bash
uv run --env-file .env uvicorn src.api:app --port 8000
API_URL=http://localhost:8000 uv run streamlit run ui/app.py
```

To reset to the original model before a demo (deletes runs, uploads and predictions):
`uv run --env-file .env python -m scripts.reset_demo`

### 5. Deploy on a VPS

The live app runs on an Ubuntu 26.04 VPS with 4 vCPU and 8 GB RAM (each API container uses about
0.5 GB, and a retraining run about 1.5 GB more). Log in as a sudo user (not root), then:

```bash
sudo ufw allow OpenSSH && sudo ufw allow 80/tcp                 # firewall: SSH and HTTP only
curl -fsSL https://get.docker.com | sudo sh                     # install Docker
sudo usermod -aG docker $USER                                   # log out and back in after this

# the images in data/ are not needed on the server, so skip the LFS download
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kagaba-etienne/ml-pipeline-summative.git
cd ml-pipeline-summative
# from your machine:  scp .env <user>@<vps-ip>:ml-pipeline-summative/   then on the server: chmod 600 .env
docker compose up -d --build
```

Add `http://<vps-ip>` to the R2 CORS rule. Only nginx publishes a port (80); the API and UI containers
are reachable only inside Docker's network. To update: `git pull && docker compose up -d --build`.

## Load testing (flood request simulation)

[Locust](https://locust.io/) simulates farmers sending leaf photos. Each simulated user sends a real
test image to `/predict` (and occasionally `/health`), waiting 0.5–1.5 s between requests. The same
test (30 users, 2 minutes) is run against 1, 2 and 4 API containers behind nginx, with each container
limited to 1 CPU core. [loadtest/run.sh](loadtest/run.sh) scales the containers, waits until they are
healthy, and runs Locust for each size:

```bash
HTTP_PORT=8080 loadtest/run.sh                    # local Docker  -> loadtest/api<N>_stats.csv, api<N>.html
SSH=<user>@<vps-ip> HOST=http://<vps-ip> PREFIX=vps_api loadtest/run.sh
                                                  # deployed VPS  -> loadtest/vps_api<N>_*
```

For the VPS test, Locust runs on a laptop and sends requests over the internet to the live app, and
the script scales the containers on the server over SSH. The full Locust reports (HTML, with charts)
and CSVs are in [loadtest/](loadtest/).

### Results for `/predict`

**Local Docker** (12-core laptop, Locust on the same machine):

| API containers | Requests | Failures | Throughput (req/s) | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 463 | 0 | 3.9 | 6,181 | 6,400 | 7,500 | 8,100 | 8,533 |
| 2 | 853 | 0 | 7.2 | 2,888 | 2,700 | 4,900 | 5,100 | 5,280 |
| 4 | 1,253 | 0 | 10.5 | 1,618 | 1,600 | 2,800 | 3,200 | 3,620 |

**Deployed VPS** (4 shared vCPU, 8 GB RAM; Locust on a laptop over the internet):

| API containers | Requests | Failures | Throughput (req/s) | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 168 | 0 | 1.4 | 17,528 | 17,000 | 29,000 | 29,000 | 29,881 |
| 2 | 269 | 0 | 2.3 | 10,658 | 9,900 | 18,000 | 20,000 | 20,768 |
| 4 | 846 | 0 | 7.1 | 2,897 | 2,700 | 6,300 | 8,200 | 11,031 |

**Scaling from 1 to 4 containers:**

| | Throughput | Average response time | p95 |
|---|---|---|---|
| Local | 3.9 → 10.5 req/s (**2.7×**) | 6.2 s → 1.6 s (**3.8× faster**) | 7.5 s → 2.8 s |
| VPS | 1.4 → 7.1 req/s (**5.0×**) | 17.5 s → 2.9 s (**6.0× faster**) | 29 s → 6.3 s |

### What the results show

- **More containers, more throughput.** In both environments a single container is the bottleneck:
  it handles one prediction at a time, so with 30 users sending requests, most of the response time
  is spent waiting in the queue. Each added container takes a share of the queue (nginx spreads
  requests round robin), so throughput rises and response times fall. With 4 containers the VPS
  serves 5× more requests than with 1, and the average wait drops from 17.5 s to 2.9 s.
- **No request failed** in any of the six runs (3,852 requests to `/predict` in total). Under
  overload the system slows down instead of dropping requests.
- **The VPS is slower per container than the laptop.** One prediction takes about 0.25 s on a laptop
  core, but 0.33–2 s on a VPS core (measured one request at a time, with nothing else running). The
  VPS's vCPUs are shared with other customers' machines, so their speed is lower and varies from
  moment to moment. That is why 1 VPS container manages only 1.4 req/s, and why VPS throughput
  fluctuates within a run (see the charts in the HTML reports).
- **The network adds a fixed cost.** From the test laptop, a round trip to the VPS takes about 0.3 s
  (the median `/health` response, against 6 ms locally), and uploading a 14 KB image takes another
  0.2–0.4 s. This adds roughly 0.5–0.8 s to every VPS request, whatever the number of containers.
- **The VPS gained more from scaling (5× vs 2.7×)**, for two reasons. First, each simulated user
  waits for its answer before sending the next photo, so 30 users can send at most about
  30 ÷ (response time + 1 s pause) requests per second. With 4 containers on the laptop, that limit
  was about 11.5 req/s and the test reached 10.5, so the laptop had little room left to grow. The
  VPS started from a much slower single container, so it had more room. Second, the shared vCPUs'
  speed varies, so the 1-container VPS run may have hit a slower period than the 4-container run.
  Part of the 5× is therefore noise between runs, but both tests show the same trend.
- **Capacity.** 4 containers on the VPS still don't keep up with 30 busy users (median 2.7 s against
  about 1 s for an unloaded request). 4 containers is the most this server can use, one per vCPU;
  serving more users would need a larger server or more servers behind the load balancer.

Each container runs one prediction at a time. An earlier local test let a container run several at
once on its single core. Docker then throttled the CPU in bursts, and about 10% of requests took
around 18 s even with 4 containers. Queuing predictions one at a time removed those outliers.

## Tech stack

TensorFlow/Keras (MobileNetV2), scikit-learn, FastAPI, Streamlit, Plotly, PostgreSQL (Supabase),
Cloudflare R2 (boto3), Docker, nginx, Locust, uv.
