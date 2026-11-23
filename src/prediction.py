"""
Load the current model and predict a single image.

The current version is the latest promoted row in training_runs. Every API container
checks it every 30 s, so a model promoted by one container is picked up by all of them.
"""
import json
import logging
import threading
import time

import numpy as np
from tensorflow import keras

from src import database as db
from src.preprocessing import CLASS_NAMES, MODELS_DIR, to_array

log = logging.getLogger(__name__)

BUNDLED_MODEL = MODELS_DIR / "crop_disease_mobilenetv2.h5"     # version 1
CACHE_DIR = MODELS_DIR / "cache"
CHECK_EVERY = 30  # seconds
SHORT_NAMES = json.loads((MODELS_DIR / "dataset_stats.json").read_text())["short_names"]

_lock = threading.Lock()
# One prediction at a time per container: concurrent CPU-bound calls on a small CPU quota
# get throttled in bursts and produce very slow outliers. Scale with more containers instead.
_infer_lock = threading.Lock()
_state = {"model": None, "version": None, "checked": 0.0}


def load_model_file(version, key):
    """Load a model version: the bundled .h5 for v1 (key is null), else from R2 (cached on disk)."""
    if key is None:
        path = BUNDLED_MODEL
    else:
        path = CACHE_DIR / f"v{version}.h5"
        if not path.exists():
            db.download_file(key, path)
    return keras.models.load_model(path, compile=False)


def get_model():
    """Return (model, version), reloading if a newer version has been promoted."""
    with _lock:
        if time.time() - _state["checked"] > CHECK_EVERY:
            _state["checked"] = time.time()
            try:
                info = db.current_model()
            except Exception as e:                      # DB down: keep serving what we have
                log.warning("could not check model version: %s", e)
                info = None
            if info and info["model_version"] != _state["version"]:
                log.info("loading model v%s", info["model_version"])
                _state["model"] = load_model_file(info["model_version"], info["model_key"])
                _state["version"] = info["model_version"]
        if _state["model"] is None:                     # first start without DB
            _state["model"], _state["version"] = load_model_file(1, None), 1
        return _state["model"], _state["version"]


def set_model(model, version):
    """Swap in a newly promoted model right away (used by retraining in this container)."""
    with _lock:
        _state.update(model=model, version=version, checked=time.time())


def predict_image(image, top_k=3):
    """Predict one PIL image. Returns the label, confidence and top-k classes."""
    model, version = get_model()
    x = to_array(image).astype(np.float32)[None, ...]   # (1, 224, 224, 3), values 0-255
    with _infer_lock:
        t0 = time.perf_counter()
        probs = model(x, training=False).numpy()[0]
        latency = (time.perf_counter() - t0) * 1000
    top = np.argsort(probs)[::-1][:top_k]
    label = CLASS_NAMES[top[0]]
    return {
        "label": label,
        "disease": SHORT_NAMES[label],
        "is_healthy": label.endswith("healthy"),
        "confidence": float(probs[top[0]]),
        "top_k": [{"label": CLASS_NAMES[i], "disease": SHORT_NAMES[CLASS_NAMES[i]],
                   "confidence": float(probs[i])} for i in top],
        "model_version": version,
        "latency_ms": round(latency, 1),
    }
