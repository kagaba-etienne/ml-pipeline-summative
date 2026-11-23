"""
Model building, evaluation and retraining (moved over from the notebook).

Retraining (champion vs challenger):
  1. take the pending uploads and download them from R2
  2. preprocess them: decode/validate, drop duplicates, resize to 224x224
  3. mix in a replay sample of the original training images (from R2, chosen with splits.csv)
  4. fine-tune a copy of the current model (pretrained MobileNetV2) with augmentation
  5. evaluate old and new on the same fixed test sample; promote the new one if its
     macro F1 is at least as good, saving it to R2 as a new version
"""
import gc
import hashlib
import logging
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, top_k_accuracy_score)
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers
from tensorflow.keras.applications import mobilenet_v2

from src import database as db
from src import prediction
from src.preprocessing import CLASS_NAMES, IMG_SIZE, MODELS_DIR, load_image, to_array

log = logging.getLogger(__name__)

MODEL_NAME = "crop_disease_mobilenetv2"
SPLITS = pd.read_csv(MODELS_DIR / "splits.csv")
BATCH_SIZE = 32

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomContrast(0.20),
    layers.RandomBrightness(0.20, value_range=(0, 255)),
], name="augmentation")


def build_model(num_classes, dropout=0.3, l2=1e-4, dense_units=256):
    """MobileNetV2 (ImageNet) backbone + small regularised head. Input: 0-255 RGB."""
    base = mobilenet_v2.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1, name="normalise")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=MODEL_NAME)


def compile_model(model, learning_rate):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])


def get_callbacks(patience=2):
    return [callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7)]


def with_augmentation(model):
    """Training wrapper: augmentation -> model. Only `model` is saved."""
    inp = keras.Input(shape=IMG_SIZE + (3,))
    return keras.Model(inp, model(augmentation(inp)), name=f"{model.name}_trainer")


def evaluate_model(model, x, y):
    """Accuracy, loss, precision, recall, F1, top-3 accuracy and ROC-AUC on arrays x (uint8), y (int)."""
    y_prob = model.predict(x.astype(np.float32), batch_size=BATCH_SIZE, verbose=0)
    y_pred = y_prob.argmax(1)
    labels = np.arange(len(CLASS_NAMES))
    m = {
        "accuracy": accuracy_score(y, y_pred),
        "loss": log_loss(y, np.clip(y_prob, 1e-7, 1), labels=labels),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
        "top3_accuracy": top_k_accuracy_score(y, y_prob, k=3, labels=labels),
        "roc_auc_ovr_macro": roc_auc_score(y, y_prob, multi_class="ovr", average="macro", labels=labels),
    }
    return {k: float(v) for k, v in m.items()}


# ---------- data for retraining ----------

def sample_split(split, per_class, seed):
    df = SPLITS[SPLITS.split == split]
    return df.groupby("label").sample(n=per_class, random_state=seed)


def load_from_r2(df, split, folder):
    """Download dataset/<split>/<file> for each row and return (x uint8, y int)."""
    pairs = [(f"dataset/{split}/{f}", folder / split / f) for f in df.file]
    failed = db.download_many(pairs)
    if failed:
        raise RuntimeError(f"{len(failed)} {split} images could not be downloaded from R2")
    x = np.stack([to_array(load_image(p.read_bytes())) for _, p in pairs])
    y = np.array([CLASS_NAMES.index(label) for label in df.label])
    return x, y


def prepare_uploads(uploads, folder):
    """Download and preprocess uploaded images. Returns (x, y, used_ids, rejected[(id, reason)])."""
    pairs = [(u["r2_key"], folder / "uploads" / PurePosixPath(u["r2_key"]).name) for u in uploads]
    db.download_many(pairs)
    seen = {r["md5"] for r in db.query("select md5 from uploads where status = 'used' and md5 is not null")}
    xs, ys, used, rejected = [], [], [], []
    for u, (_, path) in zip(uploads, pairs):
        if not path.exists():
            rejected.append((u["id"], "missing in storage")); continue
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        if md5 in seen:
            rejected.append((u["id"], "duplicate image")); continue
        try:
            xs.append(to_array(load_image(data)))           # validate + resize to 224x224
        except ValueError as e:
            rejected.append((u["id"], str(e))); continue
        seen.add(md5)
        ys.append(CLASS_NAMES.index(u["label"]))
        used.append((u["id"], md5))
    x = np.stack(xs) if xs else np.empty((0, *IMG_SIZE, 3), np.uint8)
    return x, np.array(ys, dtype=int), used, rejected


def make_ds(x, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, tf.one_hot(y, len(CLASS_NAMES))))
    if shuffle:
        ds = ds.shuffle(len(x))
    return ds.map(lambda a, b: (tf.cast(a, tf.float32), b)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------- retraining ----------

def retrain(run_id, epochs=3, learning_rate=1e-5, replay_per_class=50,
            val_per_class=20, test_per_class=30):
    """Run one retraining job and record the result in training_runs[run_id]."""
    work = Path(tempfile.mkdtemp(prefix="retrain_"))
    try:
        uploads = db.query("select id, r2_key, label from uploads where status = 'pending'")
        if not uploads:
            raise ValueError("no pending uploads")

        # 1-2. uploaded data: download + preprocess
        x_new, y_new, used, rejected = prepare_uploads(uploads, work)
        for uid, reason in rejected:
            db.query("update uploads set status = 'rejected', reason = %s, training_run_id = %s where id = %s",
                     (reason, run_id, uid))
        if not used:
            raise ValueError("all uploaded images were rejected during preprocessing")

        # 3. replay + validation + fixed test sample from the original dataset in R2
        x_rep, y_rep = load_from_r2(sample_split("train", replay_per_class, seed=run_id), "train", work)
        x_val, y_val = load_from_r2(sample_split("validation", val_per_class, seed=run_id), "validation", work)
        x_test, y_test = load_from_r2(sample_split("test", test_per_class, seed=42), "test", work)
        x_train, y_train = np.concatenate([x_new, x_rep]), np.concatenate([y_new, y_rep])
        log.info("run %s: %d new + %d replay images", run_id, len(x_new), len(x_rep))

        # 4. champion = current model; challenger = a copy fine-tuned on the new data
        current = db.current_model()
        champion = prediction.load_model_file(current["model_version"], current["model_key"])
        challenger = prediction.load_model_file(current["model_version"], current["model_key"])
        champion_metrics = evaluate_model(champion, x_test, y_test)

        trainer = with_augmentation(challenger)
        compile_model(trainer, learning_rate)
        trainer.fit(make_ds(x_train, y_train, shuffle=True), validation_data=make_ds(x_val, y_val),
                    epochs=epochs, callbacks=get_callbacks(), verbose=2)
        challenger_metrics = evaluate_model(challenger, x_test, y_test)

        # 5. promote if at least as good
        promoted = challenger_metrics["f1_macro"] >= champion_metrics["f1_macro"]
        version = key = None
        if promoted:
            version = current["model_version"] + 1
            key = f"models/v{version}.h5"
            path = prediction.CACHE_DIR / f"v{version}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            challenger.save(path)
            db.put_file(path, key)

        for uid, md5 in used:
            db.query("update uploads set status = 'used', md5 = %s, training_run_id = %s where id = %s",
                     (md5, run_id, uid))
        db.query("""update training_runs set status = 'done', n_new = %s, n_rejected = %s, n_replay = %s,
                        champion_metrics = %s, challenger_metrics = %s, promoted = %s,
                        model_version = %s, model_key = %s, finished_at = now()
                    where id = %s""",
                 (len(used), len(rejected), len(x_rep), db.jsonb(champion_metrics),
                  db.jsonb(challenger_metrics), promoted, version, key, run_id))
        if promoted:
            prediction.set_model(challenger, version)
        log.info("run %s done, promoted=%s", run_id, promoted)
    except Exception as e:
        log.exception("run %s failed", run_id)
        db.query("update training_runs set status = 'failed', error = %s, finished_at = now() where id = %s",
                 (str(e)[:1000], run_id))
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()
