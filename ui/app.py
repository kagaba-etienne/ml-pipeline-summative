"""Streamlit UI. Talks only to the API (no database or storage keys here).

    API_URL=http://localhost:8000 PUBLIC_API_URL=http://localhost:8000 uv run streamlit run ui/app.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")            # server -> API
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", API_URL)              # browser -> API
WIDGET = (Path(__file__).parent / "upload_widget.html").read_text()
FEATURES = {"lesion_ratio": "Lesion coverage", "mean_hue": "Mean hue (°)",
            "brightness": "Brightness", "contrast": "Contrast", "edge_density": "Edge density"}

st.set_page_config(page_title="Tomato Leaf Doctor", page_icon="🍅", layout="wide")


def api(method, path, **kwargs):
    try:
        r = requests.request(method, API_URL + path, timeout=60, **kwargs)
    except requests.RequestException:
        st.error(f"The API at {API_URL} is not reachable.")
        st.stop()
    if not r.ok:
        detail = r.json().get("detail", r.text) if "json" in r.headers.get("content-type", "") else r.text
        return None, detail
    return r.json(), None


@st.cache_data(ttl=3600)
def dataset_stats():
    return api("GET", "/stats")[0]


def short(label):
    return dataset_stats()["short_names"].get(label, label)


# ---------- pages ----------

def dashboard():
    st.title("🍅 Dashboard")
    health, _ = api("GET", "/health")
    m, _ = api("GET", "/metrics")
    up = health["uptime_s"]
    c = st.columns(4)
    c[0].metric("API status", "🟢 up" if health["status"] == "ok" else "🔴 down")
    c[1].metric("Uptime", f"{up // 86400}d {up % 86400 // 3600}h {up % 3600 // 60}m")
    c[2].metric("Model version", f"v{m['model_version']}")
    c[3].metric("Predictions (24 h)", m["predictions"]["last_24h"], f"{m['predictions']['total']} total",
                delta_color="off")

    st.subheader(f"Current model (v{m['model_version']}): test metrics")
    t = m["test_metrics"] or {}
    names = {"accuracy": "Accuracy", "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)",
             "f1_macro": "F1 (macro)", "loss": "Loss"}
    cols = st.columns(len(names))
    for col, (k, name) in zip(cols, names.items()):
        if k in t:
            col.metric(name, f"{t[k]:.3f}")
    if m["model_version"] > 1:
        st.caption("Measured on the fixed 300-image test sample used for retraining comparisons.")
    else:
        st.caption("Measured on the full test set (2,722 images) in the notebook.")

    left, right = st.columns(2)
    hist = pd.DataFrame(m["notebook"]["history"])
    hist["epoch"] = range(1, len(hist) + 1)
    fig = px.line(hist, x="epoch", y=["accuracy", "val_accuracy", "f1", "val_f1"],
                  title="Training history of the original model (v1)")
    left.plotly_chart(fig, width="stretch")
    by_label = m["predictions"]["by_label"]
    if by_label:
        df = pd.DataFrame({"class": [short(k) for k in by_label], "predictions": list(by_label.values())})
        right.plotly_chart(px.bar(df, x="predictions", y="class", orientation="h",
                                  title="Predictions served, by predicted class"), width="stretch")
    else:
        right.info("No predictions yet.")
    if m["predictions"]["avg_latency_ms"]:
        right.caption(f"Average model latency: {m['predictions']['avg_latency_ms']:.0f} ms")


def predict():
    st.title("🔍 Predict")
    st.write("Upload a photo of a single tomato leaf.")
    file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
    if not file:
        return
    left, right = st.columns([1, 2])
    left.image(file, width="stretch")
    with st.spinner("Predicting..."):
        result, err = api("POST", "/predict", files={"file": (file.name, file.getvalue(), file.type)})
    if err:
        right.error(err)
        return
    if result["is_healthy"]:
        right.success(f"### Healthy leaf\nConfidence {result['confidence']:.1%}")
    else:
        right.error(f"### {result['disease']}\nConfidence {result['confidence']:.1%}")
    df = pd.DataFrame(result["top_k"])
    fig = px.bar(df, x="confidence", y="disease", orientation="h", range_x=[0, 1], title="Top 3 classes")
    fig.update_yaxes(autorange="reversed")
    right.plotly_chart(fig, width="stretch")
    right.caption(f"Model v{result['model_version']}, {result['latency_ms']:.0f} ms. Label: `{result['label']}`")


def insights():
    st.title("📊 Dataset insights")
    s = dataset_stats()
    st.write(f"{s['n_images']:,} tomato leaf images (PlantVillage), {s['n_classes']} classes. "
             f"Split: {s['split_counts']['train']:,} train / {s['split_counts']['validation']:,} validation / "
             f"{s['split_counts']['test']:,} test. Features below come from 200 images per class.")
    samples = pd.DataFrame(s["feature_samples"])
    samples["class"] = samples.label.map(short)
    order = samples.groupby("class").lesion_ratio.median().sort_values(ascending=False).index.tolist()

    st.subheader("1. Class balance")
    counts = pd.DataFrame({"class": [short(k) for k in s["class_counts"]], "images": list(s["class_counts"].values())})
    st.plotly_chart(px.bar(counts.sort_values("images"), x="images", y="class", orientation="h"),
                    width="stretch")
    st.markdown("""The classes are strongly imbalanced: Yellow Leaf Curl Virus has more than 10× the images of
mosaic virus. A model biased toward big classes could have high accuracy while missing rare diseases, so the
split is stratified, class weights are used in training, and **macro F1** (every disease counts equally) is
reported next to accuracy. Retraining decisions are based on macro F1 for the same reason.""")

    st.subheader("2. Leaf colour: lesion coverage")
    st.plotly_chart(px.box(samples, x="class", y="lesion_ratio", color="status", category_orders={"class": order},
                           labels={"lesion_ratio": FEATURES["lesion_ratio"]}), width="stretch")
    st.markdown("""Lesion coverage (share of yellow/brown leaf pixels) is the clearest colour signal. Late blight
has by far the highest coverage, followed by early blight, Septoria and bacterial spot, matching their large brown
dead areas. Healthy leaves and mosaic virus are close to zero. Most classes still overlap, so **colour alone cannot
identify the disease**; the model also needs texture and shape.""")

    st.subheader("3. Brightness and contrast")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.box(samples, x="class", y="brightness", category_orders={"class": order}),
                    width="stretch")
    c2.plotly_chart(px.box(samples, x="class", y="contrast", category_orders={"class": order}),
                    width="stretch")
    st.markdown("""Brightness is similar across classes (lab photos: one leaf, plain background, even lighting).
Contrast varies more: blights, target spot and spider mites are highest (dark lesions on lighter tissue), mosaic
virus and bacterial spot the most uniform. **Field photos with sunlight and shadows will look different**, which
is why augmentation changes contrast and brightness, and why retraining on real uploaded photos matters.""")

    st.subheader("4. Texture (edge density) vs lesion coverage")
    means = pd.DataFrame(s["feature_means_by_class"]).T.reset_index(names="label")
    means["class"] = means.label.map(short)
    st.plotly_chart(px.scatter(means, x="lesion_ratio", y="edge_density", text="class", size_max=12,
                               labels={"lesion_ratio": FEATURES["lesion_ratio"], "edge_density": FEATURES["edge_density"]}),
                    width="stretch")
    st.markdown("""Healthy leaves have the *most* texture (sharp close-ups with veins), bacterial spot the least, so
edge density reflects image sharpness as much as disease. Late blight stands apart on colour damage, healthy and
target spot on texture, while mosaic virus, leaf mold, Yellow Leaf Curl Virus and Septoria cluster together and
are the hardest to tell apart. **No single feature separates all ten classes**, which is why a CNN (pretrained
MobileNetV2) that learns combinations of colour, texture and shape is used.""")


def upload_retrain():
    st.title("⬆️ Upload data & retrain")
    st.subheader("1. Upload labelled images")
    config = {"api": PUBLIC_API_URL, "classes": dataset_stats()["classes"], "maxFiles": 200}
    st.iframe(WIDGET.replace("__CONFIG__", json.dumps(config)), height=300)
    with st.expander("Expected folder structure and class names"):
        st.code("my_batch/\n├── Tomato___Late_blight/\n│   ├── leaf1.jpg\n│   └── leaf2.png\n"
                "└── Tomato___healthy/\n    └── leaf3.jpg", language=None)
        st.write("Subfolder names must be exactly one of:")
        st.code("\n".join(dataset_stats()["classes"]), language=None)
        st.caption("Files go straight from your browser to storage. Images are validated, de-duplicated and "
                   "resized to 224×224 when retraining starts.")

    st.subheader("2. Retrain")
    st.button("Refresh")
    summary, _ = api("GET", "/uploads/summary")
    pending = summary["pending_by_label"]
    n_pending = sum(pending.values())
    c = st.columns(3)
    c[0].metric("Pending images", n_pending)
    c[1].metric("Used in training", summary["by_status"].get("used", 0))
    c[2].metric("Rejected", summary["by_status"].get("rejected", 0))
    if pending:
        st.bar_chart(pd.Series({short(k): v for k, v in pending.items()}, name="pending images"), horizontal=True)

    runs, _ = api("GET", "/retrain")
    running = next((r for r in runs if r["status"] == "running"), None)
    if st.button("🔁 Retrain model", type="primary", disabled=bool(running) or n_pending == 0):
        res, err = api("POST", "/retrain")
        if err:
            st.error(err)
        else:
            st.success(f"Retraining run #{res['run_id']} started.")
            running = res
    if n_pending == 0 and not running:
        st.caption("Upload some images first.")
    if running:
        st.info(f"Run #{running.get('id', running.get('run_id'))} is running (a few minutes). Press Refresh to update.")
    elif runs and runs[0]["status"] != "running" and runs[0]["model_version"] != 1:
        show_run(runs[0])


def show_run(run):
    if run["status"] == "failed":
        st.error(f"Run #{run['id']} failed: {run['error']}")
        return
    verdict = f"promoted to **v{run['model_version']}**" if run["promoted"] else "not promoted (current model kept)"
    st.success(f"Run #{run['id']}: trained on {run['n_new']} new + {run['n_replay']} replay images "
               f"({run['n_rejected']} rejected), {verdict}.")
    df = pd.DataFrame({"current model": run["champion_metrics"], "retrained model": run["challenger_metrics"]})
    st.dataframe(df.round(4), width="stretch")


def history():
    st.title("🕑 Training history")
    runs, _ = api("GET", "/retrain")
    if not runs:
        st.info("No runs yet.")
        return
    rows = []
    for r in runs:
        ch = r["challenger_metrics"] or {}
        cp = r["champion_metrics"] or {}
        rows.append({"run": r["id"], "status": r["status"], "promoted": r["promoted"],
                     "version": r["model_version"], "new images": r["n_new"], "rejected": r["n_rejected"],
                     "replay": r["n_replay"], "F1 before": cp.get("f1_macro"), "F1 after": ch.get("f1_macro"),
                     "accuracy after": ch.get("accuracy"), "started": r["started_at"], "error": r["error"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Run with version 1 is the original notebook model. A retrained model is promoted only if its "
               "macro F1 on the test sample is at least as good as the current model's.")
    choice = st.selectbox("Details for run", [r["id"] for r in runs if r["champion_metrics"]])
    if choice:
        show_run(next(r for r in runs if r["id"] == choice))


PAGES = {"Dashboard": dashboard, "Predict": predict, "Insights": insights,
         "Upload & Retrain": upload_retrain, "Training history": history}
page = st.sidebar.radio("Pages", list(PAGES))
PAGES[page]()
