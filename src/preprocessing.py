"""Image preprocessing and the upload folder rules."""
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CLASS_NAMES = json.loads((MODELS_DIR / "class_names.json").read_text())
IMG_SIZE = (224, 224)

# upload rules
ALLOWED_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
MAX_FILES = 200
MAX_BYTES = 10 * 1024 * 1024
IGNORED_NAMES = {"thumbs.db", "desktop.ini"}


def load_image(data: bytes) -> Image.Image:
    """Decode and validate an image. Raises ValueError if it is not a readable image."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()                      # catches truncated / corrupt files
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"not a valid image ({e.__class__.__name__})") from e


def resize(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)


def to_array(image: Image.Image) -> np.ndarray:
    """224x224x3 uint8 array. Pixel values stay 0-255: normalisation is inside the model."""
    return np.asarray(resize(image), dtype=np.uint8)


def check_upload_paths(files):
    """Check the folder structure <root>/<class>/<image> of a browser folder upload.

    `files` is a list of {"relative_path", "content_type"}. Hidden/system files are skipped.
    Returns (accepted, errors); accepted items get a "label" taken from the folder name.
    The batch should be rejected if `errors` is not empty.
    """
    accepted, errors = [], []
    for f in files:
        path, ctype = f["relative_path"], f["content_type"]
        parts = PurePosixPath(path).parts
        name = parts[-1] if parts else ""
        if name.startswith(".") or name.lower() in IGNORED_NAMES:
            continue
        if len(parts) != 3:
            errors.append(f"{path}: must be <folder>/<class>/<image>")
            continue
        label, ext = parts[1], PurePosixPath(name).suffix.lower()
        if label not in CLASS_NAMES:
            errors.append(f'{path}: unknown class "{label}"')
        elif ext not in ALLOWED_TYPES:
            errors.append(f"{path}: only .jpg, .jpeg and .png are allowed")
        elif ctype != ALLOWED_TYPES[ext]:
            errors.append(f"{path}: content type {ctype} does not match {ext}")
        else:
            accepted.append({**f, "label": label, "ext": ext})
    if not accepted and not errors:
        errors.append("no images found")
    if len(accepted) > MAX_FILES:
        errors.append(f"{len(accepted)} images; at most {MAX_FILES} per upload")
    return accepted, errors
