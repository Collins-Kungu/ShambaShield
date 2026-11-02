"""Maize pest image model (maize_pest_mobilenetv2.keras): fall armyworm damage vs healthy leaf.

The model has ONE sigmoid output = probability of class index 1 ("healthy"). Probability > 0.5 -> healthy.
The saved model already contains its own Rescaling layer, so feed RAW 0-255 pixels (do NOT divide by 255).
Used by both the Streamlit app and the FastAPI backend, so they always give the same answer.

    import pest_model
    result = pest_model.predict(PIL_image)     # loads the model on first use (takes a few seconds)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)
CLASS_NAMES = ("fall_armyworm_damage", "healthy")      # index 0, index 1
DISPLAY = {"fall_armyworm_damage": "Fall Armyworm (Affected)", "healthy": "Healthy"}
THRESHOLD = 0.5
LOW_CONFIDENCE = 0.70
MODEL_FILENAME = "maize_pest_mobilenetv2.keras"

_HERE = Path(__file__).resolve().parent
_lock = threading.Lock()
_model = None


def model_path() -> Path:
    """MODEL_PATH env var, else model/ or the same folder as this file (works from any working directory)."""
    candidates = [os.environ.get("MODEL_PATH", ""), _HERE / "model" / MODEL_FILENAME, _HERE / MODEL_FILENAME]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found. Put it in the 'model' folder "
                            f"or set MODEL_PATH.")


def is_loaded() -> bool:
    return _model is not None


def load_model():
    """Load once and reuse (thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import keras      # imported here: TensorFlow is slow to import and not needed for other tasks
                _model = keras.saving.load_model(str(model_path()), compile=False)
    return _model


def predict(img: Image.Image) -> dict:
    """Classify one leaf photo."""
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype="float32")[None, ...]               # raw 0-255 pixels
    p_healthy = float(load_model().predict(x, verbose=0)[0][0])
    idx = int(p_healthy > THRESHOLD)
    confidence = p_healthy if idx == 1 else 1.0 - p_healthy
    label = CLASS_NAMES[idx]
    return {"label": label, "display": DISPLAY[label], "confidence": round(confidence, 4),
            "probability_healthy": round(p_healthy, 4), "low_confidence": confidence < LOW_CONFIDENCE}
