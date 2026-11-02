"""Out-of-distribution gate for the SmartMaize pest classifier.

Pipeline:  image -> MobileNetV2 feature extractor (1280-d pooled features)
                 -> One-Class SVM  -> familiar?  yes -> binary classifier (pest_model)
                                                 no  -> "unrecognised", classifier is skipped

The One-Class SVM was trained on the pooled features of the MobileNetV2 backbone, so the
features are taken from the SAME backbone stored in maize_pest_mobilenetv2.keras
(Rescaling -> MobileNetV2 -> GlobalAveragePooling2D, i.e. the layers just before Dropout/Dense).
"""

import os
from pathlib import Path

import joblib
import numpy as np

try:  # Keras 3 standalone, falling back to tf.keras
    import keras
except ImportError:  # pragma: no cover
    from tensorflow import keras

_HERE = Path(__file__).resolve().parent
_SEARCH_DIRS = [_HERE / "model", _HERE]
MODEL_FILE = "maize_pest_mobilenetv2.keras"
DETECTOR_FILE = "ood_detector.joblib"
IMG_SIZE = (224, 224)

_state = {"backbone": None, "rescale": None, "detector": None}


def _find(filename):
    for d in _SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"'{filename}' not found. Place it in the 'model' folder (searched: "
        + ", ".join(str(d) for d in _SEARCH_DIRS) + ")."
    )


def is_loaded():
    return _state["detector"] is not None


def load():
    """Load the feature backbone and the One-Class SVM (once)."""
    if is_loaded():
        return
    full = keras.models.load_model(_find(MODEL_FILE), compile=False)
    _state["rescale"] = full.get_layer("rescaling")
    _state["backbone"] = full.get_layer("mobilenetv2_1.00_224")
    _state["detector"] = joblib.load(_find(DETECTOR_FILE))


def _features(image):
    """1280-d pooled backbone features for one PIL image."""
    arr = np.asarray(image.convert("RGB").resize(IMG_SIZE), dtype="float32")[None, ...]
    x = _state["rescale"](arr)
    fmap = _state["backbone"](x, training=False)
    return np.asarray(fmap).mean(axis=(1, 2))  # global average pooling


def check(image):
    """Return {'familiar': bool, 'score': float}.

    score is the SVM decision function: >= 0 means inside the learned
    'familiar maize leaf' region, < 0 means outside it.
    """
    load()
    feats = _features(image)
    score = float(_state["detector"].decision_function(feats)[0])
    return {"familiar": score >= 0.0, "score": score}
