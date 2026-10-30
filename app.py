from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fatigue_strength_xgb.json"
METADATA_PATH = MODEL_DIR / "fatigue_strength_metadata.json"

# Exact order from the NIMS training script: HEAT + CHEMISTRY + EXTRA.
DEFAULT_FEATURES = [
    "NT", "THT", "THt", "THQCr", "CT", "Ct", "DT", "Dt", "QmT", "TT", "Tt", "TCr",
    "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo",
    "RedRatio", "dA", "dB", "dC",
]
DEFAULT_TARGET = "Fatigue Strength"
DEFAULT_INPUT_VALUES = {
    # Heat-treatment and processing features
    "NT": 865,
    "THT": 865,
    "THt": 30,
    "THQCr": 24,
    "CT": 30,
    "Ct": 0,
    "DT": 30,
    "Dt": 0,
    "QmT": 30,
    "TT": 600,
    "Tt": 60,
    "TCr": 24,

    # Chemical composition
    "C": 0.38,
    "Si": 0.29,
    "Mn": 0.7,
    "P": 0.01,
    "S": 0.03,
    "Ni": 0.06,
    "Cr": 0.09,
    "Cu": 0.11,
    "Mo": 0.0,

    # Additional features
    "RedRatio": 610,
    "dA": 0.06,
    "dB": 0.01,
    "dC": 0.01,
}
FEATURE_GROUPS = [
    ("Heat-treatment and processing features", ["NT", "THT", "THt", "THQCr", "CT", "Ct", "DT", "Dt", "QmT", "TT", "Tt", "TCr"]),
    ("Chemical composition", ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo"]),
    ("Processing and inclusion descriptors", ["RedRatio", "dA", "dB", "dC"]),
]

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-development-key-change-before-deploy")

model = None
metadata: dict = {}
model_error = None
FEATURES = DEFAULT_FEATURES
TARGET = DEFAULT_TARGET

try:
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        FEATURES = metadata.get("features", DEFAULT_FEATURES)
        TARGET = metadata.get("target", DEFAULT_TARGET)

    if MODEL_PATH.exists():
        model = XGBRegressor()
        model.load_model(str(MODEL_PATH))
    else:
        model_error = (
            "The trained model file is missing. Copy fatigue_strength_xgb.json "
            "into the model/ directory."
        )
except (OSError, json.JSONDecodeError, ValueError, Exception) as exc:
    print("ERROR : ",exc)
    model = None
    model_error = f"Could not initialize the model: {exc}"


def grouped_features():
    """Return groups in the metadata's feature order."""
    feature_set = set(FEATURES)
    groups = []
    known = set()
    for title, names in FEATURE_GROUPS:
        selected = [name for name in names if name in feature_set]
        if selected:
            groups.append({"title": title, "features": selected})
            known.update(selected)
    remaining = [name for name in FEATURES if name not in known]
    if remaining:
        groups.append({"title": "Other model features", "features": remaining})
    return groups


@app.get("/")
def index():
    return render_template(
        "index.html",
        feature_groups=grouped_features(),
        features=FEATURES,
        model_ready=model is not None,
        model_error=model_error,
        metadata=metadata,
        # form_values={},
        form_values=DEFAULT_INPUT_VALUES,
    )


@app.post("/predict")
def predict():
    if model is None:
        flash(model_error or "The model is not available.", "error")
        return redirect(url_for("index"))

    values: dict[str, float] = {}
    errors = []
    for feature in FEATURES:
        raw = request.form.get(feature, "").strip()
        if not raw:
            errors.append(f"{feature} is required.")
            continue
        try:
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError
            values[feature] = value
        except ValueError:
            errors.append(f"{feature} must be a finite number.")

    if errors:
        for error in errors:
            flash(error, "error")
        return render_template(
            "index.html",
            feature_groups=grouped_features(),
            features=FEATURES,
            model_ready=True,
            model_error=None,
            metadata=metadata,
            form_values=request.form,
        ), 400

    # Training code used FEATURES in this order. Pass a NumPy array in that exact
    # order to remain compatible whether the model was fitted on arrays or a DataFrame.
    input_row = np.asarray([[values[name] for name in FEATURES]], dtype=float)
    try:
        prediction = float(model.predict(input_row)[0])
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "error")
        return redirect(url_for("index"))

    return render_template(
        "results.html",
        prediction=prediction,
        target=TARGET,
        inputs=values,
        features=FEATURES,
        metadata=metadata,
    )


if __name__ == "__main__":
    # Development server only. PythonAnywhere will use its WSGI configuration.
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
        )
