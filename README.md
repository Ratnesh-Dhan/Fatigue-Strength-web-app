# Steel Fatigue Predictor

Single-project Flask web application for inference with the saved NIMS XGBoost fatigue-strength model. Uses `uv` for local dependency management.

## 1. Add the model files

Copy the trained files into `model/`:

- `fatigue_strength_xgb.json`
- `fatigue_strength_metadata.json`

The metadata should contain the ordered `features` list and `target` name. The app has the feature order from the training script as a fallback, but the metadata generated alongside the model should be used whenever possible.

## 2. Install dependencies

From this directory:

```bash
uv sync
```

This creates `.venv/` and a `uv.lock` lockfile.

## 3. Run locally

```bash
uv run python app.py
```

Open `http://127.0.0.1:5000`.

## Notes

- The model is loaded once when the Flask app starts.
- Inference uses a numeric NumPy row in the exact feature order from the training script.
- Enter values using the same units and scale as the training dataset.
- The training script reports MAE/RMSE in MPa; confirm target units in the dataset documentation.
- For public deployment, set a strong `FLASK_SECRET_KEY`, disable debug mode, and configure PythonAnywhere's WSGI file. `uv` is used locally; PythonAnywhere's deployment uses its configured Python virtual environment.
