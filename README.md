# License Plate Segmentation — U‑Net (Keras)

Trains a U‑Net to segment license plates with three classes:
- 0: outside background
- 1: border ring around plate (white)
- 2: inside plate (black)

Two mask encodings are supported:
1) Alpha + grayscale PNG: outside has alpha=0, border=255 (white), inside=0 (black).
2) Three‑class PNG: single‑channel image with values {0,1,2}.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train
python -m src.train_unet --images data/images --masks data/masks --outdir models --epochs 50

# Predict on samples
python -m src.predict_unet --images data/test --model models/unet.keras --out results --overlay
```
