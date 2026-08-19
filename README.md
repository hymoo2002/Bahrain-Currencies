# Bahraini Currency Detector

A deep-learning app that recognizes Bahraini currency (notes and coins) from a photo or a live camera capture.

**Live app:** https://bahrain-currencies.streamlit.app/

Built as a guided lab: collect and augment images, train a CNN with transfer learning, and deploy it in a Streamlit app.

## Classes

9 classes, one folder per class in `augmented/`:

- Notes: `One`, `Five`, `Ten`, `Twenty` (BHD)
- Coins: `5`, `25`, `50`, `100`, `500` (fils)

## Project layout

```
CurrencyDetector/
├── augmented/                 # training data: one folder per class, 120 images each
├── pictures_mixed/            # original source photos
├── GeneratingImages.ipynb     # augments a single source photo into 120 variations
├── Training.ipynb             # trains MobileNetV2, saves model.keras + class_names.json
├── app.py                     # Streamlit app (camera + upload input)
├── requirements.txt
├── .python-version            # pins Python 3.11 for Streamlit Cloud
├── model.keras                # produced by Training.ipynb
├── class_names.json           # produced by Training.ipynb
└── training_report.txt        # accuracy + confusion matrix
```

## How it works

1. **Data augmentation** — `GeneratingImages.ipynb` takes one source photo per class and generates ~120 variations (rotations, flips, brightness/contrast/saturation shifts, blur, noise, crops, perspective warps, background swaps).
2. **Training** — `Training.ipynb` loads the `augmented/` folder as an 80/20 train/val split, fine-tunes MobileNetV2 (ImageNet weights) at 224×224, and saves the model.
3. **Inference** — `app.py` loads the model, takes a photo (browser camera or upload), preprocesses it the same way as training, and shows the predicted class + confidence.

## Run locally

```bash
pip install -r requirements.txt
```

Train the model (needed once; takes ~5–15 min on CPU):

```bash
jupyter notebook Training.ipynb
```

Launch the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (make sure `model.keras` and `class_names.json` are committed).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at `app.py`.
3. `.python-version` pins Python to 3.11 (TensorFlow doesn't support 3.13 yet). `requirements.txt` uses `tensorflow-cpu` to stay under the free-tier memory limit.

## Model

- Architecture: MobileNetV2 (ImageNet-pretrained) + GlobalAveragePooling + Dropout(0.3) + Dense(9, softmax)
- Input: 224 × 224 RGB, `preprocess_input` scaling to `[-1, 1]`
- Training: 8 epochs head-only (Adam 1e-3) → 6 epochs fine-tune top 30 layers (Adam 1e-5)
- Validation accuracy: ~99% on the augmented split

## Limitations

The augmentations are all derived from a single source photo per class. The model may not generalize perfectly to new phone photos with different lighting or backgrounds. To improve, collect 20–30 real photos per class and re-run the augmentation notebook on each of them.
