---
title: ASL Translator
emoji: 🤟
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
short_description: Real-time American Sign Language fingerspelling translator
---

# 🤟 ASL Translator

Real-time **American Sign Language** fingerspelling translator. Show a hand sign to
your webcam and it predicts the letter **live**; or type text and see it
fingerspelled back. Built around a YOLOv8 hand-sign detection model and served with
[Gradio](https://www.gradio.app/).

**▶ Live demo:** https://huggingface.co/spaces/FaizD1/asl-translator
**💻 Source:** https://github.com/FaizD18/ASL-Translator

![ASL Translator in action](https://raw.githubusercontent.com/FaizD18/ASL-Translator/main/docs/screenshot.png)

## What it does

- **🔴 Live Camera** — continuous webcam streaming; the predicted letter and its
  confidence update in real time as you sign.
- **📷 Sign → Text** — capture a frame (or upload a photo), see the detection with a
  top-3 candidate picker for look-alike letters, and build a word or sentence.
- **⌨️ Text → Sign** — type a word or a whole sentence and see it fingerspelled in
  ASL, one letter at a time, with spaces preserved between words.

## Accuracy

| Metric | Score |
|---|---|
| Top-1 letter accuracy (held-out test set) | **87.5%** |
| mAP50 / mAP50-95 | 0.951 / 0.800 |

> These are measured on test images drawn from the same dataset as training, so
> they're an upper bound — live-webcam accuracy is lower (different lighting,
> background, and hand). See [Improving accuracy](#improving-accuracy) to close
> that gap with your own samples. Weakest letters are the closed-hand look-alikes
> (I, M, N), which the top-3 candidate picker helps with.

## How it works

```
webcam frame (raw RGB)
        │
        ▼
  YOLOv8n detector  ──  Ultralytics resizes to 416px + normalises internally
   (26 classes A–Z)
        │
        ▼
  bounding boxes → highest-confidence box → letter + confidence
```

The model is a plain object detector — **there is no MediaPipe / hand-landmark
step**. A raw frame goes straight in; Ultralytics does the letterboxing and
normalisation. The one rule that matters: **inference runs at 416px** (the size the
model was trained at). At the library's default 640px, accuracy drops to ~68%.

## Tech stack

- **Model:** Ultralytics YOLOv8n (PyTorch), trained on the Roboflow ASL Letters dataset
- **App / UI:** Gradio (native webcam streaming + Hugging Face Spaces support)
- **Serving:** Hugging Face Spaces (Gradio SDK)

## Project structure

| File | Role |
|------|------|
| `inference.py` | **Model layer** — `SignDetector` loads the model and predicts from a frame. No UI. |
| `app.py` | **UI layer** — the Gradio interface (Live / Sign→Text / Text→Sign). Imports `SignDetector`. |
| `model.pt` | Trained YOLOv8 weights the app serves. |
| `train.py` / `collect.py` | Train a better model / capture your own webcam samples. |
| `assets/letters/` | ASL fingerspelling images (a–z) for Text → Sign. |
| `requirements.txt` / `packages.txt` | Python and system dependencies. |
| `deploy_hf_space.sh` | One-command lean deploy to a Hugging Face Space. |
| `model-data/`, `ASL-DB/` | Old training runs and the dataset (not needed at runtime). |

## Run it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the URL Gradio prints (default <http://localhost:7860>). The app loads
`model.pt`; override with the `MODEL_PATH` environment variable.

## Deploy to Hugging Face Spaces

```bash
pip install -U "huggingface_hub[cli]"
hf auth login                       # token from https://huggingface.co/settings/tokens
# create the Space at https://huggingface.co/new-space  (SDK: Gradio)
./deploy_hf_space.sh <your-username>/<space-name>
```

`deploy_hf_space.sh` pushes only what the Space needs (`app.py`, `inference.py`,
`model.pt`, `assets/`, requirements) — not the dataset or old training runs — so
builds stay fast. The Space's README front-matter configures it automatically.

## Improving accuracy

The most effective way to improve live-webcam accuracy is to add examples from
**your own** camera and lighting:

```bash
# 1) Record ~30 samples per letter from your webcam (needs a camera + display).
python collect.py

# 2) Train on the bundled data plus your captures, and promote the result.
python train.py --base model.pt --extra my-data --epochs 60 --promote
```

`train.py` reports test-set metrics for the new vs current model so you can compare
before promoting.

> **Want more public data?** Drop additional [Roboflow](https://universe.roboflow.com/)
> ASL letter datasets (YOLOv8 export) into folders and merge them with repeated
> `--extra <dir>` flags.

## Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow ASL Letters Dataset](https://public.roboflow.com/object-detection/american-sign-language-letters)

## Contact

Faiz Daroga — [github.com/FaizD18/ASL-Translator](https://github.com/FaizD18/ASL-Translator)
