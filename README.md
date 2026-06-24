---
title: ASL Translator
emoji: 🤟
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
---

# 🤟 ASL Translator

Translate between **American Sign Language** and **text**, both ways — powered by
a YOLOv8 hand-sign detection model, wrapped in a [Gradio](https://www.gradio.app/)
web app that runs locally or on a Hugging Face Space.

## Features

- **📷 Sign → Text** — show a hand sign to your webcam (or upload a photo); the
  model detects the letter (A–Z), which you can build into a full word or sentence
  with add / space / backspace / clear controls.
- **⌨️ Text → Sign** — type a word or an entire sentence and see it fingerspelled
  in ASL, one letter image at a time, with spaces preserved between words.

## Run it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the URL Gradio prints (default <http://localhost:7860>).

> The app loads the trained model from `model.pt`. Point it elsewhere with the
> `MODEL_PATH` environment variable.

## Deploy to Hugging Face Spaces

1. Create a new **Space** → SDK: **Gradio**.
2. Push these files to the Space repo: `app.py`, `requirements.txt`,
   `packages.txt`, `model.pt`, `assets/`, and this `README.md`
   (its front-matter configures the Space). The `model-data/` and `ASL-DB/`
   folders are **not** needed for the Space.
3. The Space builds automatically and serves the app.

```bash
# from a fresh clone of your Space repo
huggingface-cli login
git remote add space https://huggingface.co/spaces/<your-username>/asl-translator
git push space main
```

## Improving accuracy

The model scores ~0.95 mAP on the bundled test set but is less accurate on a live
webcam, because the training data is small and all from one source. The most
effective fix is to add examples from **your own** camera and lighting:

```bash
# 1) Record ~30 samples per letter from your webcam (needs a camera + display).
python collect.py

# 2) Train a stronger model (yolov8s + heavy augmentation) on the bundled data
#    plus your captures, and promote it to the model the app serves.
python train.py --base model.pt --extra my-data --epochs 60 --promote
```

`train.py` also works without your own data (`python train.py` trains yolov8s on
the bundled dataset) and reports test-set metrics for the new vs current model so
you can compare before promoting. Inference always runs at **416px** to match
training — running at any other size noticeably lowers accuracy.

> **Want more public data?** Drop additional [Roboflow](https://universe.roboflow.com/)
> ASL letter datasets (YOLOv8 export) into folders and pass them with repeated
> `--extra <dir>` flags to merge them into training.

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | The Gradio app (both translation directions). |
| `model.pt` | Trained YOLOv8 ASL letter-detection model the app serves. |
| `train.py` / `collect.py` | Train a better model / capture your own webcam data. |
| `assets/letters/` | ASL fingerspelling images (a–z) for Text → Sign. |
| `requirements.txt` / `packages.txt` | Python and system dependencies. |
| `model-data/`, `ASL-DB/` | Old training runs and the dataset (not needed at runtime). |

## Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow ASL Letters Dataset](https://public.roboflow.com/object-detection/american-sign-language-letters)

## Contact

Faiz Daroga — [github.com/FaizD18/ASL-Identifier](https://github.com/FaizD18/ASL-Identifier)
