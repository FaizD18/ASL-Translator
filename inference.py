"""
Model / inference layer for the ASL Translator.

This module wraps the trained YOLOv8 letter-detection model and knows nothing
about the UI. Import ``SignDetector`` and call ``.predict(frame)``.

The model is an object detector (task=detect) over 26 classes (A–Z). It was
trained at 416px and inference MUST run at the same size — at Ultralytics'
default of 640px, top-1 accuracy drops from ~87% to ~68%. That single rule is
the entire "match the training preprocessing" story: Ultralytics handles the
letterbox-resize and normalisation internally, so a raw RGB frame goes straight
in. There is no MediaPipe / hand-landmark step.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "model.pt"))
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Trained input size. Do not change without retraining — see module docstring.
IMG_SIZE = 416


@dataclass
class Prediction:
    """Result of running the detector on one frame."""

    letter: str                          # best letter, "" if nothing detected
    confidence: float                    # confidence of the best letter (0–1)
    candidates: list[tuple[str, float]]  # top-k (letter, confidence), ranked
    annotated: np.ndarray | None         # frame with the best box drawn


def draw_box(image: np.ndarray, box, label: str) -> np.ndarray:
    """Draw a single clean detection box + label on a copy of the frame."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = (int(v) for v in box)
    draw.rectangle([x1, y1, x2, y2], outline=(217, 70, 239), width=4)
    draw.rectangle([x1, max(0, y1 - 20), x1 + 9 * len(label) + 8, y1], fill=(217, 70, 239))
    draw.text((x1 + 4, max(0, y1 - 18)), label, fill=(255, 255, 255))
    return np.asarray(img)


class SignDetector:
    """Loads the trained model once and predicts ASL letters from raw frames."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        imgsz: int = IMG_SIZE,
        candidate_conf: float = 0.02,  # low threshold so near-miss classes still surface
        n_candidates: int = 3,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.candidate_conf = candidate_conf
        self.n_candidates = n_candidates
        self._model = None

    @property
    def model(self):
        """Lazily load the YOLO model (keeps import/startup fast)."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    @property
    def names(self) -> dict:
        return getattr(self.model, "names", None) or dict(enumerate(CLASS_NAMES))

    def predict(self, frame: np.ndarray | None, tta: bool = False) -> Prediction:
        """Detect the ASL letter in a single RGB frame.

        ``tta=True`` enables test-time augmentation (multi-scale + flipped
        inference, averaged). It is a few times slower, so use it for one-shot
        snapshots, not the live stream.
        """
        if frame is None:
            return Prediction("", 0.0, [], None)

        result = self.model(
            frame, imgsz=self.imgsz, conf=self.candidate_conf, augment=tta, verbose=False
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Prediction("", 0.0, [], frame)

        confs = boxes.conf.tolist()
        classes = [int(c) for c in boxes.cls.tolist()]

        # Best confidence per letter, ranked → the candidate guesses.
        best_per_letter: dict[str, float] = {}
        for cls, conf in zip(classes, confs):
            letter = str(self.names[cls]).upper()
            best_per_letter[letter] = max(best_per_letter.get(letter, 0.0), conf)
        candidates = sorted(best_per_letter.items(), key=lambda kv: -kv[1])[: self.n_candidates]

        top_idx = int(np.argmax(confs))
        primary, primary_conf = candidates[0]
        annotated = draw_box(frame, boxes.xyxy[top_idx].tolist(), f"{primary} {primary_conf:.0%}")
        return Prediction(primary, primary_conf, candidates, annotated)


class TemporalSmoother:
    """Confidence-weighted vote over the last N frames of a live stream.

    Single-frame predictions flicker between similar letters (M/N, U/V…);
    aggregating a short window of per-frame candidates gives a much steadier —
    and in practice more accurate — live readout. One instance per stream
    session; call ``update`` with each frame's candidates.
    """

    def __init__(self, window: int = 5):
        self.history: deque[list[tuple[str, float]]] = deque(maxlen=window)

    def update(self, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Add one frame's (letter, conf) candidates; return the smoothed ranking."""
        self.history.append(candidates)
        scores: dict[str, float] = {}
        for frame_candidates in self.history:
            for letter, conf in frame_candidates:
                scores[letter] = scores.get(letter, 0.0) + conf
        n = len(self.history)
        return sorted(((let, s / n) for let, s in scores.items()), key=lambda kv: -kv[1])
