"""
Capture your own ASL training data from a webcam.

Run locally (needs a camera + display):

    python collect.py

Controls
--------
  a–z      choose the letter you are about to sign
  SPACE    capture the current frame as a labelled sample for that letter
  u        undo the last capture
  q / ESC  quit

Each capture is saved to  my-data/images/<letter>_<n>.jpg  with a YOLO label in
my-data/labels/.  The bounding box is taken from the current model's best
detection (geometry only — the class is forced to the letter you picked); if the
model sees nothing, a centred box is used.  Aim for ~30 varied samples per letter
(different distances, angles, lighting, backgrounds), then:

    python train.py --base model.pt --extra my-data --epochs 60 --promote
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "my-data"
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}


def label_for(model, frame: np.ndarray, letter: str) -> str:
    """Return a single-line YOLO label (class cx cy w h, normalised) for `frame`.

    Uses the model's highest-confidence box for geometry but forces the class to
    `letter`; falls back to a centred box covering 60% of the frame.
    """
    h, w = frame.shape[:2]
    cls = CLASS_INDEX[letter]
    box = None
    if model is not None:
        r = model(frame, imgsz=416, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            i = int(np.argmax(r.boxes.conf.tolist()))
            box = r.boxes.xyxy[i].tolist()
    if box is None:
        cx, cy, bw, bh = 0.5, 0.5, 0.6, 0.6  # centred fallback
    else:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    import cv2
    from ultralytics import YOLO

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ROOT / "model.pt")) if (ROOT / "model.pt").exists() else None
    counts = {c: len(list((OUT / "images").glob(f"{c}_*.jpg"))) for c in CLASS_NAMES}
    letter = "A"
    last_saved: list[Path] = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (grant camera permission to your terminal).")

    print("Capturing. Pick a letter (a–z), press SPACE to save, q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        view = frame.copy()
        cv2.putText(view, f"Letter: {letter}   saved: {counts[letter]}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(view, "a-z pick   SPACE save   u undo   q quit", (12, view.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("ASL data collection", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            n = counts[letter]
            img_path = OUT / "images" / f"{letter}_{n:04d}.jpg"
            lbl_path = OUT / "labels" / f"{letter}_{n:04d}.txt"
            cv2.imwrite(str(img_path), frame)
            lbl_path.write_text(label_for(model, frame, letter) + "\n")
            counts[letter] += 1
            last_saved = [img_path, lbl_path]
            print(f"saved {img_path.name}  (total {letter}: {counts[letter]})")
        elif key == ord("u") and last_saved:
            for p in last_saved:
                p.unlink(missing_ok=True)
            counts[letter] = max(0, counts[letter] - 1)
            print(f"undid last capture for {letter}")
            last_saved = []
        elif 97 <= key <= 122:  # a–z
            letter = chr(key).upper()

    cap.release()
    cv2.destroyAllWindows()
    total = sum(counts.values())
    print(f"\nDone. {total} samples in {OUT}/  →  "
          f"train with:  python train.py --base model.pt --extra my-data --epochs 60 --promote")


if __name__ == "__main__":
    main()
