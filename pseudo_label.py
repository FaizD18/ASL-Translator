"""
Convert the Lexset Synthetic ASL Alphabet (classification folders) into a
YOLO detection dataset by pseudo-labeling with the current model.

The trick: the model's hand *localization* generalises to this data even when
its letter classification doesn't, and the true letter is already known from
the folder name. So each image gets the model's highest-confidence box with
the folder's letter as the class. Blank images become background examples
(empty label files), teaching the model not to fire on empty scenes.

Output layout (images are symlinks — no 6 GB copy):
    ASL-Data/yolo/images/<Letter>_<orig>.png
    ASL-Data/yolo/labels/<Letter>_<orig>.txt

Feed to training with:  python train.py --extra ASL-Data/yolo ...
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "ASL-Data" / "archive" / "Train_Alphabet"
OUT = ROOT / "ASL-Data" / "yolo"
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

BATCH = 64
MIN_CONF = 0.05      # any-class confidence: we only need the box, not the label
MIN_AREA, MAX_AREA = 0.01, 0.90  # sane hand-box size as a fraction of the image


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(ROOT / "model.pt")

    kept = skipped = blanks = 0
    for folder in sorted(SRC.iterdir()):
        if not folder.is_dir():
            continue
        letter = folder.name
        paths = sorted(folder.glob("*.png"))

        if letter == "Blank":
            for p in paths:
                name = f"{letter}_{p.stem}"
                link = OUT / "images" / f"{name}.png"
                if not link.exists():
                    link.symlink_to(p)
                (OUT / "labels" / f"{name}.txt").write_text("")
                blanks += 1
            print(f"{letter}: {blanks} background images", flush=True)
            continue

        cls = CLASS_NAMES.index(letter)
        folder_kept = 0
        for i in range(0, len(paths), BATCH):
            chunk = paths[i : i + BATCH]
            results = model([str(p) for p in chunk], imgsz=416, conf=MIN_CONF, verbose=False)
            for p, r in zip(chunk, results):
                if r.boxes is None or len(r.boxes) == 0:
                    skipped += 1
                    continue
                confs = r.boxes.conf.tolist()
                top = confs.index(max(confs))
                cx, cy, w, h = r.boxes.xywhn[top].tolist()
                if not (MIN_AREA <= w * h <= MAX_AREA):
                    skipped += 1
                    continue
                name = f"{letter}_{p.stem}"
                link = OUT / "images" / f"{name}.png"
                if not link.exists():
                    link.symlink_to(p)
                (OUT / "labels" / f"{name}.txt").write_text(
                    f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                )
                kept += 1
                folder_kept += 1
        print(f"{letter}: kept {folder_kept}/{len(paths)}", flush=True)

    print(f"\nDone. labeled={kept}  backgrounds={blanks}  skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
