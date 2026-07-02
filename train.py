"""
Train / fine-tune the ASL letter detector.

Examples
--------
# Train a stronger yolov8s model on the bundled dataset, with heavy augmentation:
    python train.py

# Fine-tune the current model on your own captured webcam data (see collect.py):
    python train.py --base model.pt --extra my-data --epochs 60

# Once happy, promote the result to the model the app serves:
    python train.py --promote

The validation split is always ASL-DB/valid; the test split is reported at the
end so you can compare against the current model fairly.
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASE_DATASET = ROOT / "ASL-DB"
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple GPU
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def build_data_yaml(extra_dirs: list[Path]) -> Path:
    """Write a data.yaml that merges the base dataset with any extra train dirs.

    Ultralytics accepts a list of image directories for `train`, so extra
    datasets (e.g. your own webcam captures) are simply appended.
    """
    train_dirs = [str(BASE_DATASET / "train" / "images")]
    for d in extra_dirs:
        imgs = d / "images"
        if not imgs.is_dir():
            raise SystemExit(f"--extra dir has no images/ subfolder: {d}")
        train_dirs.append(str(imgs.resolve()))

    cfg = {
        "path": str(ROOT),
        "train": train_dirs,
        "val": str(BASE_DATASET / "valid" / "images"),
        "test": str(BASE_DATASET / "test" / "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    out = ROOT / "data.merged.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="yolov8s.pt", help="Base weights to start from (yolov8n/s/m.pt or model.pt).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=416, help="Must match how the app runs inference.")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--extra", nargs="*", default=[], help="Extra dataset dirs to merge in (each with images/ + labels/).")
    p.add_argument("--name", default="asl_v2", help="Run name under runs/detect/.")
    p.add_argument("--device", default=None, help="Override device (mps / cpu / 0).")
    p.add_argument("--lr0", type=float, default=0.01,
                   help="Initial LR. Lower (e.g. 0.002) when fine-tuning an already-trained model.")
    p.add_argument("--promote", action="store_true", help="Copy the trained best.pt to model.pt after training.")
    args = p.parse_args()

    device = args.device or pick_device()
    data_yaml = build_data_yaml([Path(e) for e in args.extra])
    print(f"▶ base={args.base}  device={device}  epochs={args.epochs}  imgsz={args.imgsz}")
    print(f"▶ data config: {data_yaml}")

    model = YOLO(args.base)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        patience=50,
        cos_lr=True,
        lr0=args.lr0,
        # --- Augmentation tuned for real-world robustness ---
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,   # lighting / colour variation
        degrees=10.0, translate=0.12, scale=0.5,  # hand position / size / angle
        fliplr=0.5,                          # works with either hand
        mosaic=1.0, mixup=0.1, erasing=0.4,  # occlusion / background variety
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\n✔ Trained weights: {best}")

    print("\n── Test-set metrics for the NEW model ──")
    YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=args.imgsz, plots=False)

    if (ROOT / "model.pt").exists():
        print("\n── Test-set metrics for the CURRENT model.pt (for comparison) ──")
        YOLO(str(ROOT / "model.pt")).val(data=str(data_yaml), split="test", imgsz=args.imgsz, plots=False)

    if args.promote:
        shutil.copy(best, ROOT / "model.pt")
        print(f"\n⬆ Promoted {best} → model.pt (the app will now serve it).")
    else:
        print(f"\nℹ Not promoted. If it's better, run:  cp '{best}' model.pt")


if __name__ == "__main__":
    main()
