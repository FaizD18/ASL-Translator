"""
ASL Translator — a Gradio app that translates between American Sign Language
hand signs and text.

  • Sign → Text : show a hand sign to your webcam (or upload a photo); a YOLOv8
                  model detects the letter, which you can build into a sentence.
  • Text → Sign : type a word or sentence and see the ASL fingerspelling for it.

Designed to run locally (`python app.py`) and to deploy as-is to a
Hugging Face Space (SDK: gradio, app_file: app.py).
"""

import base64
import html
import os
from functools import lru_cache, partial
from pathlib import Path

import gradio as gr
import numpy as np

# --------------------------------------------------------------------------- #
# Paths & assets
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "model.pt"))
LETTERS_DIR = ROOT / "assets" / "letters"

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# The model was trained at 416px; running inference at this same size is far more
# accurate than Ultralytics' default of 640 (87% vs 68% top-1 on the test set).
IMG_SIZE = 416


@lru_cache(maxsize=1)
def get_model():
    """Load the YOLOv8 model once, lazily (keeps startup/import fast)."""
    from ultralytics import YOLO

    return YOLO(MODEL_PATH)


def _letter_data_uri(letter: str) -> str | None:
    """Return a base64 data URI for a letter image, or None if missing."""
    path = LETTERS_DIR / f"{letter.lower()}.png"
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


# Preload every letter image once so the Text → Sign view is self-contained.
LETTER_URIS = {c.lower(): _letter_data_uri(c) for c in CLASS_NAMES}


# --------------------------------------------------------------------------- #
# Sign → Text  (snapshot detection)
# --------------------------------------------------------------------------- #
N_CANDIDATES = 3       # how many alternative letters to surface
CAND_CONF = 0.02       # low threshold so near-miss classes (I/M/N) still appear


def _draw_best_box(image: np.ndarray, box, label: str) -> np.ndarray:
    """Draw a single clean detection box + label on a copy of the frame."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = (int(v) for v in box)
    draw.rectangle([x1, y1, x2, y2], outline=(217, 70, 239), width=4)
    draw.rectangle([x1, max(0, y1 - 20), x1 + 9 * len(label) + 8, y1], fill=(217, 70, 239))
    draw.text((x1 + 4, max(0, y1 - 18)), label, fill=(255, 255, 255))
    return np.asarray(img)


def detect_sign(image: np.ndarray | None):
    """Run detection on a single frame.

    Returns (annotated_image, best_letter, status_markdown, candidates_state,
    and one button update per candidate slot).
    """
    blank = [gr.update(visible=False) for _ in range(N_CANDIDATES)]
    if image is None:
        return (None, "", "📸 Capture a webcam photo or upload an image, then press **Detect**.",
                [], *blank)

    model = get_model()
    names = getattr(model, "names", None) or {i: n for i, n in enumerate(CLASS_NAMES)}
    result = model(image, imgsz=IMG_SIZE, conf=CAND_CONF, verbose=False)[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return (image, "", "🤔 No hand sign detected. Try centering your hand and improving the lighting.",
                [], *blank)

    confs = boxes.conf.tolist()
    classes = [int(c) for c in boxes.cls.tolist()]

    # Best confidence per letter, ranked — these are the candidate guesses.
    best_per_letter: dict[str, float] = {}
    for cls, conf in zip(classes, confs):
        letter = str(names[cls]).upper()
        best_per_letter[letter] = max(best_per_letter.get(letter, 0.0), conf)
    candidates = sorted(best_per_letter.items(), key=lambda kv: -kv[1])[:N_CANDIDATES]

    top_idx = int(np.argmax(confs))
    primary, primary_conf = candidates[0]
    annotated = _draw_best_box(image, boxes.xyxy[top_idx].tolist(), f"{primary} {primary_conf:.0%}")

    btn_updates = []
    for i in range(N_CANDIDATES):
        if i < len(candidates):
            letter, conf = candidates[i]
            btn_updates.append(gr.update(value=f"{letter}   {conf:.0%}", visible=True))
        else:
            btn_updates.append(gr.update(visible=False))

    if len(candidates) > 1:
        status = f"✅ Best guess **{primary}** ({primary_conf:.0%}). Not right? Pick a candidate below."
    else:
        status = f"✅ Detected **{primary}**  ·  {primary_conf:.0%} confidence"

    return annotated, primary, status, candidates, *btn_updates


def pick_candidate(candidates: list, index: int):
    """Set the detected letter from a clicked candidate button."""
    if candidates and index < len(candidates):
        letter = candidates[index][0]
        return letter, f"✍️ Using **{letter}**"
    return gr.update(), gr.update()


def add_letter(sentence: str, letter: str):
    if not letter:
        return sentence, "⚠️ Nothing to add — run **Detect** first."
    sentence = (sentence or "") + letter.upper()
    return sentence, f"➕ Added **{letter.upper()}**"


def add_space(sentence: str):
    return (sentence or "") + " ", "␣ Added a space"


def backspace(sentence: str):
    return (sentence or "")[:-1], "⌫ Removed last character"


def clear_sentence():
    return "", "🗑️ Cleared"


# --------------------------------------------------------------------------- #
# Text → Sign  (fingerspelling a word or sentence)
# --------------------------------------------------------------------------- #
def text_to_signs(text: str) -> str:
    """Render the text as a wrapped strip of ASL fingerspelling cards."""
    if not text or not text.strip():
        return (
            "<div class='tts-empty'>Type a word or sentence above to see it "
            "fingerspelled in ASL. 👆</div>"
        )

    cards = []
    for ch in text:
        if ch == " ":
            cards.append("<div class='tts-space'></div>")
            continue
        uri = LETTER_URIS.get(ch.lower())
        if uri:
            cards.append(
                "<figure class='tts-card'>"
                f"<img src='{uri}' alt='{ch.upper()}'/>"
                f"<figcaption>{ch.upper()}</figcaption>"
                "</figure>"
            )
        else:
            # Digits / punctuation we don't have a sign for: show the glyph.
            glyph = html.escape(ch)
            cards.append(
                "<figure class='tts-card tts-card--text'>"
                f"<div class='tts-glyph'>{glyph}</div>"
                f"<figcaption>{glyph}</figcaption>"
                "</figure>"
            )
    return f"<div class='tts-strip'>{''.join(cards)}</div>"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_radius="16px",
    button_large_radius="12px",
)

CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }

/* Header banner */
#hero {
    background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
    color: #fff; border-radius: 20px; padding: 30px 34px; margin-bottom: 18px;
    box-shadow: 0 12px 30px rgba(99,102,241,.28);
}
#hero h1 { font-size: 2.1rem; font-weight: 700; margin: 0; line-height: 1.1; }
#hero p  { margin: 8px 0 0; opacity: .92; font-size: 1.02rem; }

/* Detected-letter chip */
#detected-letter textarea {
    font-size: 3rem !important; font-weight: 700; text-align: center;
    letter-spacing: 2px; color: #4f46e5;
}
#sentence-box textarea {
    font-size: 1.5rem !important; line-height: 1.5; letter-spacing: 1px;
}

/* Text → Sign strip */
.tts-strip {
    display: flex; flex-wrap: wrap; gap: 12px; padding: 8px 2px;
    align-items: flex-end;
}
.tts-card {
    margin: 0; display: flex; flex-direction: column; align-items: center;
    background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.tts-card img { width: 76px; height: 76px; object-fit: contain; }
.tts-card figcaption { margin-top: 4px; font-weight: 600; color: #4f46e5; }
.tts-card--text { justify-content: center; }
.tts-glyph {
    width: 76px; height: 76px; display: flex; align-items: center;
    justify-content: center; font-size: 2.2rem; font-weight: 700; color: #64748b;
}
.tts-space { width: 28px; }
.tts-empty { color: #64748b; padding: 24px 8px; font-size: 1.05rem; }

/* Candidate-guess chips */
#candidates { gap: 8px; }
#candidates button {
    font-weight: 700; color: #4f46e5; background: #eef2ff;
    border: 1px solid #c7d2fe; border-radius: 10px; min-width: 0;
}
#candidates button:hover { background: #e0e7ff; }
"""

with gr.Blocks(title="ASL Translator") as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>🤟 ASL Translator</h1>
          <p>Translate American Sign Language and text, both ways — powered by a YOLOv8 hand-sign model.</p>
        </div>
        """
    )

    with gr.Tabs():
        # ---------------- Sign → Text ----------------
        with gr.Tab("📷 Sign → Text"):
            gr.Markdown(
                "Show a hand sign to your webcam (or upload a photo) and press "
                "**Detect**. Add each detected letter to build a word or sentence."
            )
            with gr.Row():
                with gr.Column():
                    cam = gr.Image(
                        sources=["webcam", "upload"],
                        type="numpy",
                        label="Your hand sign",
                        height=320,
                    )
                    detect_btn = gr.Button("🔍 Detect", variant="primary")
                with gr.Column():
                    annotated = gr.Image(label="Detection", height=320, interactive=False)
                    detected = gr.Textbox(
                        label="Detected letter",
                        elem_id="detected-letter",
                        interactive=True,
                        max_lines=1,
                    )
                    gr.Markdown("Other guesses (tap to use):")
                    with gr.Row(elem_id="candidates"):
                        cand_btns = [
                            gr.Button(visible=False, size="sm") for _ in range(N_CANDIDATES)
                        ]
                    candidates_state = gr.State([])
                    status = gr.Markdown("📸 Capture a photo, then press **Detect**.")

            gr.Markdown("### ✍️ Your sentence")
            sentence = gr.Textbox(
                show_label=False,
                placeholder="Detected letters appear here…",
                elem_id="sentence-box",
                lines=2,
                interactive=True,
            )
            with gr.Row():
                add_btn = gr.Button("➕ Add letter")
                space_btn = gr.Button("␣ Space")
                back_btn = gr.Button("⌫ Backspace")
                clear_btn = gr.Button("🗑️ Clear")

            detect_btn.click(
                detect_sign,
                inputs=cam,
                outputs=[annotated, detected, status, candidates_state, *cand_btns],
            )
            for i, btn in enumerate(cand_btns):
                btn.click(
                    partial(pick_candidate, index=i),
                    inputs=candidates_state,
                    outputs=[detected, status],
                )
            add_btn.click(add_letter, inputs=[sentence, detected], outputs=[sentence, status])
            space_btn.click(add_space, inputs=sentence, outputs=[sentence, status])
            back_btn.click(backspace, inputs=sentence, outputs=[sentence, status])
            clear_btn.click(clear_sentence, outputs=[sentence, status])

        # ---------------- Text → Sign ----------------
        with gr.Tab("⌨️ Text → Sign"):
            gr.Markdown(
                "Type a word or a whole sentence and see it fingerspelled in ASL. "
                "Spaces separate words; unsupported characters are shown as text."
            )
            text_in = gr.Textbox(
                label="Text to fingerspell",
                placeholder="e.g.  hello world",
                lines=2,
            )
            example = gr.Examples(
                examples=["hello", "good job", "i love asl"],
                inputs=text_in,
            )
            signs_out = gr.HTML(text_to_signs(""))
            text_in.change(text_to_signs, inputs=text_in, outputs=signs_out)

    gr.HTML(
        "<p style='text-align:center;color:#94a3b8;margin-top:18px;font-size:.9rem;'>"
        "ASL Translator · YOLOv8 · Built with Gradio</p>"
    )


if __name__ == "__main__":
    # In Gradio 6 the theme and css are applied at launch time.
    demo.launch(theme=THEME, css=CSS)
