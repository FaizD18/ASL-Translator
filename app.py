"""
ASL Translator — a Gradio app that translates between American Sign Language
hand signs and text. This is the UI layer; all model logic lives in inference.py.

  • Live Camera : continuous webcam streaming with the predicted letter live.
  • Sign → Text : capture a frame (or upload a photo) and build a sentence.
  • Text → Sign : type a word or sentence and see the ASL fingerspelling for it.

Designed to run locally (`python app.py`) and to deploy as-is to a
Hugging Face Space (SDK: gradio, app_file: app.py).
"""

import base64
import html
from functools import partial
from pathlib import Path

import gradio as gr
import numpy as np

from inference import CLASS_NAMES, SignDetector

# --------------------------------------------------------------------------- #
# Paths & model
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
LETTERS_DIR = ROOT / "assets" / "letters"

N_CANDIDATES = 3  # alternative letters to surface in the UI
detector = SignDetector(n_candidates=N_CANDIDATES)


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
# Live camera (continuous streaming)
# --------------------------------------------------------------------------- #
def stream_predict(frame: np.ndarray | None):
    """Per-frame webcam handler: annotated frame + a confidence label dict."""
    pred = detector.predict(frame)
    if not pred.letter:
        return pred.annotated, {}
    return pred.annotated, dict(pred.candidates)


# --------------------------------------------------------------------------- #
# Sign → Text  (snapshot detection + sentence builder)
# --------------------------------------------------------------------------- #
def detect_sign(image: np.ndarray | None):
    """Run detection on a single captured frame.

    Returns (annotated_image, best_letter, status_markdown, candidates_state,
    and one button update per candidate slot).
    """
    blank = [gr.update(visible=False) for _ in range(N_CANDIDATES)]
    if image is None:
        return (None, "", "📸 Capture a webcam photo or upload an image, then press **Detect**.",
                [], *blank)

    pred = detector.predict(image)
    if not pred.letter:
        return (pred.annotated, "",
                "🤔 No hand sign detected. Try centering your hand and improving the lighting.",
                [], *blank)

    candidates = pred.candidates
    btn_updates = []
    for i in range(N_CANDIDATES):
        if i < len(candidates):
            letter, conf = candidates[i]
            btn_updates.append(gr.update(value=f"{letter}   {conf:.0%}", visible=True))
        else:
            btn_updates.append(gr.update(visible=False))

    if len(candidates) > 1:
        status = f"✅ Best guess **{pred.letter}** ({pred.confidence:.0%}). Not right? Pick a candidate below."
    else:
        status = f"✅ Detected **{pred.letter}**  ·  {pred.confidence:.0%} confidence"

    return pred.annotated, pred.letter, status, candidates, *btn_updates


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
THEME = gr.themes.Base(
    primary_hue="red",
    secondary_hue="rose",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0a0404",
    body_text_color="#f3e9e9",
    body_text_color_subdued="#c9a9a9",
    background_fill_primary="rgba(28,10,10,0.55)",
    background_fill_secondary="rgba(20,7,7,0.45)",
    block_background_fill="rgba(28,10,10,0.55)",
    block_border_color="rgba(200,60,60,0.28)",
    block_label_text_color="#ff8a8a",
    block_label_background_fill="rgba(60,16,16,0.6)",
    block_title_text_color="#f3e9e9",
    block_radius="18px",
    border_color_primary="rgba(200,60,60,0.28)",
    input_background_fill="rgba(0,0,0,0.40)",
    input_border_color="rgba(200,60,60,0.30)",
    button_large_radius="12px",
    button_primary_background_fill="linear-gradient(135deg,#c01f1f,#7d0f0f)",
    button_primary_background_fill_hover="linear-gradient(135deg,#d62828,#8f1212)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="rgba(45,16,16,0.7)",
    button_secondary_text_color="#f3e9e9",
    button_secondary_border_color="rgba(200,60,60,0.30)",
)


def _data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


WAVE_URI = _data_uri(ROOT / "assets" / "wave.png")

CSS = """
gradio-app {
    background: radial-gradient(125% 125% at 0% 100%,
        #5a0c0c 0%, #2a0808 38%, #0a0404 72%, #050202 100%) fixed !important;
}
.gradio-container { max-width: 1060px !important; margin: 0 auto !important; background: transparent !important; position: relative; }

/* Flowing wave line-art (bottom-left), screen-blended so only the bright lines show */
.gradio-container::before {
    content: ""; position: fixed; left: 0; bottom: 0; width: 70vw; height: 46vh;
    background: url('__WAVE__') no-repeat left bottom; background-size: contain;
    mix-blend-mode: screen; opacity: 0.55; pointer-events: none; z-index: 0;
}
.gradio-container > * { position: relative; z-index: 1; }

/* Header banner */
#hero {
    background: linear-gradient(120deg, #8a1414 0%, #3a0c0c 55%, #140707 100%);
    border: 1px solid rgba(220,90,90,.25);
    color: #fff; border-radius: 22px; padding: 30px 34px; margin-bottom: 18px;
    box-shadow: 0 16px 40px rgba(120,12,12,.4);
}
#hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0; line-height: 1.1; }
#hero p  { margin: 8px 0 0; opacity: .9; font-size: 1.02rem; }

/* Detected-letter + sentence */
#detected-letter textarea {
    font-size: 3rem !important; font-weight: 700; text-align: center;
    letter-spacing: 2px; color: #ff7b7b; background: transparent;
}
#sentence-box textarea { font-size: 1.5rem !important; line-height: 1.5; letter-spacing: 1px; color: #fff; }

/* Text → Sign strip */
.tts-strip { display: flex; flex-wrap: wrap; gap: 12px; padding: 8px 2px; align-items: flex-end; }
.tts-card {
    margin: 0; display: flex; flex-direction: column; align-items: center;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(220,90,90,.28);
    border-radius: 12px; padding: 8px;
}
.tts-card img { width: 76px; height: 76px; object-fit: contain; background: #fff; border-radius: 8px; }
.tts-card figcaption { margin-top: 4px; font-weight: 600; color: #ff9a9a; }
.tts-card--text { justify-content: center; }
.tts-glyph {
    width: 76px; height: 76px; display: flex; align-items: center;
    justify-content: center; font-size: 2.2rem; font-weight: 700; color: #c9a9a9;
}
.tts-space { width: 28px; }
.tts-empty { color: #c9a9a9; padding: 24px 8px; font-size: 1.05rem; }

/* Candidate-guess chips */
#candidates { gap: 8px; }
#candidates button {
    font-weight: 700; color: #fff; background: rgba(150,30,30,.5);
    border: 1px solid rgba(220,90,90,.45); border-radius: 10px; min-width: 0;
}
#candidates button:hover { background: rgba(185,40,40,.7); }
""".replace("__WAVE__", WAVE_URI)

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
        # ---------------- Live Camera (streaming) ----------------
        with gr.Tab("🔴 Live Camera"):
            gr.Markdown(
                "Point your hand at the webcam — the predicted letter updates **live**. "
                "On a free CPU Space this runs at a few frames per second; hold each sign "
                "briefly for a steady read."
            )
            with gr.Row():
                live_in = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam",
                    height=360,
                )
                with gr.Column():
                    live_out = gr.Image(label="Detection", height=300, interactive=False)
                    live_label = gr.Label(label="Prediction", num_top_classes=N_CANDIDATES)
            live_in.stream(
                stream_predict,
                inputs=live_in,
                outputs=[live_out, live_label],
                stream_every=0.3,        # ~3 fps — comfortable for free-tier CPU
                concurrency_limit=1,
                show_progress="hidden",
            )

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
