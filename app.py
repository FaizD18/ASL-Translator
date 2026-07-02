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

from inference import CLASS_NAMES, SignDetector, TemporalSmoother

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
def stream_predict(frame: np.ndarray | None, smoother: TemporalSmoother | None):
    """Per-frame webcam handler: annotated frame + a smoothed confidence dict.

    A per-session ``TemporalSmoother`` votes over the last few frames so the
    live letter reads steadily instead of flickering between similar signs.
    """
    smoother = smoother or TemporalSmoother()
    pred = detector.predict(frame)
    smoothed = smoother.update(pred.candidates)[:N_CANDIDATES]
    return pred.annotated, dict(smoothed), smoother


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
        return (None, "", "Capture a webcam photo or upload an image, then press **Detect**.",
                [], *blank)

    pred = detector.predict(image, tta=True)  # snapshots are one-shot: spend the extra time
    if not pred.letter:
        return (pred.annotated, "",
                "No hand sign detected. Try centering your hand and improving the lighting.",
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
        status = f"Best guess **{pred.letter}** ({pred.confidence:.0%}). Not right? Pick a candidate below."
    else:
        status = f"Detected **{pred.letter}**  ·  {pred.confidence:.0%} confidence"

    return pred.annotated, pred.letter, status, candidates, *btn_updates


def pick_candidate(candidates: list, index: int):
    """Set the detected letter from a clicked candidate button."""
    if candidates and index < len(candidates):
        letter = candidates[index][0]
        return letter, f"Using **{letter}**"
    return gr.update(), gr.update()


def add_letter(sentence: str, letter: str):
    if not letter:
        return sentence, "Nothing to add — run **Detect** first."
    sentence = (sentence or "") + letter.upper()
    return sentence, f"Added **{letter.upper()}**"


def add_space(sentence: str):
    return (sentence or "") + " ", "Added a space"


def backspace(sentence: str):
    return (sentence or "")[:-1], "Removed last character"


def clear_sentence():
    return "", "Cleared"


# --------------------------------------------------------------------------- #
# Text → Sign  (fingerspelling a word or sentence)
# --------------------------------------------------------------------------- #
def text_to_signs(text: str) -> str:
    """Render the text as a wrapped strip of ASL fingerspelling cards."""
    if not text or not text.strip():
        return (
            "<div class='tts-empty'>Type a word or sentence above to see it "
            "fingerspelled in ASL.</div>"
        )

    cards = []
    for i, ch in enumerate(text):
        delay = min(i, 20)  # stagger the pop-in, capped so long sentences don't crawl
        if ch == " ":
            cards.append("<div class='tts-space'></div>")
            continue
        uri = LETTER_URIS.get(ch.lower())
        if uri:
            cards.append(
                f"<figure class='tts-card' style='--i:{delay}'>"
                f"<img src='{uri}' alt='{ch.upper()}'/>"
                f"<figcaption>{ch.upper()}</figcaption>"
                "</figure>"
            )
        else:
            # Digits / punctuation we don't have a sign for: show the glyph.
            glyph = html.escape(ch)
            cards.append(
                f"<figure class='tts-card tts-card--text' style='--i:{delay}'>"
                f"<div class='tts-glyph'>{glyph}</div>"
                f"<figcaption>{glyph}</figcaption>"
                "</figure>"
            )
    return f"<div class='tts-strip'>{''.join(cards)}</div>"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#05060e",
    body_text_color="#e9ebf8",
    body_text_color_subdued="#9aa3c7",
    background_fill_primary="rgba(255,255,255,0.04)",
    background_fill_secondary="rgba(255,255,255,0.03)",
    block_background_fill="rgba(13,15,32,0.55)",
    block_border_color="rgba(255,255,255,0.09)",
    block_label_text_color="#a5b4fc",
    block_label_background_fill="rgba(99,102,241,0.16)",
    block_title_text_color="#e9ebf8",
    block_radius="16px",
    block_shadow="0 18px 40px rgba(2,4,18,0.45)",
    border_color_primary="rgba(255,255,255,0.09)",
    input_background_fill="rgba(255,255,255,0.04)",
    input_border_color="rgba(255,255,255,0.10)",
    input_border_color_focus="rgba(129,140,248,0.65)",
    button_large_radius="12px",
    button_small_radius="10px",
    button_primary_background_fill="linear-gradient(135deg,#6366f1,#22d3ee)",
    button_primary_background_fill_hover="linear-gradient(135deg,#7c83ff,#3ee0f5)",
    button_primary_text_color="#05060e",
    button_secondary_background_fill="rgba(255,255,255,0.06)",
    button_secondary_text_color="#e9ebf8",
    button_secondary_border_color="rgba(255,255,255,0.12)",
    # gr.Label confidence bars (default is orange — clashes with everything)
    stat_background_fill="linear-gradient(90deg,#6366f1,#22d3ee)",
)

CSS = """
@keyframes aurora {
    0%   { transform: translate(0,0) scale(1); }
    33%  { transform: translate(6%,-5%) scale(1.18); }
    66%  { transform: translate(-5%,4%) scale(1.06); }
    100% { transform: translate(0,0) scale(1); }
}
@keyframes fadeInUp { from { opacity:0; transform: translateY(20px); } to { opacity:1; transform: translateY(0); } }
@keyframes popIn    { from { opacity:0; transform: scale(.82) translateY(6px); } to { opacity:1; transform: scale(1) translateY(0); } }
@keyframes shimmer  { to { background-position: 200% center; } }
@keyframes pulse    { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.55; transform:scale(.8); } }

gradio-app { background: #05060e !important; }
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; background: transparent !important; position: relative; }

/* Slowly drifting aurora backdrop */
.gradio-container::before {
    content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
    filter: blur(90px); opacity: .5;
    background:
        radial-gradient(38% 38% at 18% 28%, #6366f1 0%, transparent 62%),
        radial-gradient(34% 34% at 82% 18%, #22d3ee 0%, transparent 62%),
        radial-gradient(46% 46% at 62% 82%, #a855f7 0%, transparent 62%);
    animation: aurora 22s ease-in-out infinite;
}
/* Faint dot-grid texture over the aurora */
.gradio-container::after {
    content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: radial-gradient(rgba(255,255,255,.14) 1px, transparent 1.4px);
    background-size: 30px 30px; opacity: .16;
    -webkit-mask-image: radial-gradient(70% 55% at 50% 0%, #000 0%, transparent 100%);
    mask-image: radial-gradient(70% 55% at 50% 0%, #000 0%, transparent 100%);
}
.gradio-container > * { position: relative; z-index: 1; }

/* Entrance animations */
#hero { animation: fadeInUp .6s ease both; }
.tabitem { animation: fadeInUp .45s ease both; background: transparent !important; border: none !important; }

/* ---------------------------------------------------------------- hero */
#hero {
    text-align: center;
    background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.025));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.10); border-radius: 24px;
    padding: 42px 36px 34px; margin-bottom: 22px;
    box-shadow: 0 24px 60px rgba(2,4,18,.55), inset 0 1px 0 rgba(255,255,255,.08);
}
#hero .eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: .72rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase;
    color: #a5b4fc; background: rgba(99,102,241,.14);
    border: 1px solid rgba(165,180,252,.35); border-radius: 999px;
    padding: 6px 14px; margin-bottom: 16px;
}
#hero .eyebrow .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #34d399; box-shadow: 0 0 10px #34d399;
    animation: pulse 2.2s ease-in-out infinite;
}
#hero h1 {
    font-size: 3.1rem; font-weight: 700; margin: 0; letter-spacing: -1.5px; line-height: 1.1;
    background: linear-gradient(90deg,#a5b4fc,#67e8f9,#c4b5fd,#a5b4fc);
    background-size: 200% auto; -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: shimmer 7s linear infinite;
}
#hero p { margin: 12px auto 0; color: #9aa3c7; font-size: 1.06rem; max-width: 620px; }
#hero .chips { display: flex; justify-content: center; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
#hero .chip {
    font-size: .78rem; font-weight: 600; color: #c7cdf0;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.11);
    border-radius: 999px; padding: 6px 14px;
}
#hero .chip b { color: #67e8f9; font-weight: 700; }

/* ---------------------------------------------------------- pill tabs */
.tabs { border: none !important; }
.tabs [role="tablist"] {
    justify-content: center; gap: 6px;
    background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.09) !important;
    border-radius: 999px; padding: 6px; width: fit-content; margin: 0 auto 10px;
    box-shadow: 0 10px 30px rgba(2,4,18,.4);
}
button[role="tab"] {
    border: none !important; border-radius: 999px !important;
    padding: 9px 26px !important; font-weight: 600; font-size: .95rem;
    color: #9aa3c7 !important; background: transparent;
}
button[role="tab"]:hover { color: #e9ebf8 !important; transform: none; }
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#22d3ee) !important;
    color: #05060e !important; font-weight: 700;
    box-shadow: 0 6px 20px rgba(99,102,241,.45);
}
.tabs [role="tablist"]::after, .tabs .tab-container::after { display: none !important; }

/* ------------------------------------------------------------- blocks */
.block { transition: border-color .25s ease, box-shadow .25s ease; }
.block:hover { border-color: rgba(165,180,252,.32) !important; box-shadow: 0 10px 34px rgba(99,102,241,.16); }
.image-container, .image-container .wrap, .image-container video, .image-container img { border-radius: 14px; }

/* hint copy + section labels */
.hint p { color: #9aa3c7 !important; font-size: .95rem; margin: 2px 4px 4px; }
.hint strong { color: #c7cdf0; }
.section-label p {
    text-transform: uppercase; letter-spacing: .16em; font-size: .74rem; font-weight: 700;
    color: #a5b4fc !important; margin: 14px 4px 2px;
}

/* Buttons: smooth lift + glow */
button { transition: transform .18s ease, box-shadow .25s ease, filter .2s ease !important; }
button:hover { transform: translateY(-2px); filter: brightness(1.07); }
button.primary:hover { box-shadow: 0 10px 28px rgba(34,211,238,.45); }

/* Live prediction label: big gradient top class */
#live-label .output-class {
    font-size: 2.4rem; font-weight: 700; letter-spacing: 2px;
    background: linear-gradient(90deg,#a5b4fc,#67e8f9);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* Detected-letter (gradient text) + sentence */
#detected-letter textarea {
    font-size: 3rem !important; font-weight: 700; text-align: center; letter-spacing: 2px;
    font-family: var(--font-mono);
    background: linear-gradient(90deg,#a5b4fc,#67e8f9); -webkit-background-clip: text; background-clip: text; color: transparent;
}
#sentence-box textarea {
    font-size: 1.45rem !important; line-height: 1.55; letter-spacing: 2px;
    color: #fff; font-family: var(--font-mono);
}

/* --------------------------------------------------- Text → Sign strip */
.tts-strip { display: flex; flex-wrap: wrap; gap: 12px; padding: 8px 2px; align-items: flex-end; }
.tts-card {
    margin: 0; display: flex; flex-direction: column; align-items: center;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px; padding: 8px;
    animation: popIn .38s ease both; animation-delay: calc(var(--i, 0) * 45ms);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.tts-card:hover { transform: translateY(-5px) scale(1.04); box-shadow: 0 12px 26px rgba(99,102,241,.4); border-color: rgba(165,180,252,.5); }
.tts-card img { width: 76px; height: 76px; object-fit: contain; background: #f4f5fb; border-radius: 10px; }
.tts-card figcaption { margin-top: 6px; font-weight: 700; color: #a5b4fc; font-family: var(--font-mono); }
.tts-card--text { justify-content: center; }
.tts-glyph {
    width: 76px; height: 76px; display: flex; align-items: center;
    justify-content: center; font-size: 2.2rem; font-weight: 700; color: #9aa3c7;
}
.tts-space { width: 28px; }
.tts-empty { color: #9aa3c7; padding: 24px 8px; font-size: 1.02rem; }

/* Candidate chips */
#candidates { gap: 8px; }
#candidates button {
    font-weight: 700; color: #e9ebf8; background: rgba(99,102,241,.18);
    border: 1px solid rgba(165,180,252,.4); border-radius: 10px; min-width: 0;
    font-family: var(--font-mono);
}
#candidates button:hover { background: rgba(99,102,241,.35); }

/* ------------------------------------------------------------- footer */
footer { display: none !important; }  /* Gradio's own footer duplicates ours */
#footer {
    text-align: center; color: #6b74a0; margin-top: 30px; padding: 18px 0 6px;
    border-top: 1px solid rgba(255,255,255,.07);
    font-size: .72rem; font-weight: 600; letter-spacing: .18em; text-transform: uppercase;
}
#footer b { color: #9aa3c7; }
"""

with gr.Blocks(title="ASL Translator") as demo:
    gr.HTML(
        """
        <div id="hero">
          <div class="eyebrow"><span class="dot"></span>Real-time sign recognition</div>
          <h1>ASL Translator</h1>
          <p>Translate American Sign Language and text, both ways — live from your webcam,
             powered by a YOLOv8 hand-sign detection model.</p>
          <div class="chips">
            <span class="chip"><b>26</b> letters, A–Z</span>
            <span class="chip"><b>Live</b> webcam streaming</span>
            <span class="chip"><b>Two-way</b> sign ↔ text</span>
          </div>
        </div>
        """
    )

    with gr.Tabs():
        # ---------------- Live Camera (streaming) ----------------
        with gr.Tab("Live Camera"):
            gr.Markdown(
                "Point your hand at the webcam — the predicted letter updates **live**. "
                "On a free CPU Space this runs at a few frames per second; hold each sign "
                "briefly for a steady read.",
                elem_classes="hint",
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
                    live_label = gr.Label(
                        label="Prediction", num_top_classes=N_CANDIDATES, elem_id="live-label"
                    )
            smoother_state = gr.State(None)
            live_in.stream(
                stream_predict,
                inputs=[live_in, smoother_state],
                outputs=[live_out, live_label, smoother_state],
                stream_every=0.3,        # ~3 fps — comfortable for free-tier CPU
                concurrency_limit=1,
                show_progress="hidden",
            )

        # ---------------- Sign → Text ----------------
        with gr.Tab("Sign → Text"):
            gr.Markdown(
                "Show a hand sign to your webcam (or upload a photo) and press "
                "**Detect**. Add each detected letter to build a word or sentence.",
                elem_classes="hint",
            )
            with gr.Row():
                with gr.Column():
                    cam = gr.Image(
                        sources=["webcam", "upload"],
                        type="numpy",
                        label="Your hand sign",
                        height=320,
                    )
                    detect_btn = gr.Button("Detect", variant="primary")
                with gr.Column():
                    annotated = gr.Image(label="Detection", height=320, interactive=False)
                    detected = gr.Textbox(
                        label="Detected letter",
                        elem_id="detected-letter",
                        interactive=True,
                        max_lines=1,
                    )
                    gr.Markdown("Other guesses (tap to use)", elem_classes="section-label")
                    with gr.Row(elem_id="candidates"):
                        cand_btns = [
                            gr.Button(visible=False, size="sm") for _ in range(N_CANDIDATES)
                        ]
                    candidates_state = gr.State([])
                    status = gr.Markdown("Capture a photo, then press **Detect**.")

            gr.Markdown("Your sentence", elem_classes="section-label")
            sentence = gr.Textbox(
                show_label=False,
                placeholder="Detected letters appear here…",
                elem_id="sentence-box",
                lines=2,
                interactive=True,
            )
            with gr.Row():
                add_btn = gr.Button("Add letter", variant="primary")
                space_btn = gr.Button("Space")
                back_btn = gr.Button("Backspace")
                clear_btn = gr.Button("Clear")

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
        with gr.Tab("Text → Sign"):
            gr.Markdown(
                "Type a word or a whole sentence and see it fingerspelled in ASL. "
                "Spaces separate words; unsupported characters are shown as text.",
                elem_classes="hint",
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

    gr.HTML("<div id='footer'><b>ASL Translator</b> · YOLOv8 · Built with Gradio</div>")


if __name__ == "__main__":
    # In Gradio 6 the theme and css are applied at launch time.
    demo.launch(theme=THEME, css=CSS)
