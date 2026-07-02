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
# Design language: Apple's typography-led minimalism on a soft gray canvas,
# Airbnb's white cards and soft shadows, Uber's black pill CTAs. One accent
# color (Apple blue) used sparingly. The *_dark values mirror the light ones
# so the app looks identical for dark-mode visitors.
THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "-apple-system", "BlinkMacSystemFont",
          "SF Pro Text", "Helvetica Neue", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "SF Mono", "ui-monospace", "monospace"],
).set(
    body_background_fill="#f5f5f7", body_background_fill_dark="#f5f5f7",
    body_text_color="#1d1d1f", body_text_color_dark="#1d1d1f",
    body_text_color_subdued="#6e6e73", body_text_color_subdued_dark="#6e6e73",
    background_fill_primary="#ffffff", background_fill_primary_dark="#ffffff",
    background_fill_secondary="#fafafa", background_fill_secondary_dark="#fafafa",
    block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
    block_border_color="#e8e8ed", block_border_color_dark="#e8e8ed",
    block_label_text_color="#6e6e73", block_label_text_color_dark="#6e6e73",
    block_label_background_fill="#ffffff", block_label_background_fill_dark="#ffffff",
    block_title_text_color="#1d1d1f", block_title_text_color_dark="#1d1d1f",
    block_radius="16px",
    block_shadow="0 1px 3px rgba(0,0,0,.05)", block_shadow_dark="0 1px 3px rgba(0,0,0,.05)",
    border_color_primary="#e8e8ed", border_color_primary_dark="#e8e8ed",
    input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
    input_border_color="#d2d2d7", input_border_color_dark="#d2d2d7",
    input_border_color_focus="#0071e3", input_border_color_focus_dark="#0071e3",
    button_large_radius="999px",
    button_small_radius="999px",
    button_primary_background_fill="#1d1d1f", button_primary_background_fill_dark="#1d1d1f",
    button_primary_background_fill_hover="#000000",
    button_primary_background_fill_hover_dark="#000000",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#ffffff",
    button_secondary_text_color="#1d1d1f", button_secondary_text_color_dark="#1d1d1f",
    button_secondary_border_color="#d2d2d7", button_secondary_border_color_dark="#d2d2d7",
    # gr.Label confidence bars
    stat_background_fill="#0071e3", stat_background_fill_dark="#0071e3",
)

CSS = """
@keyframes fadeInUp { from { opacity:0; transform: translateY(14px); } to { opacity:1; transform: translateY(0); } }
@keyframes popIn    { from { opacity:0; transform: scale(.9) translateY(4px); } to { opacity:1; transform: scale(1) translateY(0); } }
@keyframes pulse    { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.5; transform:scale(.8); } }

gradio-app { background: #f5f5f7 !important; }
.gradio-container { max-width: 1040px !important; margin: 0 auto !important; background: transparent !important; }

/* Entrance */
#hero { animation: fadeInUp .5s ease both; }
.tabitem { animation: fadeInUp .35s ease both; background: transparent !important; border: none !important; }

/* ------------------------------------------------------------- topbar */
#topbar {
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 -8px; padding: 14px 8px;
    background: rgba(245,245,247,.82); backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid rgba(0,0,0,.08);
}
#topbar .brand { font-weight: 600; font-size: .95rem; color: #1d1d1f; letter-spacing: -.01em; }
#topbar a { color: #6e6e73; font-size: .85rem; text-decoration: none; font-weight: 500; }
#topbar a:hover { color: #0071e3; }

/* ---------------------------------------------------------------- hero */
#hero { text-align: center; padding: 58px 20px 44px; }
#hero .eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: .8rem; font-weight: 500; color: #1d1d1f;
    background: #ffffff; border: 1px solid #e8e8ed; border-radius: 999px;
    padding: 6px 14px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
#hero .eyebrow .dot {
    width: 7px; height: 7px; border-radius: 50%; background: #30d158;
    animation: pulse 2.4s ease-in-out infinite;
}
#hero h1 {
    font-size: 3.6rem; font-weight: 700; margin: 0; letter-spacing: -.035em;
    line-height: 1.06; color: #1d1d1f;
}
#hero p { margin: 16px auto 0; color: #6e6e73; font-size: 1.18rem; line-height: 1.5; max-width: 620px; }
#hero .meta { margin-top: 22px; color: #86868b; font-size: .9rem; font-weight: 500; }
#hero .meta span { padding: 0 10px; }
#hero .meta span + span { border-left: 1px solid #d2d2d7; }

/* -------------------------------------------- iOS-style segmented tabs */
.tabs { border: none !important; }
.tabs [role="tablist"] {
    justify-content: center; gap: 2px;
    background: #e8e8ed; border: none !important;
    border-radius: 12px; padding: 3px; width: fit-content; margin: 0 auto 26px;
}
button[role="tab"] {
    border: none !important; border-radius: 10px !important;
    padding: 8px 24px !important; font-weight: 500; font-size: .92rem;
    color: #1d1d1f !important; background: transparent;
}
button[role="tab"]:hover { transform: none; filter: none; }
button[role="tab"][aria-selected="true"] {
    background: #ffffff !important; color: #1d1d1f !important; font-weight: 600;
    box-shadow: 0 1px 4px rgba(0,0,0,.12);
}
.tabs [role="tablist"]::after, .tabs .tab-container::after,
button[role="tab"]::after, button[role="tab"]::before { display: none !important; }

/* ------------------------------------------------------------- blocks */
.block { transition: box-shadow .25s ease; }
.block:hover { box-shadow: 0 6px 20px rgba(0,0,0,.07); }
.image-container, .image-container .wrap, .image-container video, .image-container img { border-radius: 12px; }

/* hint copy + section labels */
.hint p { color: #6e6e73 !important; font-size: .95rem; margin: 2px 4px 8px; }
.hint strong { color: #1d1d1f; }
.section-label p { font-size: 1.05rem; font-weight: 600; color: #1d1d1f !important; margin: 18px 4px 4px; letter-spacing: -.01em; }

/* Buttons: quiet, precise */
button { transition: transform .15s ease, box-shadow .2s ease, background .2s ease !important; }
button:hover { transform: translateY(-1px); }
button.primary:hover { box-shadow: 0 4px 14px rgba(0,0,0,.22); }
button.secondary:hover { border-color: #1d1d1f !important; }

/* Live prediction label */
#live-label .output-class { font-size: 2.6rem; font-weight: 700; color: #1d1d1f; letter-spacing: .02em; }

/* Detected letter + sentence */
#detected-letter textarea {
    font-size: 3rem !important; font-weight: 700; text-align: center;
    letter-spacing: .05em; color: #1d1d1f; font-family: var(--font-mono);
}
#sentence-box textarea {
    font-size: 1.4rem !important; line-height: 1.55; letter-spacing: .08em;
    color: #1d1d1f; font-family: var(--font-mono);
}

/* --------------------------------------------------- Text → Sign strip */
.tts-strip { display: flex; flex-wrap: wrap; gap: 12px; padding: 8px 2px; align-items: flex-end; }
.tts-card {
    margin: 0; display: flex; flex-direction: column; align-items: center;
    background: #ffffff; border: 1px solid #e8e8ed; border-radius: 14px; padding: 9px;
    animation: popIn .32s ease both; animation-delay: calc(var(--i, 0) * 40ms);
    transition: transform .2s ease, box-shadow .2s ease;
}
.tts-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,.12); }
.tts-card img { width: 76px; height: 76px; object-fit: contain; background: #ffffff; border-radius: 8px; }
.tts-card figcaption { margin-top: 6px; font-weight: 600; color: #1d1d1f; font-family: var(--font-mono); font-size: .9rem; }
.tts-card--text { justify-content: center; }
.tts-glyph {
    width: 76px; height: 76px; display: flex; align-items: center;
    justify-content: center; font-size: 2.1rem; font-weight: 600; color: #86868b;
}
.tts-space { width: 26px; }
.tts-empty { color: #6e6e73; padding: 24px 8px; font-size: 1rem; }

/* Candidate chips (Airbnb filter-chip style) */
#candidates { gap: 8px; }
#candidates button {
    font-weight: 600; color: #1d1d1f; background: #ffffff;
    border: 1px solid #d2d2d7; border-radius: 999px; min-width: 0;
    font-family: var(--font-mono);
}
#candidates button:hover { border-color: #1d1d1f; background: #ffffff; }

/* ------------------------------------------------------------- footer */
footer { display: none !important; }  /* Gradio's own footer */
#footer {
    text-align: center; color: #86868b; margin-top: 40px; padding: 20px 0 10px;
    border-top: 1px solid #d2d2d7; font-size: .8rem;
}
#footer b { color: #1d1d1f; font-weight: 600; }
"""

with gr.Blocks(title="ASL Translator") as demo:
    gr.HTML(
        """
        <div id="topbar">
          <span class="brand">ASL Translator</span>
          <a href="https://github.com/FaizD18/ASL-Translator" target="_blank" rel="noopener">GitHub</a>
        </div>
        <div id="hero">
          <div class="eyebrow"><span class="dot"></span>Live demo</div>
          <h1>Sign language,<br>understood.</h1>
          <p>Translate American Sign Language to text and back — live from your
             webcam, powered by a YOLOv8 detection model.</p>
          <div class="meta">
            <span>26 letters, A–Z</span><span>Live webcam</span><span>Two-way translation</span>
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

    gr.HTML("<div id='footer'><b>ASL Translator</b> — YOLOv8 · Gradio</div>")


if __name__ == "__main__":
    # In Gradio 6 the theme and css are applied at launch time.
    demo.launch(theme=THEME, css=CSS)
