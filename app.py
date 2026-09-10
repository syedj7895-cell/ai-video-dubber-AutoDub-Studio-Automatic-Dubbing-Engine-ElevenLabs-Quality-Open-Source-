# ═══════════════════════════════════════════════════════════════════════════
#   AUTOMATIC DUBBING ENGINE — app.py
#   Phase 1 · "iPhone 15 frosted-glass" Gradio experience
#
#   Visual language (inspired by KokonutUI liquid-glass / React Bits /
#   Magic-UI beacons / shader mesh gradients — rebuilt in pure CSS so it
#   runs anywhere Gradio runs, including Google Colab):
#     · animated pastel mesh-gradient backdrop (whites, silvers, pastel blues)
#     · glass panels: rgba(255,255,255,0.4) · blur(25px) saturate(190%)
#       · 1px rgba(255,255,255,0.6) border · deep float shadow
#     · action buttons: linear-gradient(135deg, #E0EAFC, #CFDEF3)
#       hover → transform: scale(1.02) with a physics-eased transition
#     · premium vector SVG icons embedded on buttons (no plain text labels)
#
#   Tabs:  1 📂 File Import & Analysis   (pipeline steps 1–2)
#          2 📝 Script Matching & Assembly (pipeline steps 3–5)
#          3 🚀 Rendering Engine          (artifact pre-flight · steps 6–8 next)
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import html
import json
import threading
import time
from pathlib import Path

import gradio as gr

import pipeline

# ── global stop signal for the auto-pilot ──────────────────────────────────
# Set by the STOP button; checked by _run_full_auto between every stage.
_stop_event = threading.Event()

# Gradio 6 moved theme/css/head from Blocks() to launch() and dropped
# show_copy_button — detect once so the app runs on Gradio 4.x / 5.x / 6.x.
_GRADIO_MAJOR = int(str(gr.__version__).split(".")[0])
_IS_G6 = _GRADIO_MAJOR >= 6
_COPY_KW = {} if _IS_G6 else {"show_copy_button": True}

BASE_DIR = Path(__file__).resolve().parent
ICONS_DIR = BASE_DIR / "assets" / "icons"


def icon(name: str):
    """Absolute path to a packaged SVG icon (or None when missing)."""
    p = ICONS_DIR / f"{name}.svg"
    return str(p) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
#  CSS — the entire frosted-glass design system
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* ═══ AUTO-DUB STUDIO · iPhone-15 frosted-glass theme ═══ */
:root {
  --ink: #2c3a5c;
  --ink-soft: #5b6b8f;
  --glass-fill: rgba(255, 255, 255, 0.4);
  --glass-border: rgba(255, 255, 255, 0.6);
}

/* ---- animated pastel mesh-gradient backdrop (shader-gradient style) ---- */
body, .gradio-container {
  background: transparent !important;
}
.gradio-container::before {
  content: ""; position: fixed; inset: -22%; z-index: 0; pointer-events: none;
  background:
    radial-gradient(42% 46% at 16% 20%, rgba(207,222,243,0.95) 0%, rgba(207,222,243,0) 60%),
    radial-gradient(48% 50% at 84% 16%, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0) 62%),
    radial-gradient(52% 56% at 74% 82%, rgba(224,234,252,0.95) 0%, rgba(224,234,252,0) 64%),
    radial-gradient(42% 44% at 26% 84%, rgba(213,226,246,0.90) 0%, rgba(213,226,246,0) 60%),
    linear-gradient(158deg, #fbfdff 0%, #f0f4fb 46%, #e7eef9 100%);
  animation: meshDrift 28s ease-in-out infinite alternate;
}
@keyframes meshDrift {
  0%   { transform: translate3d(-2.2%, -1.6%, 0) scale(1.03) rotate(0.4deg); }
  50%  { transform: translate3d( 2.0%,  2.2%, 0) scale(1.07) rotate(-0.5deg); }
  100% { transform: translate3d(-1.2%,  1.4%, 0) scale(1.04) rotate(0.3deg); }
}
.gradio-container::after {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: radial-gradient(58% 40% at 50% -6%, rgba(255,255,255,0.9), rgba(255,255,255,0) 66%);
  mix-blend-mode: screen;
}
.gradio-container .main { position: relative; z-index: 1; }
footer { display: none !important; }

/* ---- liquid-glass panel (spec: .4 white fill · blur 25px · sat 190%) ---- */
.glass {
  background: var(--glass-fill) !important;
  -webkit-backdrop-filter: blur(25px) saturate(190%);
  backdrop-filter: blur(25px) saturate(190%);
  border: 1px solid var(--glass-border) !important;
  border-radius: 24px !important;
  box-shadow: 0 12px 40px rgba(31, 38, 135, 0.14),
              inset 0 1.5px 0 rgba(255, 255, 255, 0.85),
              inset 0 -18px 28px -24px rgba(31, 38, 135, 0.10);
}
.pad { padding: 18px 20px 22px; }

/* glassy input groups (file drops, text boxes, accordions) */
.gradio-container .form {
  background: rgba(255, 255, 255, 0.28) !important;
  border: 1px solid rgba(255, 255, 255, 0.55) !important;
  border-radius: 18px !important;
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
}
.gradio-container input, .gradio-container textarea {
  color: var(--ink) !important;
}

/* ---- premium action buttons (spec gradient + hover scale 1.02) ---- */
.gradio-container button.primary, .icon-btn {
  background: linear-gradient(135deg, #E0EAFC 0%, #CFDEF3 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.6) !important;
  color: #33477c !important;
  border-radius: 18px !important;
  font-weight: 700 !important;
  box-shadow: 0 10px 26px rgba(58, 84, 150, 0.18),
              inset 0 1px 0 rgba(255, 255, 255, 0.85) !important;
  transition: transform .28s cubic-bezier(.22, 1, .36, 1),
              box-shadow .28s cubic-bezier(.22, 1, .36, 1), filter .28s;
}
.gradio-container button.primary:hover, .icon-btn:hover {
  transform: scale(1.02);
  box-shadow: 0 16px 34px rgba(58, 84, 150, 0.26) !important;
  filter: saturate(1.08);
}
.gradio-container button.primary:active, .icon-btn:active { transform: scale(.985); }
.icon-btn { min-height: 56px; width: 100%; }
.icon-btn img, .icon-btn svg {
  height: 26px; width: 26px;
  filter: drop-shadow(0 1px 2px rgba(51, 71, 124, 0.35));
}
.btn-caption { margin-top: 8px; text-align: center; color: var(--ink-soft);
  font-size: .88rem; font-weight: 600; }

/* secondary / utility buttons */
.gradio-container button.secondary {
  background: rgba(255, 255, 255, 0.5) !important;
  border: 1px solid rgba(255, 255, 255, 0.65) !important;
  color: var(--ink) !important;
  border-radius: 14px !important;
  transition: transform .25s cubic-bezier(.22, 1, .36, 1);
}
.gradio-container button.secondary:hover { transform: scale(1.02); }

/* ---- hero ---- */
.hero { padding: 30px 26px 26px; text-align: center; }
.hero h1 {
  margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(92deg, #3d519c 0%, #7d9be0 45%, #b8c8ef 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero .badge {
  display: inline-block; margin-left: 10px; padding: 4px 12px; font-size: .72rem;
  font-weight: 700; letter-spacing: .04em; vertical-align: middle;
  color: #4a60a8; border-radius: 99px;
  background: rgba(255,255,255,.55); border: 1px solid rgba(255,255,255,.7);
}
.hero p { margin: 10px auto 0; color: var(--ink-soft); max-width: 660px; font-size: .98rem; }

/* ---- tabs (three visual stages) ---- */
.tab-nav { border: none !important; gap: 6px; }
.tab-nav button {
  font-size: 1rem; font-weight: 600; color: var(--ink-soft) !important;
  border-radius: 16px 16px 0 0 !important; padding: 12px 22px !important;
  background: rgba(255, 255, 255, 0.22);
  border: 1px solid rgba(255, 255, 255, 0.4) !important;
  transition: transform .25s cubic-bezier(.22, 1, .36, 1), background .25s;
}
.tab-nav button:hover { transform: translateY(-2px); }
.tab-nav button.selected {
  color: var(--ink) !important;
  background: rgba(255, 255, 255, 0.55);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.7) !important;
  box-shadow: 0 10px 26px rgba(31, 38, 135, 0.12);
}
.tabitem { animation: rise .45s cubic-bezier(.22, 1, .36, 1); }
@keyframes rise { from { opacity: 0; transform: translateY(10px); }
                  to   { opacity: 1; transform: none; } }

/* ---- status chips with pulsing beacon dot (Magic-UI beacon style) ---- */
.chip { display: inline-flex; align-items: center; gap: 9px; padding: 8px 16px;
  border-radius: 99px; background: rgba(255, 255, 255, 0.45);
  border: 1px solid rgba(255, 255, 255, 0.65);
  backdrop-filter: blur(18px) saturate(170%);
  -webkit-backdrop-filter: blur(18px) saturate(170%);
  box-shadow: 0 6px 18px rgba(31, 38, 135, 0.10);
  color: var(--ink); font-weight: 600; font-size: .9rem; }
.chip .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.chip.ok   .dot { background: #34d399; animation: pulse 1.9s ease-out infinite; }
.chip.run  .dot { background: #60a5fa; animation: pulse 1.1s ease-out infinite; }
.chip.err  .dot { background: #f87171; }
.chip.idle .dot { background: #94a3b8; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.45); }
                   100% { box-shadow: 0 0 0 12px rgba(96, 165, 250, 0); } }

/* console / dataframe / checklist */
.console textarea {
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important;
  font-size: .84rem !important;
  background: rgba(255, 255, 255, 0.5) !important;
  color: #33436e !important;
}
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(120, 140, 190, 0.35); border-radius: 99px; }

/* dropdown popups must sit above the background & capture clicks */
body > * { pointer-events: auto !important; }
/* ── dropdown z-index fix — popups render above glass panels ── */
gradio-dropdown, .gradio-dropdown, [data-testid="dropdown"],
.svelte-1m15bcr, .svelte-vt1mxs,
[data-testid="dropdown"], [class*="dropdown"], [class*="container"] {
  z-index: 9999 !important; position: relative; }
/* dropdown option list — force above everything */
[id*="dropdown"] [class*="options"], [class*="dropdown"] [class*="options"],
[role="listbox"], [role="option"], .svelte-1m15bcr [class*="options"],
.svelte-vt1mxs [class*="options"] {
  z-index: 10000 !important; position: absolute !important; }
/* dropdown arrow — larger clickable area */
[class*="dropdown"] [class*="arrow"], [class*="dropdown"] [class*="icon"],
[aria-label="Clear"] {
  min-width: 28px !important; min-height: 28px !important;
  padding: 6px !important; cursor: pointer !important;
  font-size: 1.2rem !important; }
/* ensure the dropdown container doesn't clip the popup */
.form, .gradio-dropdown, [data-testid="dropdown"] {
  overflow: visible !important; }>>>>>>> REPLACE

@media (prefers-reduced-motion: reduce) {
  .gradio-container::before, .chip .dot, .tabitem { animation: none !important; }
}
"""

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
"""

MIC_SVG = """
<svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke-width="1.6"
     stroke-linecap="round" stroke-linejoin="round"
     style="filter:drop-shadow(0 2px 6px rgba(61,81,156,.35)); flex:none;">
  <defs><linearGradient id="hgrad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#3d519c"/><stop offset="1" stop-color="#8fb0f0"/>
  </linearGradient></defs>
  <g stroke="url(#hgrad)">
    <rect x="9" y="2.4" width="6" height="12" rx="3"/>
    <path d="M5.2 11a6.8 6.8 0 0 0 13.6 0"/>
    <path d="M12 17.8v3.4M8.6 21.2h6.8"/>
  </g>
</svg>
"""

HERO = f"""
<div class="glass hero">
  <div style="display:flex;align-items:center;justify-content:center;gap:16px;">
    {MIC_SVG}
    <h1 style="margin:0;">AutoDub Studio<span class="badge">ELEVENLABS-QUALITY · OPEN-SOURCE</span></h1>
  </div>
  <p>Drop in a video + subtitles — AutoDub isolates the voices, maps every speaker and
     emotion, and prepares a cloned multilingual performance over the original score.</p>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Small HTML builders
# ─────────────────────────────────────────────────────────────────────────────

def _chip(kind: str, text: str) -> str:
    return f'<div class="chip {kind}"><span class="dot"></span>{html.escape(text)}</div>'


def _checklist(items: dict) -> str:
    if not items:
        rows = '<li style="opacity:.6;">Run Tabs 1 & 2 to light this up ✨</li>'
    else:
        rows = "".join(
            f'<li>{"✅" if ok else "⬜"} {html.escape(k)}</li>'
            for k, ok in items.items())
    return (f'<div class="glass" style="padding:18px 22px;">'
            f'<b style="color:#2c3a5c;">🚀 Render pre-flight · artifact chain</b>'
            f'<ul style="margin:10px 0 0 2px;padding-left:18px;color:#3f4f78;'
            f'line-height:1.9;list-style:none;">{rows}</ul></div>')


def _fp(f):
    """Gradio File value → plain builtin str path (NamedString-safe)."""
    if f is None:
        return None
    if isinstance(f, str):
        return str(f) or None          # NamedString → plain builtin str
    for attr in ("name", "path"):
        p = getattr(f, attr, None)
        if p:
            return str(p)
    if isinstance(f, (list, tuple)) and f:
        return _fp(f[0])
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  UI ⇄ pipeline callback generators (streaming console)
# ─────────────────────────────────────────────────────────────────────────────

def _run_analysis(media, srt_o, srt_t, token, lang_o, lang_t):
    """TAB 1 · steps 1–2 → (console, vocals preview, music preview, status)."""
    media_path = _fp(media)
    if not media_path:
        yield "⚠ Upload an audio/video master file first.", None, None, \
              _chip("idle", "Waiting for a master file")
        return
    yield "⏳ Booting import & analysis …", None, None, _chip("run", "Steps 1–2 in progress")
    last = ""
    try:
        for line in pipeline.run_import_and_analysis(
                media_path, force=False,
                source_lang=lang_o or "auto",
                target_lang=lang_t or "en"):
            last = line
            yield line, None, None, _chip("run", "Steps 1–2 in progress")
        state = pipeline.PipelineState.load()
        if state.is_done("step2"):
            yield last, str(pipeline.VOCALS_WAV), str(pipeline.MUSIC_WAV), \
                  _chip("ok", "Vocals & music ready — continue in Tab 2")
        else:
            yield last, None, None, _chip("err", "Halted — read the console")
    except Exception as e:  # pragma: no cover
        yield f"❌ Unexpected error: {e}", None, None, _chip("err", "Unexpected error")


def _run_matching(token, srt_o, srt_t):
    """TAB 2 · steps 3–5 → (console, dataframe, emotion log, clones, status)."""
    yield "⏳ Booting speaker matching …", None, None, None, _chip("run", "Steps 3–5 in progress")
    last = ""
    try:
        for line in pipeline.run_script_matching(_fp(token), _fp(srt_o), _fp(srt_t),
                                                 force=False):
            last = line
            yield line, None, None, None, _chip("run", "Steps 3–5 in progress")
        state = pipeline.PipelineState.load()
        if state.is_done("step5"):
            yield last, _script_rows(), _emotion_log_text(), _clone_files(), \
                  _chip("ok", "Script assembled — check Tab 3 pre-flight")
        else:
            yield last, None, None, None, _chip("err", "Halted — read the console")
    except Exception as e:  # pragma: no cover
        yield f"❌ Unexpected error: {e}", None, None, None, _chip("err", "Unexpected error")


def _run_readiness():
    """TAB 3 · artifact-chain pre-flight."""
    return _checklist(pipeline.render_readiness())


def _run_full_auto(media, srt_o, srt_t, token, lang_o, lang_t):
    """AUTO-PILOT · chains Tab1 → Tab2 → Tab3 with 15 s review pauses.

    Honours the global _stop_event: if set during a review pause or between
    stages, the pipeline halts gracefully instead of ploughing on."""
    global _stop_event
    _stop_event.clear()
    log = []
    def emit(text, status):
        log.append(text)
        return "\n".join(log), None, None, None, None, _chip(status, text)

    # ── helper: 15 s interruptible countdown ─────────────────────────────
    def pause_or_stop(label: str, seconds: int = 15):
        """Yields status updates each second; stops early if user hits STOP."""
        for remaining in range(seconds, 0, -1):
            if _stop_event.is_set():
                return
            yield emit(f"⏸ {label} · {remaining} s to review — "
                       "click STOP to halt", "ok")
            time.sleep(1)

    yield emit("⏳ Auto-pilot engaged · Tab 1 — extract & split …", "run")
    for line in pipeline.run_import_and_analysis(_fp(media), force=False,
                                                 source_lang=lang_o or "auto",
                                                 target_lang=lang_t or "en"):
        yield emit(line, "run")
    if _stop_event.is_set():
        yield emit("🛑 Auto-pilot stopped by user.", "err")
        return
    yield from pause_or_stop("Tab 1 review")
    if _stop_event.is_set():
        yield emit("🛑 Auto-pilot stopped by user.", "err")
        return

    yield emit("⏳ Auto-pilot · Tab 2 — diarization, emotions & script …", "run")
    for line in pipeline.run_script_matching(_fp(token), _fp(srt_o), _fp(srt_t),
                                             force=False):
        yield emit(line, "run")
    if _stop_event.is_set():
        yield emit("🛑 Auto-pilot stopped by user.", "err")
        return
    yield from pause_or_stop("Tab 2 review")
    if _stop_event.is_set():
        yield emit("🛑 Auto-pilot stopped by user.", "err")
        return

    yield emit("⏳ Auto-pilot · Tab 3 — rendering the dub …", "run")
    for line in pipeline.run_rendering(force=False):
        yield emit(line, "run")
    if _stop_event.is_set():
        yield emit("🛑 Auto-pilot stopped by user.", "err")
        return
    yield emit("🏁 Auto-pilot complete — download the master below.", "ok")


def _stop_auto():
    """Callback for the STOP button — flips the global stop flag."""
    _stop_event.set()
    return _chip("err", "STOP signalled — halting after current stage")


def _run_rendering():
    """TAB 3 · steps 6–8 → (console, final mix, final video, status)."""
    yield "⏳ Booting render engine …", None, None, _chip("run", "Steps 6–8 rendering")
    last = ""
    try:
        for line in pipeline.run_rendering(force=False):
            last = line
            yield line, None, None, _chip("run", "Steps 6–8 rendering")
        state = pipeline.PipelineState.load()
        if state.is_done("step8"):
            mix = str(pipeline.FINAL_MIX_WAV) if pipeline.FINAL_MIX_WAV.exists() else None
            vid = str(pipeline.FINAL_VIDEO_MP4) if pipeline.FINAL_VIDEO_MP4.exists() else None
            yield last, mix, vid, _chip("ok", "Render complete — download below")
        else:
            yield last, None, None, _chip("err", "Halted — read the console")
    except Exception as e:  # pragma: no cover
        yield f"❌ Unexpected error: {e}", None, None, _chip("err", "Unexpected error")


def _script_rows():
    if not pipeline.FINAL_SCRIPT_JSON.exists():
        return []
    data = json.loads(pipeline.FINAL_SCRIPT_JSON.read_text(encoding="utf-8"))
    return [[r["index"], pipeline.fmt_ts(r["start"]), pipeline.fmt_ts(r["end"]),
             r["speaker"], r["emotion"], r["translated_text"], r["original_text"]]
            for r in data]


def _emotion_log_text() -> str:
    p = pipeline.EMOTION_LOG_TXT
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _clone_files():
    if not pipeline.DIARIZATION_JSON.exists():
        return None
    diar = json.loads(pipeline.DIARIZATION_JSON.read_text(encoding="utf-8"))
    files: list = []
    for spk in diar.get("speakers", {}).values():
        files.extend(spk.get("clone_prompts", []))
    return files or None


# ─────────────────────────────────────────────────────────────────────────────
#  UI assembly
# ─────────────────────────────────────────────────────────────────────────────

def _make_theme() -> gr.themes.Soft:
    return gr.themes.Soft(
        primary_hue="indigo", secondary_hue="blue", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )


def build_ui() -> gr.Blocks:
    # Gradio ≥ 6 wants theme/css/head on launch(); ≤ 5 takes them here.
    block_kwargs = {} if _IS_G6 else dict(theme=_make_theme(), css=CSS, head=HEAD)
    with gr.Blocks(title="AutoDub Studio · Automatic Dubbing Engine",
                   **block_kwargs) as demo:

        gr.HTML(HERO)

        with gr.Tabs():

            # ════════════════ TAB 1 · FILE IMPORT & ANALYSIS ════════════════
            with gr.Tab("📂 File Import & Analysis"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["glass", "pad"]):
                        gr.Markdown("### 📥 Source material")
                        media_in = gr.File(label="🎬 Audio / Video master",
                                           file_types=["video", "audio"])
                        with gr.Row():
                            srt_orig_in = gr.File(label="🌐 Original SRT",
                                                  file_types=[".srt"])
                            srt_trans_in = gr.File(label="🌍 Translated SRT",
                                                   file_types=[".srt"])
                        with gr.Row():
                            lang_orig_in = gr.Dropdown(
                                choices=[(lbl, code) for code, lbl
                                         in pipeline.DUBBING_LANGUAGES],
                                value="auto", label="🎙 Original language",
                                info="Language of the source audio — used by "
                                     "the emotion/ASR scan",
                                interactive=True, allow_custom_value=False)
                            lang_target_in = gr.Dropdown(
                                choices=[(lbl, code) for code, lbl
                                         in pipeline.DUBBING_LANGUAGES],
                                value="en", label="🌍 Target / dub language",
                                info="Language of your Translated SRT — the "
                                     "cloned voices speak this",
                                interactive=True, allow_custom_value=False)
                        with gr.Accordion("🔑 Advanced — Hugging Face token "
                                          "(Pyannote diarization)", open=False):
                            hf_token_in = gr.Textbox(
                                type="password",
                                label="HF access token",
                                placeholder="hf_xxxxxxxxxxxx  (or set HUGGING_FACE_HUB_TOKEN)",
                                info="Accept the terms at huggingface.co/pyannote/"
                                     "speaker-diarization-3.1 first.")
                        analyze_btn = gr.Button(value="", icon=icon("analyze"),
                                                variant="primary",
                                                elem_classes=["icon-btn"])
                        gr.HTML('<div class="btn-caption">🔍 Run Import & Analysis — '
                                'extract the audio, then split vocals / music (Demucs v4)</div>')
                    with gr.Column(scale=6, elem_classes=["glass", "pad"]):
                        analysis_status = gr.HTML(_chip("idle", "Waiting for a master file"))
                        import_log = gr.Textbox(label="Pipeline console",
                                                lines=15, max_lines=26,
                                                interactive=False,
                                                **_COPY_KW,
                                                elem_classes=["console"])
                with gr.Row():
                    vocals_preview = gr.Audio(label="🎤 Isolated vocals",
                                              elem_classes=["glass", "pad"])
                    music_preview = gr.Audio(label="🎼 Isolated music",
                                             elem_classes=["glass", "pad"])

            # ════════════════ TAB 2 · SCRIPT MATCHING & ASSEMBLY ════════════
            with gr.Tab("📝 Script Matching & Assembly"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["glass", "pad"]):
                        gr.Markdown("### 🧬 Identity & emotion pass")
                        gr.Markdown("Runs **Pyannote 3.1** diarization, mines each "
                                    "speaker's cleanest 5–10 s clone prompts, scans "
                                    "**SenseVoice-Small** for paralinguistic emotions, "
                                    "and merges your translated SRT onto the grid.")
                        match_btn = gr.Button(value="", icon=icon("brain"),
                                              variant="primary",
                                              elem_classes=["icon-btn"])
                        gr.HTML('<div class="btn-caption">🧬 Match speakers & emotions, '
                                'then assemble the translated script (Steps 3–5)</div>')
                        gr.Markdown("ℹ️ Requires Tab 1 to be complete. The HF token from "
                                    "Tab 1 ▸ Advanced is reused here.")
                    with gr.Column(scale=6, elem_classes=["glass", "pad"]):
                        match_status = gr.HTML(_chip("idle", "Not started"))
                        match_log = gr.Textbox(label="Pipeline console",
                                               lines=15, max_lines=26,
                                               interactive=False,
                                               **_COPY_KW,
                                               elem_classes=["console"])
                gr.Markdown("### 🧾 Consolidated dubbing script")
                script_df = gr.DataFrame(
                    headers=["#", "Start", "End", "Speaker", "Emotion",
                             "Translated line", "Original line"],
                    interactive=False,
                    elem_classes=["glass"])
                with gr.Row():
                    with gr.Column(elem_classes=["glass", "pad"]):
                        gr.Markdown("### 🎭 Emotion log")
                        gr.Markdown("`[MM:SS.mmm] SpeakerN [emotion]`")
                        emotion_log_tb = gr.Textbox(label=None, lines=10, max_lines=18,
                                                    interactive=False,
                                                    **_COPY_KW,
                                                    elem_classes=["console"])
                    with gr.Column(elem_classes=["glass", "pad"]):
                        gr.Markdown("### 🗣 Clone prompts (mined per speaker)")
                        gr.Markdown("Top-3 cleanest 5–10 s voice references — "
                                    "CosyVoice-ready for the render phase.")
                        clone_files = gr.Files(label=None, interactive=False)

            # ════════════════ TAB 3 · RENDERING ENGINE ══════════════════════
            with gr.Tab("🚀 Rendering Engine"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["glass", "pad"]):
                        gr.Markdown("### ⚡ Auto-pilot (one-click)")
                        gr.Markdown("Runs the **entire pipeline end-to-end** — "
                                    "analysis → matching → render — with 15 s "
                                    "review pauses between stages.")
                        with gr.Row():
                            auto_btn = gr.Button(value="", icon=icon("rocket"),
                                                 variant="primary",
                                                 elem_classes=["icon-btn"],
                                                 scale=4)
                            stop_btn = gr.Button(value="🛑 STOP",
                                                 variant="secondary",
                                                 elem_classes=["icon-btn"],
                                                 scale=1)
                        gr.HTML('<div class="btn-caption">🚀 Auto-pilot — runs all 3 '
                                'tabs automatically · 🛑 STOP halts between stages</div>')
                        gr.Markdown("### 🎙 Render the dub")
                        gr.Markdown("**Step 6** splits the script into per-speaker "
                                    "TTS-safe channels · **Step 7** clones every line "
                                    "with **CosyVoice 3.0** zero-shot + emotion tags, "
                                    "padding each voice onto a silent master timeline "
                                    "with 1.15× overlap protection · **Step 8** mixes "
                                    "all voices over the Demucs instrumental with a "
                                    "lookahead −6 dB ducking curve, then remuxes the "
                                    "master into the original video.")
                        render_btn = gr.Button(value="", icon=icon("play"),
                                               variant="primary",
                                               elem_classes=["icon-btn"])
                        gr.HTML('<div class="btn-caption">🚀 Render — split, speak, '
                                'duck & mux (Steps 6–8)</div>')
                        preflight_btn = gr.Button(value="", icon=icon("sparkles"),
                                                  variant="secondary",
                                                  elem_classes=["icon-btn"])
                        gr.HTML('<div class="btn-caption">✨ Pre-flight — verify the '
                                'full artifact chain</div>')
                        ready_html = gr.HTML(_checklist({}))
                    with gr.Column(scale=6, elem_classes=["glass", "pad"]):
                        render_status = gr.HTML(_chip("idle", "Not rendered yet"))
                        render_log = gr.Textbox(label="Render console",
                                                lines=15, max_lines=26,
                                                interactive=False,
                                                **_COPY_KW,
                                                elem_classes=["console"])
                final_audio = gr.Audio(label="🎧 Final master mix",
                                       elem_classes=["glass", "pad"])
                final_video = gr.Video(label="🎬 Final dubbed video "
                                             "(when the source was a video)",
                                       elem_classes=["glass", "pad"])

        # ── event wiring ────────────────────────────────────────────────────
        analyze_btn.click(
            fn=_run_analysis,
            inputs=[media_in, srt_orig_in, srt_trans_in, hf_token_in,
                    lang_orig_in, lang_target_in],
            outputs=[import_log, vocals_preview, music_preview, analysis_status],
        )
        match_btn.click(
            fn=_run_matching,
            inputs=[hf_token_in, srt_orig_in, srt_trans_in],
            outputs=[match_log, script_df, emotion_log_tb, clone_files, match_status],
        )
        auto_btn.click(
            fn=_run_full_auto,
            inputs=[media_in, srt_orig_in, srt_trans_in, hf_token_in,
                    lang_orig_in, lang_target_in],
            outputs=[render_log, final_audio, final_video, render_status,
                     match_status, analysis_status],
        )
        render_btn.click(
            fn=_run_rendering,
            inputs=None,
            outputs=[render_log, final_audio, final_video, render_status],
        )
        preflight_btn.click(
            fn=_run_readiness,
            inputs=None,
            outputs=[ready_html],
        )

    return demo


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401  (type: ignore)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    demo = build_ui()
    launch_kwargs = dict(
        share=_in_colab(),                     # public link when on Colab
        server_name="0.0.0.0" if _in_colab() else "127.0.0.1",
        show_error=True,
    )
    if _IS_G6:                                 # v6 home for the visual params
        launch_kwargs.update(theme=_make_theme(), css=CSS, head=HEAD)
    demo.queue().launch(**launch_kwargs)
