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

import csv
import html
import importlib
import json
import os
import threading
import time
from pathlib import Path

import gradio as gr

# ── which pipeline module to drive ───────────────────────────────────────────
# `pipeline.py` is the original six-engine build; `pipeline2.py` is the
# DEDICATED copy that AutoDub Studio 2.0 launches. One env var picks between
# them, so app.py never hard-codes a variant and neither pipeline file has to
# reference the other — they can evolve independently.
#   Colab_Runner.ipynb        → never sets it → pipeline
#   AutoDub Studio 2.0.ipynb  → AUTODUB_PIPELINE=pipeline2
#
# AUTODUB_PIPELINE_VARIANTS (optional, comma-separated) additionally renders a
# SWITCHER at the very top of Tab 1, so the driving module can be changed in an
# ALREADY-RUNNING app:
#   AutoDub Studio 2.0.ipynb  → AUTODUB_PIPELINE_VARIANTS=pipeline2,new_pipeline
# Leave it unset and no switcher is rendered at all, so Colab_Runner.ipynb's UI
# stays exactly as it was. The launch module is always offered first and is
# always the radio's default, so the control can never disagree with what is
# actually bound. Names that are not plain identifiers, or whose .py file is
# absent, drop out of the list instead of becoming a broken option.
BASE_DIR = Path(__file__).resolve().parent


def _is_module_name(name: str) -> bool:
    """True for a bare module name — no dots, no separators, not empty."""
    return bool(name) and name.isidentifier()


_PIPELINE_NAME = (os.environ.get("AUTODUB_PIPELINE") or "pipeline").strip()
if not _is_module_name(_PIPELINE_NAME):
    raise RuntimeError(
        f"AUTODUB_PIPELINE must be a plain Python module name, got "
        f"{_PIPELINE_NAME!r} — refusing to import it.")

_VARIANTS_RAW = (os.environ.get("AUTODUB_PIPELINE_VARIANTS") or "").strip()
_PIPELINE_VARIANTS: tuple = ()
if _VARIANTS_RAW:
    _extra = [n for n in (v.strip() for v in _VARIANTS_RAW.split(","))
              if _is_module_name(n) and n != _PIPELINE_NAME
              and (BASE_DIR / f"{n}.py").is_file()]
    _PIPELINE_VARIANTS = tuple(dict.fromkeys([_PIPELINE_NAME] + _extra))

# One import per name, cached, so flipping back and forth in Tab 1 is instant.
# Lazy on purpose: `new_pipeline.py` is the experimental twin, so a syntax
# error while it is being edited must not stop the app from BOOTING — it is
# caught by the switcher and reported in Tab 1 instead.
_LOADED: dict = {}


def _load_pipeline(name: str):
    """Import `name` once, then serve it from the cache."""
    if name not in _LOADED:
        _LOADED[name] = importlib.import_module(name)
    return _LOADED[name]


pipeline = _load_pipeline(_PIPELINE_NAME)

# ── global stop signal for the auto-pilot ──────────────────────────────────
# Set by the STOP button; checked by _run_full_auto between every stage.
_stop_event = threading.Event()

# Gradio 6 moved theme/css/head from Blocks() to launch() and dropped
# show_copy_button — detect once so the app runs on Gradio 4.x / 5.x / 6.x.
_GRADIO_MAJOR = int(str(gr.__version__).split(".")[0])
_IS_G6 = _GRADIO_MAJOR >= 6
_COPY_KW = {} if _IS_G6 else {"show_copy_button": True}

# JS-only copy-to-clipboard (Gradio 6 removed show_copy_button; this works
# on 4.x / 5.x / 6.x and falls back to execCommand on non-https origins).
_COPY_JS = ("(text) => {"
            "const fb=(t)=>{const ta=document.createElement('textarea');"
            "ta.value=t||'';ta.style.position='fixed';ta.style.opacity='0';"
            "document.body.appendChild(ta);ta.select();"
            "try{document.execCommand('copy');}catch(e){}"
            "document.body.removeChild(ta);};"
            "if(navigator.clipboard&&window.isSecureContext){"
            "navigator.clipboard.writeText(text||'').catch(()=>fb(text));}"
            "else{fb(text);}}")

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

def _run_analysis(media, srt_o, srt_t, token, lang_o, lang_t, translit,
                 diagnostic):
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
                target_lang=lang_t or "en",
                translit=bool(translit),
                diagnostic=bool(diagnostic)):
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


def _sidecar_rows(attr: str) -> list:
    """TAB 2 · read one of the active module's CSV side-cars verbatim.

    Reads the FILE rather than re-deriving from the JSON, so the table shows
    exactly what the export contains. After a switcher flip it therefore shows
    what the newly bound module wrote and never a stale blend of the two.
    Returns [] when the attribute is missing or the file has not been emitted
    yet (fresh session, or a build that predates side-cars).
    """
    path = getattr(pipeline, attr, None)
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    try:
        with p.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
    except Exception:
        return []
    return rows[1:] if rows else []


def _refresh_tables():
    """TAB 2 · reload the diarization-turn and emotion-grid tables."""
    return (_sidecar_rows("DIAR_CUES_CSV"),
            _sidecar_rows("EMOTION_GRID_CSV"))


def _run_matching(token, srt_o, srt_t, num_speakers, diagnostic):
    """TAB 2 · steps 3–5 → (console, dataframe, emotion log, clones, status)."""
    yield "⏳ Booting speaker matching …", None, None, None, _chip("run", "Steps 3–5 in progress")
    last = ""
    try:
        for line in pipeline.run_script_matching(
                _fp(token), _fp(srt_o), _fp(srt_t), force=False,
                num_speakers=int(num_speakers) if num_speakers else 0,
                diagnostic=bool(diagnostic)):
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


def _run_full_auto(media, srt_o, srt_t, token, lang_o, lang_t, num_speakers, translit, diagnostic):
    """AUTO-PILOT - chains Tab1 -> Tab2 -> Tab3 with 15 s review pauses.

    Console handling: pipeline yields are CUMULATIVE transcripts, so each
    stage's output REPLACES its own console section (zero duplication);
    auto-pilot status lines are appended once each."""
    global _stop_event
    _stop_event.clear()
    con = {"text": ""}

    def emit_msg(text, status):
        con["text"] = (con["text"] + "\n" + text).strip("\n")
        return (con["text"], None, None, None, None, _chip(status, text),
                _original_audio())

    def emit_stage(line, status, start, label):
        con["text"] = con["text"][:start] + line
        return (con["text"], None, None, None, None, _chip(status, label),
                _original_audio())

    def pause_or_stop(label: str, seconds: int = 15):
        """Yields status updates each second; stops early if user hits STOP."""
        for remaining in range(seconds, 0, -1):
            if _stop_event.is_set():
                return
            yield emit_msg(f"[pause] {label} - {remaining} s to review - "
                           "click STOP to halt", "ok")
            time.sleep(1)

    yield emit_msg("[auto] Auto-pilot engaged - Tab 1: extract & split ...", "run")
    s_start = len(con["text"])
    for line in pipeline.run_import_and_analysis(_fp(media), force=False,
                                                 source_lang=lang_o or "auto",
                                                 target_lang=lang_t or "en",
                                                 translit=bool(translit),
                                                 diagnostic=bool(diagnostic)):
        yield emit_stage(line, "run", s_start, "Tab 1 - steps 1-2 running")
    if _stop_event.is_set():
        yield emit_msg("[stop] Auto-pilot stopped by user.", "err")
        return
    yield from pause_or_stop("Tab 1 review")
    if _stop_event.is_set():
        yield emit_msg("[stop] Auto-pilot stopped by user.", "err")
        return

    yield emit_msg("[auto] Auto-pilot - Tab 2: diarization, emotions & script ...", "run")
    s_start = len(con["text"])
    for line in pipeline.run_script_matching(
            _fp(token), _fp(srt_o), _fp(srt_t), force=False,
            num_speakers=int(num_speakers) if num_speakers else 0,
            diagnostic=bool(diagnostic)):
        yield emit_stage(line, "run", s_start, "Tab 2 - steps 3-5 running")
    if _stop_event.is_set():
        yield emit_msg("[stop] Auto-pilot stopped by user.", "err")
        return
    yield from pause_or_stop("Tab 2 review")
    if _stop_event.is_set():
        yield emit_msg("[stop] Auto-pilot stopped by user.", "err")
        return

    yield emit_msg("[auto] Auto-pilot - Tab 3: rendering the dub ...", "run")
    s_start = len(con["text"])
    for line in pipeline.run_rendering(force=False, diagnostic=bool(diagnostic)):
        yield emit_stage(line, "run", s_start, "Tab 3 - steps 6-8 rendering")
    if _stop_event.is_set():
        yield emit_msg("[stop] Auto-pilot stopped by user.", "err")
        return
    yield emit_msg("[done] Auto-pilot complete - download the master below.", "ok")


def _stop_auto():
    """Callback for the STOP button — flips the global stop flag."""
    _stop_event.set()
    return _chip("err", "STOP signalled — halting after current stage")


def _original_audio():
    """TAB 3 · path to the source audio extracted in Step 1, else None.

    This is the file Step 1 pulled out of the upload, i.e. the track that is
    about to be replaced — not the Demucs vocal stem, which would be a
    stripped-down half of it. Returns None until Tab 1 has run, so the player
    simply stays empty rather than pointing at a file that isn't there.
    """
    p = pipeline.AUDIO_WAV
    return str(p) if p.exists() else None


def _run_rendering(diagnostic):
    """TAB 3 · steps 6–8 → (console, final mix, final video, status, source)."""
    src = _original_audio()
    yield "⏳ Booting render engine …", None, None, \
        _chip("run", "Steps 6–8 rendering"), src
    last = ""
    try:
        for line in pipeline.run_rendering(force=False, diagnostic=bool(diagnostic)):
            last = line
            yield line, None, None, _chip("run", "Steps 6–8 rendering"), src
        state = pipeline.PipelineState.load()
        if state.is_done("step8"):
            mix = str(pipeline.FINAL_MIX_WAV) if pipeline.FINAL_MIX_WAV.exists() else None
            vid = str(pipeline.FINAL_VIDEO_MP4) if pipeline.FINAL_VIDEO_MP4.exists() else None
            yield last, mix, vid, _chip("ok", "Render complete — download below"), src
        else:
            yield last, None, None, _chip("err", "Halted — read the console"), src
    except Exception as e:  # pragma: no cover
        yield f"❌ Unexpected error: {e}", None, None, \
            _chip("err", "Unexpected error"), src



def _spk_overview_rows():
    return [[pipeline.display_name(r["speaker"]), r["gender"] or "?",
             (f"{r['f0_hz']:.0f} Hz" if r.get("f0_hz") else "—"),
             r.get("gender_source", "—"),
             r["lines"], r["first"], r["last"], f"{r['total_s']}s"]
            for r in pipeline.speaker_overview()]


def _spk_prefill(speaker):
    """Fill the editor when a speaker is picked — so Save = confirm, not retype."""
    if not speaker:
        return "", ""
    row = next((r for r in pipeline.speaker_overview()
                if r["speaker"] == speaker), None)
    if row is None:
        return "", ""
    return (row.get("name") or "",
            row.get("gender_override") or row.get("auto_gender") or "")


def _spk_ids():
    return [r["speaker"] for r in pipeline.speaker_overview()]


def _save_profile(speaker, name, gender):
    if not speaker:
        return (_spk_overview_rows(),
                _chip("err", "No speaker selected"))
    pipeline.save_speaker_profile(speaker, name or "", gender or "")
    prof = pipeline.load_speaker_profiles().get(speaker, {})
    return (_spk_overview_rows(),
            _chip("ok", f"Saved: {speaker} is now '{prof.get('name', speaker)}' "
                        f"({prof.get('gender', '?')})"))


def _fix_line(row_no, speaker):
    if not row_no or not speaker:
        return (_script_rows(), _chip("err", "Pick a row # and a speaker"))

    ok = pipeline.set_row_speaker(int(row_no), speaker)
    return (_script_rows(),
            _chip("ok" if ok else "err",
                  f"Row {row_no} -> {speaker}" if ok else
                  f"Row {row_no} not found"))


def _preview_voice_app(engine_id, speaker, voice, pitch, rate, volume,
                       speed, partner, weight):
    """Synthesize one audition line and return it as a playable clip."""
    try:
        path, msg = pipeline.preview_voice(
            engine_id=engine_id or "", speaker=speaker or "",
            voice=voice or "", pitch=int(pitch or 0), rate=int(rate or 0),
            volume=int(volume or 0), speed=float(speed or 1.0),
            blend_partner=partner or "", blend_weight=float(weight or 0.5))
    except Exception as e:  # never let a preview crash the app
        return None, _chip("err", f"Preview failed ({type(e).__name__}: {e})")
    return (path, _chip("ok", msg)) if path else (None, _chip("err", msg))


def _save_voice_settings(pitch, rate, volume, speed, partner, weight):
    """Persist Edge prosody + Kokoro speed/blend; report exactly what is stored."""
    prosody = pipeline.set_engine_prosody(pitch=int(pitch or 0),
                                          rate=int(rate or 0),
                                          volume=int(volume or 0))
    spd = pipeline.set_kokoro_speed(float(speed or 1.0))
    blend = pipeline.set_kokoro_blend(partner or "", float(weight or 0.5))
    parts = [f"Edge pitch {prosody['pitch']:+d} Hz, rate {prosody['rate']:+d}%, "
             f"volume {prosody['volume']:+d}%",
             f"Kokoro speed ×{spd:.2f}"]
    if blend.get("partner"):
        parts.append(f"blend {blend['partner']} @ {float(blend['weight']):.2f}")
    return _chip("ok", "Saved · " + " · ".join(parts))



def _persist_bar(pct, msg):
    pct = max(0, min(100, int(pct)))
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    return (f'<div class="glass" style="padding:10px 14px;">'
            f'<code>[{bar}] {pct}%</code> · {msg}</div>')


def _start_upload(persist_on, backend, token):
    """Streaming upload: yields progress updates; never blocks the UI queue."""
    if not persist_on:
        yield _chip("err", "Persistence is OFF - enable it first"), \
              gr.update(), _chip("idle", "")
        return
    steps = []
    def _cb(pct, msg):
        steps.append((pct, msg))
    res = pipeline.upload_model_cache(progress_cb=_cb, backend=backend,
                                      hf_token=token or "")
    for pct, msg in steps:
        yield _persist_bar(pct, msg), gr.update(), \
              _chip("run", msg)
    yield _persist_bar(100, res), gr.update(), \
          _chip("ok" if "failed" not in res.lower() else "err", res)


def _toggle_persist(on):
    return _chip("ok" if on else "idle",
                 "Persistence ON - choose backend & click Start upload"
                 if on else "Persistence OFF")


def _bind_pipeline(name, orig, target):
    """TAB 1 · re-point the whole app at a DIFFERENT pipeline module.

    `pipeline` is a module-level global that every callback resolves at CALL
    time, so rebinding it here redirects Tab 1 analysis, Tab 2 matching, Tab 3
    render, the auto-pilot and the pre-flight in one move. Nothing captured the
    module at build time except the engine/language widgets — which is exactly
    why they are re-derived below, mirroring _switch_tts_engine().

    Deliberately tolerant: a name that was not offered, or that fails to
    import, reverts the radio and explains itself in the chip. It must never
    raise, because a raise would leave the radio showing the NEW name while
    `pipeline` still points at the OLD one — a silent half-switch. Returning a
    gr.update(value=<current>) keeps the two in step.

    An already-running stage is unaffected: its generator was built from the old
    module's function before the flip, so it finishes there and the new module
    takes over from the next stage onward.
    """
    global pipeline

    name = (name or "").strip()
    cur = pipeline.__name__
    if name not in _PIPELINE_VARIANTS:
        return (gr.update(value=cur),
                _chip("err", f"Unknown pipeline {name!r} — still on {cur}"),
                gr.update(), gr.update(), gr.update(), gr.update())

    try:
        mod = _load_pipeline(name)
    except Exception as e:  # a broken experimental twin must not be fatal
        return (gr.update(value=cur),
                _chip("err", f"Cannot load {name}.py — still on {cur} · "
                             f"{type(e).__name__}: {str(e)[:160]}"),
                gr.update(), gr.update(), gr.update(), gr.update())

    pipeline = mod

    # Same three re-derivations _switch_tts_engine does: the engine radio, and
    # both language menus, which are engine-scoped and would otherwise keep
    # advertising languages the new module's engine cannot speak.
    actual = mod.get_tts_engine()
    label = mod.TTS_ENGINES.get(actual, {}).get("label", actual)
    fb = mod.engine_fallback(actual)
    fb_note = f" · falls back to {fb}" if fb else " · terminal tier"
    orig_new = mod.language_choices(include_auto=True)
    tgt_new = mod.language_choices()
    o_val = orig if any(c == orig for _, c in orig_new) else "auto"
    t_val = target if any(c == target for _, c in tgt_new) else \
        (tgt_new[0][1] if tgt_new else None)
    # ⚠ This tuple must line up with outputs=[pipe_in, pipe_note,
    # tts_engine_note, tts_engine_in, lang_orig_in, lang_target_in]: the two
    # RADIOs take a `value`, the two HTML chips take markup. Swapping any pair
    # leaves a radio holding an HTML string that is not one of its choices —
    # the switch then looks like it silently did nothing.
    return (gr.update(value=name),
            _chip("ok", f"Pipeline: {name} (v{mod.PIPELINE_VARIANT}) · "
                        f"steps 3–4: {_step34_tools(mod)}"),
            _chip("ok", f"TTS Engine: {label}{fb_note}"),
            gr.update(choices=mod.engine_choices(), value=actual),
            gr.update(choices=orig_new, value=o_val),
            gr.update(choices=tgt_new, value=t_val))


def _switch_tts_engine(engine_id, orig, target):
    """Persist the engine, refresh its chip, and re-derive BOTH language menus.

    Each engine speaks a different set, so leaving the old choices on screen
    after a switch would keep offering languages the newly selected engine
    cannot synthesise — the target code goes straight to generate(). A pick the
    new engine still supports is kept; one it doesn't falls back to a sensible
    default rather than leaving the box blank.
    """
    actual = pipeline.set_tts_engine(engine_id)
    label = pipeline.TTS_ENGINES.get(actual, {}).get("label", actual)
    fb = pipeline.engine_fallback(actual)
    fb_note = f" · falls back to {fb}" if fb else " · terminal tier"
    orig_new = pipeline.language_choices(include_auto=True)
    tgt_new = pipeline.language_choices()
    o_val = orig if any(c == orig for _, c in orig_new) else "auto"
    t_val = target if any(c == target for _, c in tgt_new) else \
        (tgt_new[0][1] if tgt_new else None)
    return (_chip("ok", f"TTS Engine: {label}{fb_note}"),
            gr.update(choices=orig_new, value=o_val),
            gr.update(choices=tgt_new, value=t_val))


def _clear_cloud():
    msg = pipeline.clear_cloud_storage(include_models=True)
    return _chip("ok", msg)



def _refresh_speakers():
    ids = _spk_ids()
    row_nos = [r[0] for r in _script_rows()]
    v = ids[0] if ids else None
    r0 = row_nos[0] if row_nos else None
    return (gr.DataFrame(value=_spk_overview_rows()),
            gr.Dropdown(choices=ids, value=v),
            gr.Dropdown(choices=row_nos, value=r0),
            gr.Dropdown(choices=ids, value=v),
            gr.Dropdown(choices=ids, value=v))


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


def _pretty_engine(eid: str) -> str:
    """'CosyVoice 2.0  (default · offline)' → 'CosyVoice 2.0' for prose."""
    return pipeline.TTS_ENGINES.get(eid, {}).get("label", eid).split("  ")[0].strip()


def _step34_tools(mod) -> str:
    """Human-readable Step 3/4 toolchain published by pipeline module `mod`.

    Each build names its own stack in `STEP34_TOOLS` (Pyannote 3.1 +
    SenseVoice-Small for pipeline/pipeline2, Silero-VAD + CAM++ + emotion2vec+
    for new_pipeline). A module predating the switcher publishes no such
    attribute, so fall back to the engines it actually ran rather than guessing
    from its name.
    """
    return str(getattr(mod, "STEP34_TOOLS", None)
               or "Pyannote 3.1 · SenseVoice-Small")


def _pipeline_note() -> str:
    """Tab 1 chip naming the module CURRENTLY driving the app.

    Reads the live global rather than the launch-time choice, so after a
    switcher flip it reports the new module instead of staying stale.
    """
    return _chip("ok", f"Pipeline: {pipeline.__name__} "
                       f"(v{getattr(pipeline, 'PIPELINE_VARIANT', '?')}) · "
                       f"engine {_pretty_engine(pipeline.get_tts_engine())} · "
                       f"steps 3–4: {_step34_tools(pipeline)}")


def _engine_info() -> str:
    """Tab 1 helper copy, computed from the engines THIS build publishes.

    The original text hard-coded "CosyVoice 2.0 is the default … the other five
    are TERMINAL", which stops being true the moment a launcher narrows the
    registry (AutoDub Studio 2.0 publishes Chatterbox alone). Deriving it keeps
    the sentence honest in both builds without maintaining two strings.
    """
    ids = list(pipeline.TTS_ENGINES)
    act = pipeline.get_tts_engine()
    soft = [e for e in ids if pipeline.engine_fallback(e)]
    hard = [e for e in ids if not pipeline.engine_fallback(e)]
    out = [f"{_pretty_engine(act)} is the engine in use.",
           "Every engine casts a DISTINCT Hindi voice per speaker."]
    if len(ids) == 1:
        out.append(f"{_pretty_engine(act)} is TERMINAL — if it cannot load, the "
                   "run stops with a detailed report instead of silently "
                   "swapping in another voice.")
    elif soft:
        out.append(f"{_pretty_engine(soft[0])} falls back to Edge-TTS on failure; "
                   f"the other {len(hard)} are TERMINAL and report a detailed "
                   "error instead of silently swapping voices.")
    elif hard:
        out.append(f"All {len(hard)} are TERMINAL and report a detailed error "
                   "instead of silently swapping voices.")
    return " ".join(out)


def _engine_licence() -> str:
    """Licence/watermark notice — emitted only for engines actually published."""
    bits = []
    if "chatterbox" in pipeline.TTS_ENGINES:
        bits.append("Chatterbox embeds a Resemble PerTh neural watermark in "
                    "every clip it generates.")
    if "fishs2" in pipeline.TTS_ENGINES:
        bits.append("Fish Audio S2-Pro output is governed by the Fish Audio "
                    "Research License (non-commercial use).")
    if "edge" in pipeline.TTS_ENGINES:
        bits.append("Edge-TTS audio is synthesised remotely by Microsoft.")
    return ("⚠️ **Licence & watermark notice** — " + " ".join(bits)) if bits else ""


def build_ui() -> gr.Blocks:
    # Persisted voice/prosody settings → slider defaults (read once per build).
    _prosody = pipeline.get_engine_prosody()
    _kokoro_speed = pipeline.get_kokoro_speed()
    _kokoro_blend = pipeline.get_kokoro_blend()

    # Gradio ≥ 6 wants theme/css/head on launch(); ≤ 5 takes them here.
    block_kwargs = {} if _IS_G6 else dict(theme=_make_theme(), css=CSS, head=HEAD)
    with gr.Blocks(title="AutoDub Studio · Automatic Dubbing Engine",
                   **block_kwargs) as demo:

        gr.HTML(HERO)

        with gr.Tabs():

            # ════════════════ TAB 1 · FILE IMPORT & ANALYSIS ════════════════
            with gr.Tab("📂 File Import & Analysis"):
                # ── 🧩 pipeline switcher — the VERY top of Tab 1 ────────────
                # Rendered ONLY when the launcher asks for it (see the bind
                # block at the top of this file), so Colab_Runner.ipynb's Tab 1
                # is left byte-for-byte as it was.
                if len(_PIPELINE_VARIANTS) > 1:
                    with gr.Column(elem_classes=["glass", "pad"]):
                        gr.Markdown("### 🧩 Active pipeline")
                        gr.Markdown(
                            "Pick which build drives the three tabs — **no "
                            "restart needed**. Both share this session's "
                            "`outputs/` folder, so an in-flight project carries "
                            "on where it left off. The switch applies from the "
                            "**next stage you run**; a stage already iterating "
                            "finishes on the module it started with.")
                        pipe_in = gr.Radio(
                            choices=[(n, n) for n in _PIPELINE_VARIANTS],
                            value=_PIPELINE_NAME,
                            label="Pipeline module",
                            info="The launch default is listed first. "
                                 "new_pipeline.py is the experimental twin: "
                                 "edit it freely — a broken copy is reported "
                                 "here rather than stopping the app.",
                            interactive=True)
                        pipe_note = gr.HTML(_pipeline_note())
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["glass", "pad"]):
                        gr.Markdown("### 🎙 TTS Engine — pick the voice synthesizer")
                        tts_engine_in = gr.Radio(
                            choices=pipeline.engine_choices(),
                            value=pipeline.get_tts_engine(),
                            label="Engine",
                            info=_engine_info(),
                            interactive=True)
                        tts_engine_note = gr.HTML(
                            _chip("ok", "Active: " + pipeline.TTS_ENGINES[
                                pipeline.get_tts_engine()]["label"]))
                        _lic = _engine_licence()
                        if _lic:
                            gr.Markdown(_lic)
                        gr.Markdown("---")
                        gr.Markdown("### 📥 Source material")
                        media_in = gr.File(label="🎬 Audio / Video master",
                                           file_types=["video", "audio"])
                        with gr.Row():
                            srt_orig_in = gr.File(label="🌐 Original SRT",
                                                  file_types=[".srt"])
                            srt_trans_in = gr.File(label="🌍 Translated SRT",
                                                   file_types=[".srt"])
                        with gr.Row():
                            # language_choices() returns (label, value) pairs in
                            # the exact order Gradio unpacks them. Never reorder
                            # to (value, label): preprocess() would then reject
                            # the code the browser posts and Tab 1 shows
                            # "Error · Value: … is not in the list of choices".
                            lang_orig_in = gr.Dropdown(
                                choices=pipeline.language_choices(
                                    include_auto=True),
                                value="auto", label="🎙 Original language",
                                info="Language of the source audio — used by "
                                     "the emotion/ASR scan",
                                interactive=True, allow_custom_value=False)
                            lang_target_in = gr.Dropdown(
                                choices=pipeline.language_choices(),
                                value="en", label="🌍 Target / dub language",
                                info="Limited to what the active TTS engine "
                                     "can speak — the cloned voices speak this",
                                interactive=True, allow_custom_value=False)
                        spk_hint_in = gr.Dropdown(
                            choices=["Auto", "1", "2", "3", "4", "5",
                                     "6", "7", "8", "9", "10"],
                            value="Auto", label="👥 Expected speakers",
                            info="Hint only - helps merge stray voice clusters; "
                                 "not a strict limit",
                            interactive=True, allow_custom_value=False)
                        translit_in = gr.Checkbox(
                            value=True,
                            label="🔤 Transliterate Roman text to native script (e.g. Hindi)",
                            info="Auto-converts Latin-script translated lines (Roman Hindi) into Devanagari so TTS can pronounce them.")
                        # Both builds honour HUGGING_FACE_HUB_TOKEN, but only the
                        # legacy one has a gated repo to spend it on — so the
                        # label names the mechanism, not Pyannote, and does not
                        # promise a token is required when it is not.
                        with gr.Accordion("🔑 Advanced — Hugging Face token "
                                          "(optional, gated models only)",
                                          open=False):
                            hf_token_in = gr.Textbox(
                                type="password",
                                label="HF access token",
                                placeholder="hf_xxxxxxxxxxxx  (or set HUGGING_FACE_HUB_TOKEN)",
                                info="Only used when the active build pulls a "
                                     "gated repo. Silero-VAD, CAM++ and "
                                     "emotion2vec are open — no token, no "
                                     "terms-acceptance step.")
                            clear_cloud_btn = gr.Button("🧹 Clear cloud storage",
                                                        variant="secondary",
                                                        elem_classes=["icon-btn"])
                            clear_cloud_msg = gr.HTML("")
                            gr.Markdown("---")
                            gr.Markdown("#### ☁️ Model persistence "
                                        "(optional)")
                            persist_on = gr.Checkbox(
                                value=False,
                                label="Enable model persistence "
                                      "(reuse models across sessions)")
                            persist_backend = gr.Radio(
                                choices=[("Hugging Face Hub", "hf"),
                                         ("Google Drive", "drive")],
                                value="hf", label="Backend", interactive=True)
                            persist_token = gr.Textbox(
                                type="password", label="HF token (for HF Hub)",
                                placeholder="hf_xxxxxxxxxxxx",
                                info="Required for HF Hub upload/restore.")
                            upload_btn = gr.Button("⬆️ Start upload (background)",
                                                   variant="primary")
                            upload_progress = gr.HTML("")
                            upload_msg = gr.HTML("")
                            diag_in = gr.Checkbox(
                                value=False,
                                label="🩺 Diagnostic Mode — collect ALL errors "
                                      "in one run (skip dependent steps, print "
                                      "full report at end)",
                                info="ON = pipeline runs to the finish line and "
                                     "shows every error at once. OFF = halt at "
                                     "the first error (production mode).")
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
                        gr.Button("Copy console", variant="secondary").click(
                            fn=None, inputs=[import_log], js=_COPY_JS)
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
                        gr.Markdown("Splits the speech into segments, clusters "
                                    "each voice, mines every speaker's cleanest "
                                    "5–10 s clone prompt, tags each line with its "
                                    "emotion, and merges your translated SRT onto "
                                    "the grid. It runs on whichever build the "
                                    "Tab 1 switcher has active — that build's "
                                    "toolchain is named in the Tab 1 chip.")
                        match_btn = gr.Button(value="", icon=icon("brain"),
                                              variant="primary",
                                              elem_classes=["icon-btn"])
                        gr.HTML('<div class="btn-caption">🧬 Match speakers & emotions, '
                                'then assemble the translated script (Steps 3–5)</div>')
                        gr.Markdown("ℹ️ Requires Tab 1 to be complete. Credentials "
                                    "from Tab 1 ▸ Advanced are reused here only "
                                    "when the active build needs them.")
                    with gr.Column(scale=6, elem_classes=["glass", "pad"]):
                        match_status = gr.HTML(_chip("idle", "Not started"))
                        match_log = gr.Textbox(label="Pipeline console",
                                               lines=15, max_lines=26,
                                               interactive=False,
                                               **_COPY_KW,
                                               elem_classes=["console"])
                        gr.Button("Copy console", variant="secondary").click(
                            fn=None, inputs=[match_log], js=_COPY_JS)
                gr.Markdown("### 🧬 Speaker identification")
                gr.Markdown("Click a speaker below to give it a custom name "
                            "and pick its gender - saves instantly and applies "
                            "everywhere.")
                spk_table = gr.DataFrame(
                    headers=["Speaker", "Gender", "Pitch", "Detected from",
                             "Lines", "First", "Last", "Total"],
                    interactive=False, elem_classes=["glass"])
                with gr.Row():
                    spk_pick = gr.Dropdown(label="Speaker",
                                           interactive=True,
                                           allow_custom_value=False)
                    spk_name = gr.Textbox(label="Custom name",
                                          placeholder="e.g. Narrator",
                                          interactive=True)
                    spk_gender = gr.Dropdown(choices=["", "male", "female"],
                                             value="", label="Gender",
                                             interactive=True,
                                             allow_custom_value=False)
                    spk_save_btn = gr.Button("Save", variant="primary")
                spk_save_msg = gr.HTML("")
                gr.Markdown("### 🎚 Voice audition & prosody")
                gr.Markdown("Gender is **auto-detected** per speaker from their "
                            "pitch (Step 3) — the table above shows the measured "
                            "F0 and where it came from, and your override always "
                            "wins. Use **Preview** to *hear* a voice / pitch / "
                            "blend choice before committing to a full render.")
                with gr.Row():
                    voz_speaker = gr.Dropdown(label="Speaker", interactive=True,
                                              allow_custom_value=False)
                    voz_voice = gr.Textbox(
                        label="Voice (blank = auto-cast for that speaker)",
                        placeholder="hi-IN-MadhurNeural · hm_omega",
                        interactive=True)
                    voz_engine = gr.Dropdown(
                        choices=list(pipeline.TTS_ENGINES.keys()),
                        value=pipeline.get_tts_engine(),
                        label="Engine to audition", interactive=True,
                        allow_custom_value=False)
                with gr.Row():
                    with gr.Column(elem_classes=["glass", "pad"]):
                        gr.Markdown("**Edge-TTS prosody** · pitch needs "
                                    "edge-tts ≥ 6.1")
                        voz_pitch = gr.Slider(minimum=-100, maximum=100,
                                              value=int(_prosody.get("pitch", 0)),
                                              step=1, label="Pitch (Hz)")
                        voz_rate = gr.Slider(minimum=-50, maximum=100,
                                             value=int(_prosody.get("rate", 0)),
                                             step=1, label="Rate (%)")
                        voz_volume = gr.Slider(minimum=-100, maximum=100,
                                               value=int(_prosody.get("volume", 0)),
                                               step=1, label="Volume (%)")
                    with gr.Column(elem_classes=["glass", "pad"]):
                        gr.Markdown("**Kokoro speed & blending** · same-gender "
                                    "partners only, weight clamped 0.30–0.70")
                        gr.Markdown("_Above 4 speakers Kokoro must reuse a Hindi "
                                    "voice, so each shared speaker is "
                                    "auto-blended with a **distinct** same-gender "
                                    "mix (Phase D). Setting a partner below "
                                    "disables that and applies your blend to "
                                    "everyone._")
                        voz_speed = gr.Slider(minimum=0.5, maximum=1.5,
                                              value=float(_kokoro_speed),
                                              step=0.05, label="Speed ×")
                        voz_partner = gr.Dropdown(
                            choices=["", "hf_alpha", "hf_beta", "hm_omega",
                                     "hm_psi"],
                            value=str(_kokoro_blend.get("partner", "") or ""),
                            label="Blend partner (empty = no blend)",
                            interactive=True, allow_custom_value=False)
                        voz_weight = gr.Slider(
                            minimum=0.30, maximum=0.70,
                            value=float(_kokoro_blend.get("weight", 0.5)),
                            step=0.05, label="Blend weight")
                with gr.Row():
                    voz_preview_btn = gr.Button("🔊 Preview voice",
                                                variant="primary")
                    voz_save_btn = gr.Button("💾 Save voice settings",
                                             variant="secondary")
                voz_preview = gr.Audio(label="🎧 Audition", interactive=False,
                                       elem_classes=["glass", "pad"])
                voz_msg = gr.HTML("")
                gr.Markdown("### ⚙️ Fix a line's speaker")
                gr.Markdown("Diarization got a line wrong? Pick its row # and "
                            "the correct speaker.")
                with gr.Row():
                    fix_row = gr.Dropdown(label="Row #", interactive=True,
                                          allow_custom_value=False)
                    fix_spk = gr.Dropdown(label="Correct speaker",
                                          interactive=True,
                                          allow_custom_value=False)
                    fix_btn = gr.Button("Apply", variant="primary")
                fix_msg = gr.HTML("")
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

                # ── CSV side-cars (what Steps 3 & 4 actually export) ─────────
                # Both tables read the FILE, not the JSON the UI also renders
                # from, so what you see here is byte-for-byte what lands on disk
                # for spreadsheet review or a re-run diff.
                gr.Markdown("### 📊 Diarization & emotion side-cars (CSV)")
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["glass", "pad"]):
                        gr.Markdown("`outputs/diarization_cues.csv` — one row per "
                                    "detected turn.")
                        diar_tbl = gr.Dataframe(
                            headers=["Speaker ID", "Start_Time", "End_Time", "Text"],
                            interactive=False, elem_classes=["glass"])
                    with gr.Column(scale=5, elem_classes=["glass", "pad"]):
                        gr.Markdown("`outputs/emotion_grid.csv` — one row per "
                                    "assembled line.")
                        emo_tbl = gr.Dataframe(
                            headers=["Speaker ID", "Start_Time", "End_Time",
                                     "Emotion"],
                            interactive=False, elem_classes=["glass"])

            # ════════════════ TAB 3 · RENDERING ENGINE ══════════════════════
            with gr.Tab("🚀 Rendering Engine") as tab3:
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
                        gr.Button("Copy console", variant="secondary").click(
                            fn=None, inputs=[render_log], js=_COPY_JS)
                final_audio = gr.Audio(label="🎧 Final master mix",
                                       elem_classes=["glass", "pad"])
                gr.Markdown("#### 🔉 Original audio")
                original_audio = gr.Audio(
                    label="🔈 The source as uploaded — before dubbing",
                    elem_classes=["glass", "pad"])
                final_video = gr.Video(label="🎬 Final dubbed video "
                                             "(when the source was a video)",
                                       elem_classes=["glass", "pad"])

        # ── event wiring ────────────────────────────────────────────────────
        analyze_btn.click(
            fn=_run_analysis,
            inputs=[media_in, srt_orig_in, srt_trans_in, hf_token_in,
                    lang_orig_in, lang_target_in, translit_in, diag_in],
            outputs=[import_log, vocals_preview, music_preview, analysis_status],
        )
        match_btn.click(
            fn=_run_matching,
            # ⚠ positional order MUST match _run_matching(token, srt_o, srt_t,
            # num_speakers, diagnostic) — a bool in the speaker-count slot makes
            # int(num_speakers) throw, so the stage dies before it starts.
            inputs=[hf_token_in, srt_orig_in, srt_trans_in, spk_hint_in, diag_in],
            outputs=[match_log, script_df, emotion_log_tb, clone_files, match_status],
        ).then(
            fn=_refresh_speakers,
            inputs=None,
            outputs=[spk_table, spk_pick, fix_row, fix_spk, voz_speaker],
        ).then(
            fn=_refresh_tables,
            inputs=None,
            outputs=[diar_tbl, emo_tbl],
        )
        auto_btn.click(
            fn=_run_full_auto,
            # ⚠ positional order MUST match _run_full_auto(media, srt_o, srt_t,
            # token, lang_o, lang_t, num_speakers, translit, diagnostic): the
            # speaker hint comes BEFORE transliterate, diagnostic comes LAST.
            inputs=[media_in, srt_orig_in, srt_trans_in, hf_token_in,
                    lang_orig_in, lang_target_in, spk_hint_in, translit_in,
                    diag_in],
            outputs=[render_log, final_audio, final_video, render_status,
                     match_status, analysis_status, original_audio],
        ).then(
            fn=_refresh_tables,
            inputs=None,
            outputs=[diar_tbl, emo_tbl],
        )
        render_btn.click(
            fn=_run_rendering,
            inputs=[diag_in],
            outputs=[render_log, final_audio, final_video, render_status,
                     original_audio],
        )
        # The source track exists from Step 1, so fill the player as soon as the
        # tab is opened instead of making people render first. Guarded because
        # it is a convenience — the render and auto-pilot callbacks above already
        # populate it, so losing this never leaves the player unreachable.
        if hasattr(tab3, "select"):
            tab3.select(fn=_original_audio, inputs=None,
                        outputs=[original_audio])
        preflight_btn.click(
            fn=_run_readiness,
            inputs=None,
            outputs=[ready_html],
        )

        # speaker-ID editor + per-line fixer + cloud clear
        spk_save_btn.click(
            fn=_save_profile,
            inputs=[spk_pick, spk_name, spk_gender],
            outputs=[spk_table, spk_save_msg],
        )
        # pick a speaker → pre-fill the editor (Save becomes "confirm")
        spk_pick.change(
            fn=_spk_prefill,
            inputs=[spk_pick],
            outputs=[spk_name, spk_gender],
        )
        # voice audition + persisted prosody / blend settings
        voz_preview_btn.click(
            fn=_preview_voice_app,
            inputs=[voz_engine, spk_pick, voz_voice, voz_pitch, voz_rate,
                    voz_volume, voz_speed, voz_partner, voz_weight],
            outputs=[voz_preview, voz_msg],
        )
        voz_save_btn.click(
            fn=_save_voice_settings,
            inputs=[voz_pitch, voz_rate, voz_volume, voz_speed, voz_partner,
                    voz_weight],
            outputs=[voz_msg],
        )
        fix_btn.click(
            fn=_fix_line,
            inputs=[fix_row, fix_spk],
            outputs=[script_df, fix_msg],
        )
        upload_btn.click(
            fn=_start_upload,
            inputs=[persist_on, persist_backend, persist_token],
            outputs=[upload_progress, upload_msg, analysis_status],
        )
        persist_on.change(fn=_toggle_persist, inputs=[persist_on],
                          outputs=[analysis_status])
        tts_engine_in.change(fn=_switch_tts_engine,
                             inputs=[tts_engine_in, lang_orig_in,
                                     lang_target_in],
                             outputs=[tts_engine_note, lang_orig_in,
                                      lang_target_in])
        # Pipeline switcher — only built when _PIPELINE_VARIANTS has >1 entry,
        # so this reference is unreachable (and therefore never NameErrors)
        # on builds without it, e.g. Colab_Runner.ipynb.
        if len(_PIPELINE_VARIANTS) > 1:
            pipe_in.change(fn=_bind_pipeline,
                           inputs=[pipe_in, lang_orig_in, lang_target_in],
                           outputs=[pipe_in, pipe_note, tts_engine_note,
                                    tts_engine_in, lang_orig_in, lang_target_in])
        clear_cloud_btn.click(fn=_clear_cloud, inputs=None,
                              outputs=[clear_cloud_msg])
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
