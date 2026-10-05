#   AUTOMATIC DUBBING ENGINE — pipeline2.py · AutoDub Studio 2.0
#   Core file-routing layer · sequential GPU execution · OOM defense
#   Target runtime: Google Colab Free T4 (16 GB VRAM) — CPU-safe fallbacks
#
#   ╔═════════════════════════════════════════════════════════════════════════╗
#   ║  DEDICATED COPY — AutoDub Studio 2.0 runs THIS file, never pipeline.py.  ║
#   ╚═════════════════════════════════════════════════════════════════════════╝
#
#   It is a byte-for-byte duplicate of pipeline.py apart from this header and
#   the two identity constants (PIPELINE_VARIANT, ENGINE_ALLOWLIST), so
#       diff pipeline.py pipeline2.py
#   shows exactly what makes 2.0 2.0 — and nothing else. Because the two are
#   separate files, a fix made HERE does not reach the original build and a fix
#   made to pipeline.py does not reach this one: port changes deliberately.
#
#   Selection: AutoDub Studio 2.0.ipynb sets AUTODUB_PIPELINE=pipeline2 before
#   launching app.py, which then binds this module. Colab_Runner.ipynb never
#   sets it and keeps driving pipeline.py.
#
#   Chatterbox-only: ENGINE_ALLOWLIST below bakes the registry down to
#   Chatterbox, so this build needs no env var to stay single-engine.
# ═══════════════════════════════════════════════════════════════════════════
#
#   EXECUTION MAP
#   ────────────
#     Step 1 · extract_audio        FFmpeg / MoviePy ......... CPU
#     Step 2 · separate_vocals      Demucs v4 htdemucs ....... GPU → flush
#     Step 3 · diarize + mine       silero-vad + CAM++ ....... GPU → flush
#     Step 4 · emotion scan         emotion2vec_plus_large ... GPU → flush
#     Step 5 · assemble script      pysrt merge .............. CPU
#
#   DESIGN RULE — exactly ONE heavy model lives on the GPU at any moment.
#   After each model finishes, `clear_gpu_cache()` releases the model,
#   garbage-collects, and empties the CUDA cache before the next stage
#   boots. This is the defense against Colab T4 Out-Of-Memory kills.
#
#   All heavy libraries (torch / demucs / silero-vad / funasr) are imported
#   LAZILY inside their step functions, so:
#     · importing this module is instant and dependency-light,
#     · the UI launches even on machines without torch,
#     · no GPU memory is touched until a step actually runs.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  Paths & runtime constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_WAV = OUTPUTS_DIR / "step1_audio.wav"          # Step 1 artifact
VOCALS_WAV = OUTPUTS_DIR / "vocals.wav"              # Step 2 artifact
MUSIC_WAV = OUTPUTS_DIR / "music.wav"                # Step 2 artifact
DIARIZATION_JSON = OUTPUTS_DIR / "diarization_map.json"   # Step 3 artifact
EMOTION_LOG_TXT = OUTPUTS_DIR / "emotion_log.txt"         # Step 4 artifact
EMOTION_GRID_JSON = OUTPUTS_DIR / "emotion_grid.json"     # Step 4 artifact
FINAL_SCRIPT_JSON = OUTPUTS_DIR / "final_script.json"     # Step 5 artifact
SPEAKER_SCRIPTS_JSON = OUTPUTS_DIR / "speaker_scripts.json"   # Step 6 artifact
TTS_REPORT_JSON = OUTPUTS_DIR / "tts_report.json"             # Step 7 artifact
FINAL_MIX_WAV = OUTPUTS_DIR / "final_mix.wav"                 # Step 8 artifact
FINAL_VIDEO_MP4 = OUTPUTS_DIR / "final_dubbed.mp4"            # Step 8 artifact
PROMPT_TRANSCRIPTS_JSON = OUTPUTS_DIR / "clone_prompt_transcripts.json"
SPEAKER_PROFILES_JSON = OUTPUTS_DIR / "speaker_profiles.json"   # names/gender
STATE_JSON = OUTPUTS_DIR / "state.json"                   # pipeline state
# ── CSV side-cars (AutoDub Studio 2.0 · review + spreadsheet export) ─────────
# Written ALONGSIDE the JSON artifacts, never INSTEAD of them: the JSON stays
# the machine contract Steps 5–7 read, the CSV is for humans and Excel.
SPEAKER_TURNS_CSV = OUTPUTS_DIR / "speaker_turns.csv"     # Step 3 · VAD+cluster turns
DIAR_CUES_CSV = OUTPUTS_DIR / "diarization_cues.csv"      # Step 3 · per-cue speaker
EMOTION_GRID_CSV = OUTPUTS_DIR / "emotion_grid.csv"       # Step 4 · speaker/emotion
LINE_FIT_CSV = OUTPUTS_DIR / "line_fit_report.csv"        # Step 7 · per-line fit

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg",
              ".mpeg", ".ts", ".flv", ".wmv", ".3gp"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}

# The 7 canonical paralinguistic emotions (spec · Phase 3, Step 4)
EMOTIONS = ("happy", "sad", "angry", "surprised", "neutral", "fearful", "disgusted")

# SenseVoice rich-transcription tag → canonical emotion
SENSEVOICE_EMO_MAP = {
    "HAPPY": "happy",
    "EXCITED": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "EMO_UNKNOWN": "neutral",
    "SURPRISED": "surprised",
    "FEARFUL": "fearful",
    "FEARSOME": "fearful",     # alias seen in some FunASR builds
    "DISGUSTED": "disgusted",
}

# emotion2vec+ `labels` entry → canonical emotion.  emotion2vec_plus_large emits
# NINE classes and returns them as '生气/angry' (Chinese/short-English), so every
# key is normalised through `_normalise_emotion` — which lowercases, strips the
# '中文/' prefix and trims — before this table is consulted. That means the
# short-English, long-English and bare-Chinese spellings all land here.
#
#   angry · disgusted · fearful · happy · neutral · sad · surprised   → 1:1
#   other · unknown · <empty>                                        → neutral
#
# The last group is why there is no 'Gasp'/'event' class to map: emotion2vec+
# has no acoustic-event head at all, so non-speech lands in 'other'. Adding an
# AED model for it would cost a fourth sequential model load and change nothing
# downstream — a line that is only a gasp carries no words to synthesise.
EMOTION2VEC_MAP = {
    # ── the 7 canonical classes ──
    "angry": "angry",
    "disgusted": "disgusted",
    "fearful": "fearful",
    "fear": "fearful",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprised": "surprised",
    "surprise": "surprised",
    # ── Chinese spellings seen across emotion2vec builds ──
    "生气": "angry", "愤怒": "angry",
    "厌恶": "disgusted", "反感": "disgusted",
    "恐惧": "fearful", "害怕": "fearful",
    "开心": "happy", "高兴": "happy", "快乐": "happy",
    "中立": "neutral", "中性": "neutral",
    "难过": "sad", "悲伤": "sad", "伤心": "sad",
    "吃惊": "surprised", "惊讶": "surprised",
    # ── catch-alls: no emotion signal → the neutral default ──
    "other": "neutral", "其他": "neutral",
    "unknown": "neutral", "未知": "neutral", "": "neutral",
}

# emotion2vec+ returns labels such as '生气/angry' or '开心/happy'.
_EMO_LABEL_SPLIT = re.compile(r"[\s/｜|]")


def _normalise_emotion(label: str) -> str:
    """Map any emotion2vec+ label spelling onto a canonical emotion.

    Accepts '生气/angry', 'angry', '开心/happy', ''. Keeps the LAST non-empty
    segment (the English tail in every shipped vocabulary) and falls back to
    the first, so a label that is Chinese-only still resolves. Unknown strings
    degrade to 'neutral' rather than crashing the grid — Step 5 only ever needs
    a member of EMOTIONS.
    """
    parts = [p for p in _EMO_LABEL_SPLIT.split(str(label or "").strip()) if p]
    for cand in (parts[-1].lower() if parts else "",
                 parts[0].lower() if parts else "",
                 str(label or "").strip()):
        if cand in EMOTION2VEC_MAP:
            return EMOTION2VEC_MAP[cand]
    return "neutral"


# SenseVoice language tag → readable name (bonus metadata for the grid)
LANG_MAP = {"zh": "Chinese", "en": "English", "yue": "Cantonese",
            "ja": "Japanese", "ko": "Korean"}

# Dubbing languages offered in the UI · (code, label) — the curated fallback
DUBBING_LANGUAGES = [
    ("auto", "🌐 Auto-detect"),
    ("en", "English"),
    ("hi", "हिन्दी · Hindi"),
    ("es", "Español · Spanish"),
    ("zh", "中文 · Chinese"),
    ("yue", "粵語 · Cantonese"),
    ("ja", "日本語 · Japanese"),
    ("ko", "한국어 · Korean"),
]

# ── What each engine can actually SAY · (code, label) ────────────────────────
# The target code reaches the synthesiser as `model.generate(text,
# language_id=...)`, and chatterbox's generate() raises ValueError for anything
# outside its own SUPPORTED_LANGUAGES. The curated list above offers "yue"
# (Cantonese), which Chatterbox does not have — so offering it wasn't just
# unhelpful, it guaranteed a failure one line into Step 7. These entries are
# copied from each vendor's published set rather than guessed:
#   chatterbox  chatterbox/mtl_tts.py SUPPORTED_LANGUAGES — all 23, verbatim
#   cosyvoice2/3 FunAudioLLM/CosyVoice2-0.5B model card — "9 languages"
#   kokoro      hexgrad/Kokoro-82M model card — "Langs & Voices: 8 & 54"
#               (lang codes a/b/e/f/h/i/j/p → 7 distinct ISO tags)
# edge and fishs2 publish no machine-readable language set, so they fall back
# to DUBBING_LANGUAGES rather than have us invent one for them.
ENGINE_LANGUAGES: Dict[str, List[Tuple[str, str]]] = {
    "chatterbox": [
        ("ar", "العربية · Arabic"), ("da", "Dansk · Danish"),
        ("de", "Deutsch · German"), ("el", "Ελληνικά · Greek"),
        ("en", "English"), ("es", "Español · Spanish"),
        ("fi", "Suomi · Finnish"), ("fr", "Français · French"),
        ("he", "עברית · Hebrew"), ("hi", "हिन्दी · Hindi"),
        ("it", "Italiano · Italian"), ("ja", "日本語 · Japanese"),
        ("ko", "한국어 · Korean"), ("ms", "Bahasa Melayu · Malay"),
        ("nl", "Nederlands · Dutch"), ("no", "Norsk · Norwegian"),
        ("pl", "Polski · Polish"), ("pt", "Português · Portuguese"),
        ("ru", "Русский · Russian"), ("sv", "Svenska · Swedish"),
        ("sw", "Kiswahili · Swahili"), ("tr", "Türkçe · Turkish"),
        ("zh", "中文 · Chinese"),
    ],
    "cosyvoice2": [
        ("zh", "中文 · Chinese"), ("en", "English"),
        ("ja", "日本語 · Japanese"), ("ko", "한국어 · Korean"),
        ("de", "Deutsch · German"), ("es", "Español · Spanish"),
        ("fr", "Français · French"), ("it", "Italiano · Italian"),
        ("ru", "Русский · Russian"),
    ],
    "cosyvoice3": [
        ("zh", "中文 · Chinese"), ("en", "English"),
        ("ja", "日本語 · Japanese"), ("ko", "한국어 · Korean"),
        ("de", "Deutsch · German"), ("es", "Español · Spanish"),
        ("fr", "Français · French"), ("it", "Italiano · Italian"),
        ("ru", "Русский · Russian"),
    ],
    "kokoro": [
        ("en", "English"), ("es", "Español · Spanish"),
        ("fr", "Français · French"), ("hi", "हिन्दी · Hindi"),
        ("it", "Italiano · Italian"), ("ja", "日本語 · Japanese"),
        ("pt", "Português · Portuguese"),
    ],
}

# CosyVoice-3 friendly instruct hints (consumed by the later TTS phase)
EMOTION_INSTRUCT = {
    "happy": "in a happy, upbeat tone",
    "sad": "in a soft, sorrowful tone",
    "angry": "in an angry, tense tone",
    "surprised": "in a surprised, wide-eyed tone",
    "neutral": "in a calm, neutral tone",
    "fearful": "in a fearful, trembling tone",
    "disgusted": "in a disgusted, repulsed tone",
}

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 / 4 MODEL IDS — AutoDub Studio 2.0 experimental stack
# ═══════════════════════════════════════════════════════════════════════════
#
#  Step 3 (speaker identity) no longer uses Pyannote. The 2.0 stack is:
#    ① silero-vad ............ voice activity  (MIT, ungated, pip or torch.hub)
#    ② CAMPPlus (FunASR) ..... 192-d speaker embeddings, already in the FunASR
#                              stack Step 4 uses → ZERO new heavy dependency
#    ③ sklearn AgglomerativeClustering over cosine distance → speakers
#
#  Why this replaces Pyannote's speaker-diarization-3.1:
#    · size      ~43 MB of GATED weights (3.1 10.9 + segmentation 5.9 +
#                wespeaker 26.6) → ~41 MB (Silero wheel 11.3 + CAM++ 30.4)
#    · gating    two gated HF repos + a valid token → NO token, NO gate, NO
#                huggingface.co/…/'Agree and access repository' checklist
#    · licence   MIT (silero) + Apache-2.0 (CAM++) + the FunASR stack already
#                installed. Neither new repo is gated.
#  NOTE the saving is GATING, not weight: this step is size-neutral. The Step 4
#  swap below is what actually moves the number (emotion2vec+ is ~1 GB larger
#  than SenseVoice) — see docs/PIPELINE_2.0_OVERVIEW.md §6.
#  What is deliberately NOT swapped: gender profiling still uses librosa pYIN
#  (`_estimate_gender`) — it is validated by selftest §12 and needs no model.
DIAR_ENGINE = "silero-vad + campplus-192d + agglomerative"
CAMPPLUS_MODEL = "iic/speech_campplus_sv_en_voxceleb_16k"
# Official FunASR mirror on Hugging Face. ModelScope is the primary route for
# CAM++; this is the automatic fallback when modelscope.cn is unreachable
# (SSL / proxy / offline), which is the usual reason AutoModel fails.
CAMPPLUS_MODEL_HF = "funasr/campplus"
# Shown in Tab 1's live "Pipeline:" chip, which IS re-derived on every switcher
# flip — so the step 3/4 toolchain named on screen always matches the module
# actually driving the app.
STEP34_TOOLS = "Silero-VAD + CAM++ · AgglomerativeClustering · emotion2vec+"
# The legacy stack needed a Hugging Face token for two GATED repositories.
# Nothing here does, which is why app.py must stop promising a token is "reused".
NEEDS_HF_TOKEN = False

# Step 4 (emotion) — emotion2vec_plus_large, FunASR-native, 9 classes.
EMOTION2VEC_MODEL = "iic/emotion2vec_plus_large"

# ── Legacy identifiers · NOT used by this module any more ────────────────────
# Kept defined so an external introspection, a stale pickled state or a diff
# against pipeline2.py still resolves the attribute instead of raising.
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming log — each call returns the FULL transcript so UI generators
#  can render a growing console window.
# ─────────────────────────────────────────────────────────────────────────────

class Log:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, msg: str = "") -> str:
        for ln in (str(msg).splitlines() or [""]):
            self.lines.append(ln)
        return self.render()

    def render(self) -> str:
        return "\n".join(self.lines)


def _noop(_msg: str = "") -> None:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  🧹 MEMORY DEFENSE  (Phase 2 — Colab T4 OOM protection)
# ─────────────────────────────────────────────────────────────────────────────

def clear_gpu_cache(*models) -> str:
    """
    OOM-defense wrapper (spec: 'del model → gc.collect() → empty_cache').

    The owning scope drops its model reference (`model = None`) and passes
    the object(s) here. This function then:
      1. nudges any remaining CUDA tensors off the GPU,
      2. runs the Python garbage collector (gc.collect),
      3. calls torch.cuda.empty_cache() to return VRAM to the driver.

    Returns a one-line human-readable status string for the console.
    """
    released: List[str] = []
    for m in models:
        try:
            released.append(type(m).__name__)
        except Exception:
            released.append("model")
        try:  # best effort — move weights to CPU before releasing
            if "cuda" in str(getattr(m, "device", "")):
                m.cpu()
        except Exception:
            pass
        m = None
    models = ()  # drop our own references too

    swept = gc.collect()

    msg = f"🧹 memory flush · gc swept {swept} object(s)"
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved()
            msg += f" · CUDA cache {before / 2**30:.2f}→{after / 2**30:.2f} GB"
        if released:
            msg += f" · released [{', '.join(released)}]"
    except ImportError:
        msg += " · (torch not loaded — CPU mode)"
    return msg


def _hf_hub_compat() -> None:
    """
    huggingface_hub ≥ 0.25 removed the deprecated `use_auth_token` parameter
    from `hf_hub_download` — but Pyannote 3.1's internal code still passes it.
    Monkey-patch the function to accept (and silently convert) the old kwarg
    into the modern `token` parameter. Idempotent — safe to call before every
    Pyannote import.
    """
    try:
        import huggingface_hub
        from huggingface_hub import hf_hub_download as _orig_hf_hub_download
    except ImportError:
        return

    if getattr(_orig_hf_hub_download, "_autodub_patched", False):
        return  # already patched

    def _patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_hub_download(*args, **kwargs)

    _patched_hf_hub_download._autodub_patched = True
    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    # also patch the re-exported reference some libraries cache
    try:
        import huggingface_hub.file_download as _fd
        _fd.hf_hub_download = _patched_hf_hub_download
    except Exception:
        pass


def _torchaudio_compat() -> None:
    """
    Bleeding-edge torchaudio (≥ 2.9 — Colab's default) removed the deprecated
    `set_audio_backend` / `list_audio_backends` / `get_audio_backend` APIs,
    the `torchaudio.backend` submodule, `torchaudio.info()`, etc.
    Some audio/ML libraries (e.g. pyannote's dependency chain) still call
    these at import time. Install harmless no-op shims + a dummy
    `torchaudio.backend` module so those code paths keep working. Idempotent
    — safe to call before every heavy import.
    """
    import sys
    try:
        import torchaudio
    except ImportError:
        return
    from collections import namedtuple
    from types import ModuleType

    # AudioMetaData NamedTuple — matches torchaudio's original API.
    _AudioMetaData = namedtuple(
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels",
         "bits_per_sample", "encoding"])

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    # ── functional soundfile-backed I/O for APIs torchaudio ≥ 2.9 removed ──
    # CRITICAL: load() must honour frame_offset/num_frames — pyannote's
    # chunked reader asks for exact 10 s windows (160 000 samples @16 kHz).
    # A stub returning the full file (or info() reporting num_frames=0,
    # which corrupts pyannote's window math) crashes the batch with a
    # tensor-size mismatch. These implementations are fully functional.
    def _sf_info(filepath, *a, **k):
        import soundfile as _sf
        i = _sf.info(str(filepath))
        return _AudioMetaData(i.samplerate, i.frames, i.channels,
                              16, i.subtype or "PCM_S")

    def _sf_load(filepath, frame_offset: int = 0, num_frames: int = -1,
                 normalize: bool = True, channels_first: bool = True,
                 *a, **k):
        import numpy as _np
        import soundfile as _sf
        import torch as _t
        start = int(frame_offset or 0)
        stop = start + int(num_frames) if num_frames and num_frames > 0 else None
        data, sr = _sf.read(str(filepath), start=start, stop=stop,
                            dtype="float32", always_2d=True)      # (N, C)
        if channels_first:
            data = data.T                                          # → (C, N)
        return _t.from_numpy(_np.ascontiguousarray(data)), sr

    def _sf_save(filepath, src, sample_rate: int, *a, **k):
        import soundfile as _sf
        t = src.detach().cpu().numpy() if hasattr(src, "detach") else src
        _sf.write(str(filepath), t.T if getattr(t, "ndim", 1) == 2 else t,
                  int(sample_rate))

    if not hasattr(torchaudio, "info"):
        torchaudio.info = _sf_info
    if not hasattr(torchaudio, "load"):
        torchaudio.load = _sf_load
    if not hasattr(torchaudio, "save"):
        torchaudio.save = _sf_save
    # torchcodec-backed native load (torchaudio >= 2.9) crashes when a
    # library passes an already-decoded TENSOR (e.g. CosyVoice's load_wav
    # on prompt speech). Pass tensors through as 16 kHz floats instead.
    if not getattr(torchaudio.load, "_autodub_patched", False):
        _native_load = torchaudio.load

        def _tensor_load(filepath, *a, **k):
            import torch as _t
            if isinstance(filepath, _t.Tensor):
                t = filepath
                if t.dim() > 1:
                    t = t.mean(dim=0, keepdim=True)
                return t.float(), 16000
            return _native_load(filepath, *a, **k)

        _tensor_load._autodub_patched = True
        torchaudio.load = _tensor_load

    # torchaudio ≥ 2.9 removed the `backend` submodule — register dummy
    # modules in sys.modules so `import torchaudio.backend.*` doesn't crash.
    # The parent must have __path__ so Python treats it as a package.
    def _make_dummy(name: str) -> ModuleType:
        m = ModuleType(name)
        m.get_audio_backend = lambda: "soundfile"
        m.set_audio_backend = lambda *a, **k: None
        m.list_audio_backends = lambda: ["soundfile"]
        m.AudioMetaData = _AudioMetaData
        m.load = _sf_load
        m.save = _sf_save
        m.info = _sf_info
        return m

    try:
        if "torchaudio.backend" not in sys.modules:
            _pkg = _make_dummy("torchaudio.backend")
            _pkg.__path__ = []                      # mark as package
            torchaudio.backend = _pkg               # attribute access too
            sys.modules["torchaudio.backend"] = _pkg
        for _sub in ("common", "soundfile_backend", "no_backend", "utils"):
            _name = f"torchaudio.backend.{_sub}"
            if _name not in sys.modules:
                sys.modules[_name] = _make_dummy(_name)
    except Exception:
        pass


def _torch_load_compat() -> None:
    """
    PyTorch 2.6 changed the default of `torch.load(weights_only=...)` from
    False → True. Older model checkpoints (Pyannote, FunASR, Demucs) pickle
    auxiliary objects (torch_version, numpy arrays, …) that the strict loader
    rejects with 'Weights only load failed'.

    Pyannote also passes weights_only=True explicitly in some code paths, so
    we FORCE the permissive mode unconditionally — our model sources are
    trusted (official pyannote/funasr/demucs hubs). Idempotent.
    """
    try:
        import torch
        import functools
    except ImportError:
        return
    orig = torch.load
    if getattr(orig, "_autodub_patched", False):
        return

    @functools.wraps(orig)
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False          # FORCE permissive (trusted sources)
        return orig(*args, **kwargs)

    _patched_load._autodub_patched = True
    torch.load = _patched_load
    # also allowlist the globals the strict loader complains about, in case
    # any code path bypasses our patch (e.g. cached function references)
    try:
        import torch.serialization
        torch.serialization.add_safe_globals(
            [torch.torch_version.TorchVersion])
    except Exception:
        pass


def _numpy2_compat() -> None:
    """
    NumPy 2.0 removed legacy aliases (np.NaN, np.float_, …) that older ML
    libraries in our stack still reference. Reinstall them on the shared
    numpy module so pyannote/funasr keep working on Colab's NumPy 2.x.
    Idempotent — only adds aliases that are actually missing.
    """
    try:
        import numpy as np
    except ImportError:
        return
    aliases = {
        "NaN": np.nan, "NAN": np.nan,
        "Inf": np.inf, "Infinity": np.inf, "PINF": np.inf, "NINF": -np.inf,
        "float_": np.float64, "complex_": np.complex128,
        "string_": np.bytes_, "unicode_": np.str_,
        "bool8": np.bool_, "int0": np.intp, "uint0": np.uintp,
        "object0": np.object_,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def log_memory() -> str:
    """One-line VRAM / device report for the pipeline console."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            alloc = torch.cuda.memory_allocated() / 2**30
            resv = torch.cuda.memory_reserved() / 2**30
            return (f"🖥 VRAM[{props.name}] allocated {alloc:.2f} GB · "
                    f"reserved {resv:.2f} / {props.total_memory / 2**30:.1f} GB")
        return "🖥 CUDA unavailable — running on CPU (slower, but functional)"
    except Exception:
        return "🖥 torch not loaded — CPU mode"


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline state — persists stage completion across tab clicks / reruns
#  so completed GPU stages auto-skip instead of re-burning VRAM & quota.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    done: Dict[str, dict] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    ORDER = ["step1", "step2", "step3", "step4", "step5",
             "step6", "step7", "step8"]

    def is_done(self, step: str) -> bool:
        return step in self.done

    def mark(self, step: str, **info) -> None:
        self.done[step] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                           **{k: str(v) for k, v in info.items()}}

    def invalidate_from(self, step: str) -> List[str]:
        """Cascade-reset `step` and everything after it (e.g. new source file)."""
        if step not in self.ORDER:
            return []
        doomed = self.ORDER[self.ORDER.index(step):]
        removed = [s for s in doomed if s in self.done]
        for s in doomed:
            self.done.pop(s, None)
        return removed

    def save(self) -> None:
        STATE_JSON.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "PipelineState":
        if STATE_JSON.exists():
            try:
                return cls(**json.loads(STATE_JSON.read_text(encoding="utf-8")))
            except Exception:
                pass
        return cls()


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Intersection length of two 1-D intervals."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _merge_intervals(ivs: List[Tuple[float, float]]) -> List[List[float]]:
    """Merge overlapping/adjacent [start, end] intervals."""
    ivs = sorted(ivs)
    out: List[List[float]] = []
    for s, e in ivs:
        if out and s <= out[-1][1] + 0.01:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _wav_duration(path: Path) -> float:
    import soundfile as sf
    return sf.info(str(path)).duration


def fmt_ts(seconds: float) -> str:
    """Seconds → 'MM:SS.mmm' (hours prepended beyond 1 h) — spec log format."""
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    core = f"{m:02d}:{s:02d}.{ms:03d}"
    return f"{h:02d}:{core}" if h else core


def load_srt(path) -> List[dict]:
    """
    STEP 5 helper (pysrt) — parse an SRT into [{'index','start','end','text'}]
    with start/end as float seconds. Tries common encodings gracefully.
    """
    import pysrt
    subs = None
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            subs = pysrt.open(str(path), encoding=enc)
            break
        except Exception as e:  # pragma: no cover
            last_err = e
    if subs is None:
        raise RuntimeError(f"Could not read SRT file '{path}' ({last_err})")
    cues = []
    for i, s in enumerate(subs):
        cues.append({
            "index": i + 1,
            "start": s.start.ordinal / 1000.0,   # SubRipTime.ordinal = milliseconds
            "end": s.end.ordinal / 1000.0,
            "text": s.text.replace("\r", " ").replace("\n", " ").strip(),
        })
    return cues



# ---------------------------------------------------------------------------
#  Speaker profiles + cloud-storage helpers (Tab 2 editing / persistence)
# ---------------------------------------------------------------------------

def _persist_config() -> dict:
    """Cloud-persistence config written by Colab Cell 0 (mode/hf_token)."""
    p = Path("/content/autodub_persist.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"mode": "none"}


def load_speaker_profiles() -> Dict[str, dict]:
    """speaker -> {name, gender} custom edits (Tab 2 identification table)."""
    if SPEAKER_PROFILES_JSON.exists():
        try:
            return json.loads(SPEAKER_PROFILES_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_speaker_profile(speaker: str, name: str = "", gender: str = "") -> dict:
    profiles = load_speaker_profiles()
    prof = profiles.setdefault(speaker, {})
    if name:
        prof["name"] = name.strip()
    if gender:
        prof["gender"] = gender.strip().lower()
    SPEAKER_PROFILES_JSON.write_text(json.dumps(profiles, indent=2,
                                                ensure_ascii=False),
                                     encoding="utf-8")
    return profiles


def display_name(speaker: str) -> str:
    prof = load_speaker_profiles().get(speaker, {})
    return prof.get("name") or speaker


def speaker_overview() -> List[dict]:
    """Per-speaker stats for the Tab 2 identification table.

    `gender` is the EFFECTIVE value actually used for casting: the user's Tab 2
    override when set, otherwise the acoustic estimate from Step 3. The raw
    inputs stay visible via `auto_gender`, `gender_override` and
    `gender_source`, so the table can show WHY a voice was chosen.
    """
    if not DIARIZATION_JSON.exists():
        return []
    diar = json.loads(DIARIZATION_JSON.read_text(encoding="utf-8"))
    profiles = load_speaker_profiles()
    rows = []
    for spk in sorted(diar.get("speakers", {})):
        cues = [c for c in diar.get("cues", []) if c.get("speaker") == spk]
        info = diar["speakers"][spk] or {}
        prof = profiles.get(spk, {})
        override = str(prof.get("gender", "")).strip().lower()
        auto = str(info.get("gender", "")).strip().lower()
        rows.append({
            "speaker": spk,
            "name": prof.get("name", ""),
            "gender": override or auto,
            "auto_gender": auto,
            "gender_override": override,
            "gender_source": voice_cast_reason(spk, info, profiles),
            "f0_hz": info.get("f0_hz", 0.0),
            "gender_confidence": info.get("confidence", 0.0),
            "lines": len(cues),
            "first": fmt_ts(min((c["start"] for c in cues), default=0.0)),
            "last": fmt_ts(max((c["end"] for c in cues), default=0.0)),
            "total_s": round(sum(c["end"] - c["start"] for c in cues), 1),
        })
    return rows


def set_row_speaker(index: int, speaker: str) -> bool:
    """Per-line speaker fixer: rewrite the speaker of one consolidated row."""
    if not FINAL_SCRIPT_JSON.exists():
        return False
    rows = json.loads(FINAL_SCRIPT_JSON.read_text(encoding="utf-8"))
    hit = False
    for r in rows:
        if r.get("index") == index:
            r["speaker"] = speaker
            hit = True
    if hit:
        FINAL_SCRIPT_JSON.write_text(json.dumps(rows, indent=2,
                                                ensure_ascii=False),
                                     encoding="utf-8")
    return hit


def clear_cloud_storage(include_models: bool = True) -> str:
    """Wipe cached models from the configured cloud backend (Drive / HF).
    Job artifacts are never stored in the cloud (auto-cleanup), so this
    only frees model-cache space."""
    import shutil as _sh
    cfg = _persist_config()
    mode = cfg.get("mode", "none")
    if mode == "drive":
        base = Path("/content/drive/MyDrive/AutoDub_Studio")
        if not base.exists():
            return "Drive not mounted - nothing to clear."
        freed = []
        if include_models:
            mc = base / "model_cache"
            if mc.exists():
                _sh.rmtree(mc, ignore_errors=True)
                freed.append("Drive model_cache")
        return "Cleared: " + (", ".join(freed) if freed else "nothing found")
    if mode == "hf":
        repo, token = cfg.get("hf_repo", ""), cfg.get("hf_token", "")
        if not (repo and token):
            return "HF backend not configured - nothing to clear."
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            files = api.list_repo_files(repo_id=repo, repo_type="dataset")
            if include_models:
                for f in files:
                    try:
                        api.delete_file(repo_id=repo, repo_type="dataset",
                                        path_in_repo=f)
                    except Exception:
                        pass
                return f"Cleared: HF dataset {repo} ({len(files)} file(s))"
            return "HF dataset kept (include_models=False)."
        except Exception as e:
            return f"HF cleanup failed: {str(e)[:160]}"
    return "Persistence is OFF - no cloud storage in use."


def _cache_roots():
    import os
    home = os.path.expanduser("~")
    ms = Path(os.environ.get("MODELSCOPE_CACHE", home + "/.cache/modelscope"))
    hf = Path(os.environ.get("HF_HOME", home + "/.cache/huggingface"))
    return ms, hf


def _copy_tree(src, dst) -> int:
    import shutil as _sh
    n = 0
    src = Path(src)
    if not src.exists():
        return 0
    for f in src.rglob("*"):
        if f.is_file() and f.suffix != ".lock" and "tmp" not in f.name:
            out = Path(dst) / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                _sh.copy2(str(f), str(out))
                n += 1
    return n


def _hf_auth(tok: str) -> None:
    """Make `tok` the ambient HF credential for THIS process.

    `snapshot_download` / `from_pretrained` default to `token=None`, meaning
    "use the ambient credential" - the HF_TOKEN env var or the file left by
    `huggingface-cli login`. A fresh Colab VM has NEITHER, so any call that
    forgets to pass token= goes out unauthenticated; on a PRIVATE repo that
    returns 401, which downstream code then misreads as "empty cache".
    """
    if not tok:
        return
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    try:
        from huggingface_hub import login
        try:
            login(token=tok, add_to_git_credential=False)
        except TypeError:                # older hub lacking that kwarg
            login(tok)
    except Exception:
        pass


def restore_model_cache(hf_token: str = "", hf_repo: str = "") -> str:
    """Reuse a prior model cache (HF Hub if token, else Google Drive).

    Never raises - returns a status string that distinguishes an ACCESS
    failure from a genuinely EMPTY repo, because those need different fixes.
    """
    ms, hf = _cache_roots()
    ms.mkdir(parents=True, exist_ok=True)
    hf.mkdir(parents=True, exist_ok=True)
    stage = Path("/content/autodub_restore")
    cfg = _persist_config()
    tok = (hf_token or "").strip() or str(cfg.get("hf_token", "")).strip()

    if tok:
        _hf_auth(tok)
        try:
            from huggingface_hub import HfApi, snapshot_download
        except Exception as e:
            return f"huggingface_hub unavailable ({str(e)[:100]})"

        api = HfApi(token=tok)
        try:
            user = api.whoami().get("name", "user")
        except Exception as e:
            return (f"HF token rejected ({str(e)[:110]}) - regenerate it at "
                    f"huggingface.co/settings/tokens")

        candidates: List[str] = []
        for cand in (hf_repo, cfg.get("hf_repo"),
                     f"{user}/autodub-model-cache"):
            cand = (cand or "").strip()
            if cand and cand not in candidates:
                candidates.append(cand)

        problems: List[str] = []
        for repo in candidates:
            try:
                files = api.list_repo_files(repo_id=repo, repo_type="dataset",
                                            token=tok)
            except Exception as e:
                msg = str(e)
                if "401" in msg or "403" in msg or "Unauthorized" in msg:
                    problems.append(f"{repo}: token has NO ACCESS (401)")
                elif "404" in msg or "not found" in msg.lower():
                    problems.append(f"{repo}: no such private dataset")
                else:
                    problems.append(f"{repo}: {msg[:90]}")
                continue

            if not files:
                problems.append(f"{repo}: accessible but EMPTY (0 files)")
                continue

            try:
                if stage.exists():
                    import shutil as _sh
                    _sh.rmtree(stage, ignore_errors=True)
                snapshot_download(repo, repo_type="dataset", token=tok,
                                  local_dir=str(stage))
            except Exception as e:
                problems.append(f"{repo}: download failed ({str(e)[:90]})")
                continue

            a = _copy_tree(stage / "modelscope", ms)
            b = _copy_tree(stage / "hf", hf)
            if not (a + b):             # layout fallback: flat stage root
                a = _copy_tree(stage, ms)
            return (f"Restored from HF Hub: {repo} ({a + b} file(s), "
                    f"{len(files)} in repo)")

        detail = " | ".join(problems[:3]) if problems else "no candidate repo"
        return (f"HF Hub has no usable cache - {detail}. "
                f"Models will download normally instead.")

    try:
        drv = Path("/content/drive/MyDrive/AutoDub_Studio/model_cache")
        if not Path("/content/drive/MyDrive").exists():
            from google.colab import drive as _d
            _d.mount("/content/drive")
        if drv.exists():
            a = _copy_tree(drv / "modelscope", ms)
            b = _copy_tree(drv / "hf", hf)
            return f"Restored from Drive: {a + b} file(s)"
        return "No prior cache found - models will download normally."
    except Exception as e:
        return f"No prior cache (Drive unavailable: {str(e)[:100]})"


def upload_model_cache(progress_cb=None, backend: str = "hf",
                       hf_token: str = "", hf_repo: str = "") -> str:
    """Upload cached models for future sessions. models-only (auto-cleanup).
    progress_cb(pct, message) is called at each step when supplied."""
    import shutil as _sh
    ms, hf = _cache_roots()

    def _p(pct, msg):
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

    if backend == "drive":
        try:
            drv = Path("/content/drive/MyDrive/AutoDub_Studio/model_cache")
            if not Path("/content/drive/MyDrive").exists():
                from google.colab import drive as _d
                _d.mount("/content/drive")
            _p(5, "copying models -> Drive ...")
            a = _copy_tree(ms, drv / "modelscope")
            b = _copy_tree(hf, drv / "hf")
            _p(100, f"Drive upload done ({a + b} file(s))")
            return f"Saved to Drive: {a + b} file(s)"
        except Exception as e:
            _p(100, "Drive upload failed")
            return f"Drive upload failed: {str(e)[:150]}"

    # HF Hub (default) - resumable
    tok = (hf_token or "").strip() or _persist_config().get("hf_token", "")
    if not tok:
        _p(100, "HF upload needs a token")
        return "HF upload needs a token (paste it in the persistence panel)."
    _hf_auth(tok)                     # authenticate the whole session
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        repo = (hf_repo or _persist_config().get("hf_repo")
                or f"{api.whoami().get('name', 'user')}/autodub-model-cache")
        _p(3, "creating private repo ...")
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        stage = Path("/root/.cache/autodub_stage")
        _sh.rmtree(stage, ignore_errors=True)
        _p(10, "staging models ...")
        a = _copy_tree(ms, stage / "modelscope")
        b = _copy_tree(hf, stage / "hf")
        total = max(1, a + b)
        _p(35, f"uploading {a + b} file(s) to HF (resumable) ...")
        api.upload_folder(folder_path=str(stage), repo_id=repo,
                          repo_type="dataset",
                          commit_message="AutoDub model cache sync")
        _p(100, f"HF upload done -> {repo}")
        cfg = _persist_config()
        cfg.update({"mode": "hf", "hf_token": tok, "hf_repo": repo,
                    "auto_cleanup": True})
        Path("/content/autodub_persist.json").write_text(
            json.dumps(cfg), encoding="utf-8")
        return f"Uploaded to HF Hub: {repo} ({a + b} file(s))"
    except Exception as e:
        _p(100, "HF upload failed")
        return f"HF upload failed: {str(e)[:180]}"


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 · AUDIO EXTRACTION  (MoviePy / raw FFmpeg bindings)
# ═════════════════════════════════════════════════════════════════════════════

def step1_extract_audio(media_path: str, log: Log, force: bool = False) -> Path:
    """
    Detect whether the upload is a video container; if so pull the raw audio
    track out losslessly (PCM-16 @ 44.1 kHz stereo) via FFmpeg bindings, with
    a MoviePy fallback when the FFmpeg binary is unavailable.
    """
    src = Path(media_path)
    if not src.exists():
        raise FileNotFoundError(f"Uploaded media not found: {src}")

    if AUDIO_WAV.exists() and not force:
        log(f"↩ Step 1 cached ({AUDIO_WAV.name}) — skipping (force to redo).")
        return AUDIO_WAV

    ext = src.suffix.lower()
    kind = "video" if ext in VIDEO_EXTS else ("audio" if ext in AUDIO_EXTS else "unknown")
    log(f"🔎 Container check … .{ext.lstrip('.')} ⇒ {kind}")

    t0 = time.time()
    if _ffmpeg_available():
        log("⚙ FFmpeg binding — demuxing audio (no re-decode of video) …")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(src), "-vn", "-acodec", "pcm_s16le",
               "-ar", "44100", "-ac", "2", str(AUDIO_WAV)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {proc.stderr.strip()[-500:]}")
    else:
        log("⚠ FFmpeg binary not found — engaging MoviePy fallback …")
        try:
            from moviepy.editor import AudioFileClip, VideoFileClip
        except ImportError:  # moviepy ≥ 2.0 moved the imports
            from moviepy import AudioFileClip, VideoFileClip
        if kind == "video":
            clip = VideoFileClip(str(src)).audio
            if clip is None:
                raise RuntimeError("No embedded audio track found in this video.")
        else:
            clip = AudioFileClip(str(src))
        clip.write_audiofile(str(AUDIO_WAV), fps=44100, codec="pcm_s16le", logger=None)
        clip.close()

    dur = _wav_duration(AUDIO_WAV)
    log(f"✅ Step 1 → {AUDIO_WAV.name}  ({dur:.1f}s of audio in {time.time() - t0:.1f}s)")
    return AUDIO_WAV


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 · VOCAL / MUSIC SEPARATION  (Meta Demucs v4)
# ═════════════════════════════════════════════════════════════════════════════

def _demucs_cli_separate(log: Log) -> None:
    """
    Fallback separation via the demucs CLI - used when the installed demucs
    wheel lacks the demucs.api submodule (PyPI demucs==4.0.1 ships WITHOUT
    api.py; it only exists in the GitHub source). Produces vocals.wav +
    no_vocals.wav via --two-stems=vocals, then renames them to the
    pipeline's expected artifacts.
    """
    import sys as _sys
    t0 = time.time()
    exe = shutil.which("demucs")
    cmd = ([exe] if exe else [_sys.executable, "-m", "demucs.separate"])
    cmd += ["--two-stems=vocals", "-n", "htdemucs", "--shifts", "0",
            "-o", str(OUTPUTS_DIR), str(AUDIO_WAV)]
    log("[i] running: " + " ".join(cmd[:8]) + " ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("demucs CLI failed: "
                           + (proc.stderr or proc.stdout or "")[-400:])
    out_dir = OUTPUTS_DIR / "htdemucs" / AUDIO_WAV.stem
    vocals = out_dir / "vocals.wav"
    no_vocals = out_dir / "no_vocals.wav"
    if not vocals.exists() or not no_vocals.exists():
        raise RuntimeError(f"demucs CLI finished but stems not found in {out_dir}")
    shutil.move(str(vocals), str(VOCALS_WAV))
    shutil.move(str(no_vocals), str(MUSIC_WAV))
    shutil.rmtree(out_dir, ignore_errors=True)
    log(f"   CLI separation done in {time.time() - t0:.1f}s")


def step2_separate_vocals(log: Log, force: bool = False) -> Tuple[Path, Path]:
    """
    Load Demucs v4 (htdemucs), split the master audio into a clean vocal
    stem and a background instrumental stem, write both to disk as .wav,
    then IMMEDIATELY tear the model down with clear_gpu_cache().
    Falls back to the demucs CLI when demucs.api is unavailable (the PyPI
    4.0.1 wheel does not ship the api submodule - known upstream gap).
    """
    if VOCALS_WAV.exists() and MUSIC_WAV.exists() and not force:
        log("<- Step 2 cached (vocals.wav + music.wav) - skipping.")
        return VOCALS_WAV, MUSIC_WAV

    try:
        import torch
        import soundfile as sf
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Step 2 needs the ML stack - module '{missing}' is not installed here. "
            "Local CPU fix:  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu  then  "
            "pip install demucs  -  Or run on Google Colab (T4) where the "
            "stack ships preinstalled.") from e

    try:
        from demucs.api import Separator
    except ImportError:
        Separator = None
    if Separator is None:
        log("[i] demucs.api missing (PyPI wheel) - engaging demucs CLI fallback ...")
        _demucs_cli_separate(log)
        dur = _wav_duration(VOCALS_WAV)
        log("   " + clear_gpu_cache())
        log(f"[OK] Step 2 -> vocals.wav + music.wav  ({dur:.1f}s, CLI path)")
        return VOCALS_WAV, MUSIC_WAV

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[GPU] Loading Demucs v4 - model=htdemucs - device={device}")
    log("   " + log_memory())

    t0 = time.time()
    # shifts=0 -> single pass (2x faster, T4-friendly); split=True keeps VRAM flat
    sep = Separator(model="htdemucs", device=device, shifts=0,
                    overlap=0.25, split=True, progress=False)
    log("Separating stems (chunked streaming - OOM-safe) ...")
    _wav, sources = sep.separate_audio_file(str(AUDIO_WAV))
    sr = sep.samplerate  # 44 100 Hz

    vocals = sources["vocals"]
    music = sources["drums"].cpu() + sources["bass"].cpu() + sources["other"].cpu()
    sf.write(str(VOCALS_WAV), vocals.cpu().numpy().T, sr)   # (frames, channels)
    sf.write(str(MUSIC_WAV), music.numpy().T, sr)

    dur = _wav_duration(VOCALS_WAV)
    # full model teardown before anything else loads
    del sources, vocals, music, sep
    log("   " + clear_gpu_cache())
    log(f"[OK] Step 2 -> vocals.wav + music.wav  ({dur:.1f}s in {time.time() - t0:.1f}s)")
    return VOCALS_WAV, MUSIC_WAV


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 · SPEAKER DIARIZATION + CLONE-PROMPT MINING
#            (Silero-VAD → CAMPPlus 192-d → AgglomerativeClustering)
# ═════════════════════════════════════════════════════════════════════════════

def _score_window(mono, sr: int, a: float, b: float,
                  other_ivs: List[Tuple[float, float]]):
    """
    Heuristic cleanliness score for a candidate voice-clone window:
      · speech ratio (reject silence / dead air)
      · envelope stability (LOW variance = clean, steady voice — the spec's
        'signal variance' criterion)
      · loudness in dB (avoid whisper-quiet mud)
      · zero-crossing sanity (reject hiss / electrical noise)
      · purity (no bleed from other speakers)
    """
    import numpy as np
    seg = mono[int(a * sr): int(b * sr)]
    if seg.size < sr:  # < 1 s guard
        return None

    fl = int(0.02 * sr)                       # 20 ms frames
    n = (seg.size // fl) * fl
    if n == 0:
        return None
    frames = seg[:n].reshape(-1, fl)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)

    floor = max(1e-5, 0.08 * rms.max())
    active = rms > floor
    speech_ratio = float(active.mean())
    if speech_ratio < 0.55:                   # mostly silence → skip
        return None

    act = rms[active]
    stability = 1.0 - float(act.std() / (act.mean() + 1e-9))

    rms_db = 20.0 * np.log10(float(np.sqrt((seg ** 2).mean())) + 1e-9)
    loud = float(np.clip((rms_db + 45.0) / 40.0, 0.0, 1.0))   # −45…−5 dB → 0…1

    zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
    z_ok = float(np.clip(1.0 - abs(zcr - 0.10) / 0.15, 0.0, 1.0))

    span = b - a
    contam = sum(_overlap(a, b, oa, ob) for oa, ob in other_ivs) / span
    purity = 1.0 - min(1.0, contam)

    return (speech_ratio
            * (0.45 + 0.55 * max(0.0, stability))
            * (0.35 + 0.65 * loud)
            * (0.60 + 0.40 * purity)
            * (0.55 + 0.45 * z_ok))


def _mine_clone_prompts(log: Log,
                        spk_intervals: Dict[str, List[List[float]]]) -> Dict[str, List[str]]:
    """
    Isolate the TOP-3 cleanest 5–10 s reference segments per speaker from the
    Demucs vocal track (avoids muddy cloning inputs) and save them as
    outputs/speakerX_clone_prompt.wav (16 kHz mono — CosyVoice-ready).
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(VOCALS_WAV), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16_000:
        import torch
        import torchaudio
        mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, 16_000).numpy()
        sr = 16_000

    out: Dict[str, List[str]] = {}
    for cname, ivals in sorted(spk_intervals.items()):
        others = [tuple(iv) for name, ivs in spk_intervals.items()
                  if name != cname for iv in ivs]

        # candidate windows: sliding 5–10 s, 1 s hop, inside merged turns
        scored = []
        for s0, e0 in ivals:
            dur = e0 - s0
            if dur < 3.0:
                continue
            win = min(10.0, max(5.0, dur))
            if dur < win:
                candidates = [(s0, e0)]
            else:
                candidates = [(t, t + win) for t in np.arange(s0, e0 - win + 1e-6, 1.0)]
            for a, b in candidates:
                sc = _score_window(mono, sr, a, b, others)
                if sc is not None:
                    scored.append((sc, float(a), float(b)))

        # greedy top-3, non-overlapping (≥ 0.5 s apart)
        scored.sort(reverse=True, key=lambda x: x[0])
        picked: List[Tuple[float, float, float]] = []
        for sc, a, b in scored:
            if all(_overlap(a, b, pa, pb) < 0.5 for _, pa, pb in picked):
                picked.append((sc, a, b))
            if len(picked) == 3:
                break

        paths: List[str] = []
        for sc, a, b in picked:
            seg = mono[int(a * sr): int(b * sr)]
            fname = f"{cname.lower()}_clone_prompt.wav"
            sf.write(str(OUTPUTS_DIR / fname), seg.astype(np.float32), sr)
            paths.append(str((OUTPUTS_DIR / fname).relative_to(BASE_DIR)))
        out[cname] = paths
        detail = f"best score {picked[0][0]:.3f}" if picked else "⚠ none found"
        log(f"   {cname}: {len(picked)} prompt(s) · {detail}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 PRIMITIVES — Silero-VAD · CAMPPlus embeddings · clustering
# ═════════════════════════════════════════════════════════════════════════════

def _module_device(model) -> str:
    """Best-effort device string ("cpu" / "cuda:0") for a torch module.

    Reads the first parameter, then the first buffer. A lazily-loaded
    TorchScript module can expose neither, and "cpu" is the honest answer
    there: it is where a freshly-built tensor is born.
    """
    for _getter in ("parameters", "buffers"):
        try:
            for _t in getattr(model, _getter)():
                return str(_t.device)
        except Exception:
            pass
    return "cpu"


def _load_silero_vad(device: str, log: Log):
    """Load Silero-VAD (MIT, ~2 MB) → (model, True), else (None, False).

    The pip package is tried first; `torch.hub` is the fallback for a machine
    that has torch but never installed `silero-vad`. Returning the failure
    instead of raising lets the caller degrade to an energy VAD, so a missing
    model can never make Step 3 unrunnable.

    The model needs only `torch`: we always hand it an in-memory float32
    tensor, never a path, so no audio backend (torchcodec/sox/FFmpeg) is needed.

    `device` is accepted for call-site symmetry with the CAM++ loader and is
    deliberately NOT applied to the model — see the CPU note below.
    """
    first = ""
    try:
        from silero_vad import load_silero_vad       # pip install silero-vad
        # ── Silero stays on CPU ON PURPOSE ──────────────────────────────────
        # `load_silero_vad()` returns a TorchScript module whose LSTM state
        # (`_h`, `_c`, `_context`) is allocated INSIDE the script and is not a
        # registered buffer, so moving the module to "cuda" relocates the
        # weights and leaves that state on the CPU. The first forward pass then
        # dies with
        #     RuntimeError: Expected all tensors to be on the same device, but
        #     got weight is on cuda:0, different from other tensors on cpu
        # which dropped Step 3 to the energy VAD on every T4 session. The
        # checkpoint is ~2 MB and a 45 s track is ~0.2 s of CPU work, so the
        # GPU buys nothing here while the mismatch costs the whole Silero path.
        # `_vad_regions` still aligns the waveform with `_module_device()`, so
        # the hub fallback (or a future GPU-capable wrapper) stays correct.
        model = load_silero_vad()
        return model, True
    except Exception as e:
        first = f"{type(e).__name__}: {e}"
    try:
        import torch  # torch.hub fallback — must be imported, not assumed
        model, _utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                       trust_repo=True, onnx=False)
        return model, True
    except Exception as e:
        log(f"   ⚠ Silero-VAD unavailable ({first} | "
            f"{type(e).__name__}: {e})")
        return None, False

def _vad_regions(mono, sr: int, vad_model, log: Log) -> List[List[float]]:
    """Speech regions as merged [[start_s, end_s], …].

    Silero-VAD when available, otherwise a pure-numpy energy VAD — the step
    must still produce a timeline on a machine that has torch but no model.
    """
    # `torch` is imported HERE, not at module scope: this function is the only
    # place in Step 3 that needs it, and a missing module-scope import turned
    # into `NameError: name 'torch' is not defined` at inference time — which
    # silently dropped us to the energy VAD.
    import numpy as np
    import torch
    t0 = time.time()
    log("🎙 Detecting speech regions (VAD) …")
    if vad_model is not None:
        try:
            from silero_vad import get_speech_timestamps
            audio = torch.from_numpy(
                np.ascontiguousarray(mono, dtype=np.float32))
            # The waveform must ride on the MODEL's device: `get_speech_timestamps`
            # slices THIS tensor and hands the slices straight to the forward
            # pass, so CPU audio + cuda weights is a hard RuntimeError rather
            # than a warning. Silero is pinned to CPU in `_load_silero_vad`, but
            # aligning here means the torch.hub fallback — or any future
            # GPU-capable wrapper — cannot reintroduce the mismatch.
            mdev = _module_device(vad_model)
            if str(audio.device) != mdev:
                audio = audio.to(mdev)
            ts = get_speech_timestamps(
                audio,
                vad_model, sampling_rate=int(sr), return_seconds=False,
                min_speech_duration_ms=250, min_silence_duration_ms=100,
                speech_pad_ms=30)
            ivs = [(t["start"] / float(sr), t["end"] / float(sr)) for t in ts]
            log(f"   Silero-VAD: {len(ivs)} speech region(s) "
                f"· {time.time() - t0:.1f}s")
            return _merge_intervals(ivs)
        except Exception as e:
            log(f"   ⚠ Silero inference failed ({type(e).__name__}: {e}) — "
                f"falling back to energy VAD")

    # ── energy fallback · 20 ms frames, floor from the 95th percentile ────────
    fl = max(1, int(0.02 * sr))
    n = (mono.size // fl) * fl
    if n == 0:
        return []
    rms = np.sqrt((mono[:n].reshape(-1, fl) ** 2).mean(axis=1) + 1e-12)
    active = rms > max(1e-4, 0.12 * float(np.percentile(rms, 95)))
    ivs: List[Tuple[float, float]] = []
    run = None
    for i, on in enumerate(active.tolist()):
        if on and run is None:
            run = i
        elif not on and run is not None:
            ivs.append((run * 0.02, i * 0.02))
            run = None
    if run is not None:
        ivs.append((run * 0.02, len(active) * 0.02))
    ivs = [iv for iv in ivs if iv[1] - iv[0] >= 0.25]
    log(f"   energy-VAD fallback: {len(ivs)} speech region(s) "
        f"· {time.time() - t0:.1f}s")
    return _merge_intervals(ivs)


def _funasr_auto(routes: List[tuple], device: str, log: Log, what: str):
    """Build a FunASR `AutoModel` from the first route that works.

    Why this exists instead of one `AutoModel(...)` call:

    FunASR does **not** surface download failures. `download_from_ms` catches
    every exception and downgrades it to a bare `print(...)`, leaving
    `kwargs["model"]` as the raw hub id; `build_model` then asserts
    `<id> is not registered`. The message you actually see therefore names the
    *model* rather than the fault (expired SSL cert, proxy, offline box,
    missing `modelscope`), which is why this helper captures stdout/stderr
    around the call and reports both the assertion and what FunASR printed.
    """
    import contextlib
    import io as _io

    # Imported HERE: `step3`/`step4` only probe for funasr above to raise a
    # step-specific "pip install funasr" message, so this helper cannot rely on
    # a name they imported. Without this line the call dies with
    # `NameError: name 'AutoModel' is not defined` — the same class of bug as
    # the missing `torch` import in `_vad_regions`.
    try:
        from funasr import AutoModel
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"{what} needs FunASR — module '{missing}' is not installed. "
            "Fix:  pip install funasr modelscope") from e

    notes: List[str] = []
    for label, kwargs in routes:
        buf = _io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                model = AutoModel(device=device, disable_update=True,
                                  disable_pbar=True, **kwargs)
            log(f"   ✓ {label} · device={device}")
            return model
        except Exception as e:
            captured = [ln for ln in (buf.getvalue() or "").splitlines() if ln.strip()]
            funasr_said = captured[-1] if captured else "(nothing printed)"
            notes.append(f"   ✗ {label}\n"
                         f"       {type(e).__name__}: {e}\n"
                         f"       funasr printed: {funasr_said}")
            log(f"   ⚠ {label} failed — {type(e).__name__}: {e}")
    raise RuntimeError(
        f"{what} could not be loaded from any hub:\n" + "\n".join(notes)
        + "\n\nCheck network reachability of modelscope.cn and huggingface.co, "
        "or pre-download one of those repos and pass its local directory."
    )


def _load_campplus(device: str, log: Log):
    """CAM++ speaker embeddings, ModelScope first, Hugging Face as fallback.

    ModelScope is primary because it hosts the English VoxCeleb CAM++ we chose;
    HF is the fallback because `funasr/campplus` (FunASR's own mirror, shipping
    both `configuration.json` and `config.yaml`) is usually reachable when
    modelscope.cn is not.
    """
    return _funasr_auto(
        [(f"modelscope · {CAMPPLUS_MODEL}", {"model": CAMPPLUS_MODEL}),
         (f"huggingface · {CAMPPLUS_MODEL_HF}", {"model": CAMPPLUS_MODEL_HF, "hub": "hf"})],
        device, log, "CAM++ speaker embedding")


def _campplus_embed(model, segs, log: Log):
    """L2-normalised CAMPPlus embeddings → (np.ndarray[n, d], kept_indices).

    FunASR returns one `spk_embedding` tensor per call. The DIMENSION is READ
    from the tensor, never assumed: the FunASR docs are explicit that 192 is a
    property of this checkpoint rather than a universal API guarantee. A
    segment that fails or yields nothing is skipped and its index returned, so
    the caller can keep embeddings aligned with their windows.
    """
    import numpy as np
    embs: List = []
    keep: List[int] = []
    for i, seg in enumerate(segs):
        try:
            res = model.generate(
                input=np.ascontiguousarray(seg, dtype=np.float32), batch_size=1)
            vec = res[0].get("spk_embedding") if res else None
        except Exception:
            vec = None
        if vec is None:
            continue
        try:                       # torch.Tensor → numpy
            arr = vec.detach().cpu().numpy().reshape(-1).astype(np.float32)
        except AttributeError:     # already numpy / list
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        nrm = float(np.linalg.norm(arr)) if arr.size else 0.0
        if arr.size and nrm > 1e-9:
            embs.append(arr / nrm)   # unit vector ⇒ cosine == dot product
            keep.append(i)
    dim = embs[0].size if embs else 0
    log(f"   {len(embs)}/{len(segs)} segment embedding(s) · dim={dim}")
    return (np.vstack(embs) if embs else np.zeros((0, 0), np.float32)), keep


def _cluster_speakers(embs, n_hint: int, log: Log):
    """Cluster cosine-normalised embeddings → one integer label per row.

    `n_hint` is Tab 2's 'Expected speakers' hint and becomes `n_clusters` when
    supplied. Without it we sweep the cosine distance threshold instead of
    trusting one hard-coded cut, which reliably over-splits a long recording.
    Returns an int array aligned with `embs`.
    """
    import numpy as np
    n = int(embs.shape[0])
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as e:
        raise RuntimeError(
            "Step 3 clustering needs scikit-learn → pip install scikit-learn"
        ) from e

    hint = int(n_hint or 0)
    if 0 < hint <= n:
        lab = AgglomerativeClustering(n_clusters=hint, metric="cosine",
                                      linkage="average").fit_predict(embs)
        k = hint
    else:
        chosen = None
        for thr in (0.50, 0.60, 0.70, 0.40, 0.80, 0.30, 0.90):
            lab = AgglomerativeClustering(
                n_clusters=None, distance_threshold=float(thr),
                metric="cosine", linkage="average").fit_predict(embs)
            k = len(set(int(x) for x in lab))
            if 2 <= k <= 12:
                chosen = (k, lab)
                break
        if chosen is None:            # nothing separable, or one huge cluster
            chosen = (1, np.zeros(n, dtype=int))
        k, lab = chosen
    log(f"   agglomerative clustering → {k} speaker cluster(s)")
    return np.asarray(lab, dtype=int)


def _write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    """Write a CSV side-car (RFC 4180, via the stdlib `csv` module).

    Deliberately not pandas: the pipeline's contract stays stdlib-only so the
    CSV is produced even in an environment where pandas was never installed.
    """
    import csv
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def step3_diarization(hf_token: Optional[str],
                      original_srt_path: Optional[str],
                      log: Log,
                      force: bool = False,
                      expected_speakers: int = 0) -> dict:
    """
    Speaker identity for AutoDub Studio 2.0: silero-vad → CAMPPlus embeddings →
    agglomerative clustering, cross-referenced with the user's ORIGINAL SRT
    timeline (max-overlap speaker vote per cue), then clone-prompt mining +
    full GPU teardown.

    Signature and every artifact shape are UNCHANGED from the Pyannote version
    on purpose: `run_script_matching`, Step 5 and app.py's Tab 2 all read
    `diarization_map.json` by key, so the engine swap stays invisible to them.
    `hf_token` is still accepted (and now ignored) rather than removed, so the
    orchestrator and the UI keep one call convention across both builds.
    """
    if DIARIZATION_JSON.exists() and not force:
        log("↩ Step 3 cached (diarization_map.json) — skipping.")
        return json.loads(DIARIZATION_JSON.read_text(encoding="utf-8"))

    try:
        import torch
        import torchaudio  # noqa: F401  (resample + torchaudio ≥ 2.9 shims)
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Step 3 needs the ML stack — module '{missing}' is not installed here. "
            "Local CPU fix:  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu  then  "
            "pip install silero-vad funasr modelscope scikit-learn  ·  "
            "Or run on Google Colab (T4).") from e
    _torchaudio_compat()   # shim APIs removed in torchaudio ≥ 2.9
    _numpy2_compat()       # restore np.NaN / np.float_ aliases for NumPy 2.x
    _torch_load_compat()   # restore legacy torch.load default for trusted checkpoints
    # Dependency probe — gives a step-specific install hint. `_funasr_auto`
    # does its own import, but this runs first so the message names Step 3.
    try:
        from funasr import AutoModel  # noqa: F401
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Step 3 needs FunASR for CAMPPlus embeddings — module '{missing}' "
            "is not installed here.  Fix:  pip install funasr modelscope") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # NOTE: no HF token, no gated-repo pre-flight, no `Pipeline.from_pretrained`.
    # Silero-VAD (~2 MB, MIT) and CAMPPlus are both ungated, so the whole
    # "visit huggingface.co/… and click 'Agree and access repository'" class of
    # failure no longer exists. `hf_token` is deliberately unused above.
    t0 = time.time()
    mono, sr = _load_vocals_16k()
    log(f"🎙 Step 3 · {DIAR_ENGINE} · device={device} · {mono.size / sr:.1f}s audio")
    log("   " + log_memory())

    # ── ① voice activity ─────────────────────────────────────────────────────
    vad_model, vad_ok = _load_silero_vad(device, log)
    log(f"🧠 Silero-VAD: {'loaded' if vad_ok else 'unavailable → energy fallback'}")
    regions = _vad_regions(mono, sr, vad_model, log)
    if not regions:
        raise RuntimeError(
            "Voice activity detection found no speech in vocals.wav. Check that "
            "Step 2 produced a non-silent vocal stem.")

    # ── ② sub-segment → CAMPPlus embeddings ──────────────────────────────────
    # 1.5 s windows on a 0.75 s hop: long enough for a stable speaker vector,
    # short enough that a turn change inside one VAD region stays separable.
    win_s, hop_s = 1.5, 0.75
    segs: List = []
    owners: List[int] = []              # region index per window
    for ri, (a, b) in enumerate(regions):
        if (b - a) < 0.4:
            segs.append(mono[int(a * sr):int(b * sr)])
            owners.append(ri)
            continue
        t = a
        while t < b - 0.3:
            e = min(b, t + win_s)
            segs.append(mono[int(t * sr):int(e * sr)])
            owners.append(ri)
            t += hop_s
    log(f"✂ {len(regions)} region(s) → {len(segs)} embedding window(s)")

    log(f"🧠 Loading CAMPPlus · device={device}")
    sv_model = _load_campplus(device, log)
    embs, kept = _campplus_embed(sv_model, segs, log)
    del sv_model
    log("   " + clear_gpu_cache())

    # ── ③ cluster → one speaker per VAD region (majority vote) ───────────────
    labels = _cluster_speakers(embs, int(expected_speakers or 0), log)
    turns: List[dict] = []
    if len(labels):
        by_region: Dict[int, Counter] = defaultdict(Counter)
        for wi, lab in zip(kept, labels.tolist()):
            by_region[owners[wi]][int(lab)] += 1
        for ri, (a, b) in enumerate(regions):
            votes = by_region.get(ri)
            if not votes:
                continue
            turns.append({"start": float(a), "end": float(b),
                          "raw": f"SPEAKER_{votes.most_common(1)[0][0]:02d}"})
    if not turns:
        log("   ⚠ no speaker embedding survived — treating the track as one voice")
        turns = [{"start": 0.0, "end": mono.size / sr, "raw": "SPEAKER_00"}]

    n_raw = len({t["raw"] for t in turns})
    log(f"   {len(turns)} speaker turn(s) · {n_raw} distinct voice(s) "
        f"· {time.time() - t0:.1f}s")

    # ── cross-reference with the ORIGINAL SRT timeline ────────────────────
    if original_srt_path and Path(original_srt_path).exists():
        cues = load_srt(original_srt_path)
        log(f"🔗 Cross-referencing {len(cues)} original-SRT cues with turns …")
    else:
        log("⚠ No original SRT supplied — deriving cues from speaker turns.")
        cues = [{"index": i + 1, "start": t["start"], "end": t["end"], "text": ""}
                for i, t in enumerate(turns)]

    cue_rows: List[dict] = []
    raw_totals: Dict[str, float] = defaultdict(float)
    for c in cues:
        votes: Dict[str, float] = defaultdict(float)
        for t in turns:
            ov = _overlap(c["start"], c["end"], t["start"], t["end"])
            if ov > 0:
                votes[t["raw"]] += ov
        best = max(votes, key=votes.get) if votes else "SPEAKER_00"
        for k, v in votes.items():
            raw_totals[k] += v
        cue_rows.append({**c, "raw": best})

    # canonical identity tags — most total speech becomes Speaker1
    ranked = sorted(raw_totals.items(), key=lambda kv: -kv[1])
    canon = {raw: f"Speaker{i + 1}" for i, (raw, _) in enumerate(ranked)}
    # optional user hint: keep the N largest voice clusters and fold the
    # rest into their nearest big cluster (temporal-overlap affinity)
    n_exp = int(expected_speakers or 0)
    if 0 < n_exp < len(ranked):
        keep = [raw for raw, _ in ranked[:n_exp]]
        keep_ivs = {r: _merge_intervals([(t["start"], t["end"])
                                         for t in turns if t["raw"] == r])
                    for r in keep}
        for raw, _tot in ranked[n_exp:]:
            ivs = _merge_intervals([(t["start"], t["end"])
                                    for t in turns if t["raw"] == raw])
            best, best_ov = keep[0], 0.0
            for r in keep:
                ov = sum(_overlap(a, b, c, d) for a, b in ivs
                         for c, d in keep_ivs[r])
                if ov > best_ov:
                    best, best_ov = r, ov
            canon[raw] = canon[best]
            log(f"   hint: folded '{raw}' into '{canon[best]}' "
                f"(nearest of {n_exp} expected speakers)")
    for row in cue_rows:
        row["speaker"] = canon.get(row["raw"], "Speaker1")

    # ── clone-prompt mining (top-3 cleanest 5–10 s windows per speaker) ───
    log("🎚 Mining cleanest 5–10 s voice-clone reference windows …")
    spk_intervals: Dict[str, List[List[float]]] = defaultdict(list)
    for t in turns:
        spk_intervals[canon.get(t["raw"], "Speaker1")].append([t["start"], t["end"]])
    for cname in spk_intervals:
        spk_intervals[cname] = _merge_intervals(spk_intervals[cname])
    prompts = _mine_clone_prompts(log, spk_intervals)

    speakers: Dict[str, dict] = {}
    for raw, cname in canon.items():
        spk_cues = [r for r in cue_rows if r["speaker"] == cname]
        speakers[cname] = {
            "raw_label": raw,
            "total_speech_s": round(sum(r["end"] - r["start"] for r in spk_cues), 2),
            "cue_count": len(spk_cues),
            "clone_prompts": prompts.get(cname, []),
        }

    # ── gender profiling — casting must not be a coin flip ────────────────
    log("♀♂ Profiling speaker pitch so voices can be gender-matched …")
    speakers = profile_speakers(speakers, log)

    result = {"model": DIAR_ENGINE, "speakers": speakers, "cues": cue_rows}
    DIARIZATION_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    # ── CSV side-cars (human review + spreadsheet export) ─────────────────────
    # Written ALONGSIDE the JSON, never instead of it: Steps 5–7 read the JSON
    # by key, so a CSV problem can never break the pipeline. Guarded for the
    # same reason.
    try:
        _write_csv(SPEAKER_TURNS_CSV,
                   ["Speaker ID", "Start_Time", "End_Time"],
                   [[canon.get(t["raw"], "Speaker1"), fmt_ts(t["start"]),
                     fmt_ts(t["end"])] for t in turns])
        _write_csv(DIAR_CUES_CSV,
                   ["Speaker ID", "Start_Time", "End_Time", "Text"],
                   [[r["speaker"], fmt_ts(r["start"]), fmt_ts(r["end"]),
                     (r.get("text") or "").replace("\n", " ")] for r in cue_rows])
        log(f"🧾 CSV side-cars → {SPEAKER_TURNS_CSV.name} ({len(turns)} row(s)) · "
            f"{DIAR_CUES_CSV.name} ({len(cue_rows)} row(s))")
    except Exception as e:
        log(f"   ⚠ CSV export skipped: {type(e).__name__}: {e}")

    log(f"✅ Step 3 → diarization_map.json · {len(speakers)} identity tag(s) mapped")
    return result


# ── Automatic gender detection ──────────────────────────────────────────────
# Pyannote tells us HOW MANY voices are present, never WHICH is which — it
# emits no `gender` field at all. With no gender to cast against, the voice
# allocator fell through to a pool whose first key is "Female" and handed every
# speaker a female voice. We recover gender acoustically instead of guessing.
#
# Adult median speaking F0 is ~85-155 Hz for men and ~165-255 Hz for women, so
# anything outside the ambiguous 150-175 Hz overlap commits immediately; inside
# that overlap we require a decisive long-term spectrum, and otherwise report
# UNDETERMINED. We use pYIN rather than plain YIN because pYIN returns an
# explicit voiced/unvoiced decision — see `_estimate_gender`.
_F0_MIN = 60.0
_F0_MAX = 400.0
_F0_MALE_MAX = 150.0
_F0_FEMALE_MIN = 175.0

# Fraction of frames pYIN must mark voiced before we trust any pitch reading.
# Measured on synthetic material: pure silence, near-silence, white noise and
# pink noise all score 0.00; voiced tones score 1.00. 0.15 sits in a wide gap.
_MIN_VOICED_FRAC = 0.15

# Intra-speaker F0 spread within one 5-10 s window. A single speaker stays
# well under this; a window that mixes two speakers blows straight past it.
_MAX_F0_IQR = 70.0

# Long-term spectral centroid at 16 kHz separates the sexes even at equal F0
# (~1500-1800 Hz male vs ~2000-2400 Hz female), but the bands overlap, so we
# only commit on a DECISIVE reading and leave a wide dead zone. This branch is
# literature-calibrated and has NOT been validated against real speech.
_CENTROID_MALE_MAX = 1500.0
_CENTROID_FEMALE_MIN = 2400.0


def _estimate_gender(wave, sr: int) -> dict:
    """Median-F0 gender estimate for one voice sample, using pYIN.

    pYIN rather than plain YIN, because plain YIN has no way to say "there is
    no pitch here": it reports its `fmax` sentinel on silence and a near-random
    value on noise. Measured on synthetic input, plain YIN labelled pure
    silence as *female, 400 Hz* and white noise as *male, 73 Hz* — both at full
    confidence. pYIN's voiced flag rejected all four non-voice inputs outright.
    """
    import librosa
    import numpy as np

    blank = {"gender": "", "f0_hz": 0.0, "centroid_hz": 0.0,
             "voiced_ratio": 0.0, "voiced_frac": 0.0, "confidence": 0.0}
    y = np.asarray(wave, dtype=np.float32).reshape(-1)
    if y.size < sr // 2:                    # under 0.5 s there is no evidence
        return blank

    # Silence drags the median toward the noise floor — measure voiced speech.
    spans = librosa.effects.split(y, top_db=30)
    voiced = np.concatenate([y[a:b] for a, b in spans]) if len(spans) else y
    if voiced.size < sr // 4:
        voiced = y
    voiced_ratio = round(float(voiced.size / max(y.size, 1)), 3)

    f0, flag, _prob = librosa.pyin(voiced, fmin=_F0_MIN, fmax=_F0_MAX, sr=sr,
                                   frame_length=2048, hop_length=512)
    flag = np.asarray(flag, dtype=bool)
    voiced_frac = float(flag.mean()) if flag.size else 0.0

    def _undetermined(**extra):
        return {**blank, "voiced_ratio": voiced_ratio,
                "voiced_frac": round(voiced_frac, 3), **extra}

    if not flag.any() or voiced_frac < _MIN_VOICED_FRAC:
        return _undetermined()              # not a voice — never invent a gender

    f0v = np.asarray(f0, dtype=float)[flag]
    f0v = f0v[np.isfinite(f0v) & (f0v > 0.0)]
    if not f0v.size:
        return _undetermined()

    f0_med = float(np.median(f0v))
    f0_iqr = float(np.percentile(f0v, 75) - np.percentile(f0v, 25))
    centroid = float(np.median(librosa.feature.spectral_centroid(
        y=voiced, sr=sr, hop_length=512)))

    if f0_iqr > _MAX_F0_IQR:
        # A window this wide almost certainly mixes two speakers.
        return _undetermined(f0_hz=round(f0_med, 1),
                            f0_iqr_hz=round(f0_iqr, 1),
                            centroid_hz=round(centroid, 0))

    if f0_med <= _F0_MALE_MAX:
        gender, conf = "male", 0.90
    elif f0_med >= _F0_FEMALE_MIN:
        gender, conf = "female", 0.90
    elif centroid <= _CENTROID_MALE_MAX:
        gender, conf = "male", 0.55         # decisive spectrum, low confidence
    elif centroid >= _CENTROID_FEMALE_MIN:
        gender, conf = "female", 0.55       # decisive spectrum, low confidence
    else:
        gender, conf = "", 0.0              # genuinely ambiguous — no guess

    return {"gender": gender, "f0_hz": round(f0_med, 1),
            "f0_iqr_hz": round(f0_iqr, 1),
            "centroid_hz": round(centroid, 0),
            "voiced_ratio": voiced_ratio,
            "voiced_frac": round(voiced_frac, 3),
            "confidence": round(conf, 2)}


def profile_speakers(speakers: Dict[str, dict], log: Log = _noop) -> Dict[str, dict]:
    """Attach an acoustic gender estimate to each diarized speaker (Step 3).

    The estimate is written into `diarization_map.json` under each speaker, and
    is only ever a *default*: a gender saved by the user in Tab 2
    (`speaker_profiles.json`) always wins at casting time — see
    `_speaker_gender`.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return speakers

    for spk, info in speakers.items():
        # Pool EVERY mined clone window, not just the first one. A speaker's
        # top-3 windows carry up to 30 s of clean speech, so the median F0 and
        # the IQR gate in `_estimate_gender` stop depending on whether one
        # particular 5 s sample happened to be unlucky.
        rels = [r for r in (info.get("clone_prompts") or []) if r]
        paths = [p for p in (BASE_DIR / r for r in rels) if p.exists()]
        if not paths:
            info.setdefault("gender", "")
            continue
        try:
            waves, rates = [], []
            for p in paths:
                data, sr = sf.read(str(p), dtype="float32", always_2d=True)
                waves.append(data.mean(axis=1))
                rates.append(int(sr))
            if len(set(rates)) != 1:
                # Concatenating across sample rates would corrupt the pitch
                # estimate, so fall back to the single best window instead.
                waves, rates = waves[:1], rates[:1]
            prof = _estimate_gender(np.concatenate(waves), rates[0])
        except Exception as e:
            log(f"   ⚠ {spk}: gender profiling skipped ({str(e)[:70]})")
            info.setdefault("gender", "")
            continue
        info.update(prof)
        if prof["gender"]:
            log(f"   ♀♂ {spk} → {prof['gender']} · F0 {prof['f0_hz']:.0f} Hz "
                f"· centroid {prof['centroid_hz']:.0f} Hz "
                f"· confidence {prof['confidence']:.2f}")
        else:
            log(f"   ♀♂ {spk} → undetermined · F0 {prof['f0_hz']:.0f} Hz "
                f"— voices will rotate instead of defaulting")
    return speakers


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 · PARALINGUISTIC EMOTION SCAN  (emotion2vec_plus_large / FunASR)
# ═════════════════════════════════════════════════════════════════════════════

def _load_vocals_16k():
    """Vocals → float32 mono @16 kHz numpy array (ASR/emotion-friendly)."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(VOCALS_WAV), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16_000:
        import torch
        import torchaudio
        mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, 16_000).numpy()
        sr = 16_000
    return mono.astype(np.float32), sr


def step4_emotion_analysis(log: Log, force: bool = False) -> List[dict]:
    """
    Tag every timeline-matched voice segment (the diarized SRT cue windows) with
    one of the 7 canonical emotions, using emotion2vec_plus_large.

    PER-LINE BY DESIGN: one `generate()` call per cue, never a single pass over
    the whole vocal track — a global call would collapse a whole scene onto one
    emotion and Step 7's per-line `exaggeration` would come out flat.

    Artifact shapes are UNCHANGED from the SenseVoice version (Step 5 reads them
    by key), plus a CSV side-car:
      · outputs/emotion_log.txt   — spec format:  [00:50.000] Speaker1 [angry]
      · outputs/emotion_grid.json — machine grid consumed by Step 5
      · outputs/emotion_grid.csv  — the same grid for humans / spreadsheets
    """
    if EMOTION_GRID_JSON.exists() and not force:
        log("↩ Step 4 cached (emotion_grid.json) — skipping.")
        return json.loads(EMOTION_GRID_JSON.read_text(encoding="utf-8"))

    try:
        import numpy as np
        import torch
        import torchaudio  # noqa: F401  (resample + torchaudio ≥ 2.9 shims)
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Step 4 needs the ML stack — module '{missing}' is not installed here. "
            "Local CPU fix:  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu  then  "
            "pip install funasr modelscope  ·  Or run on Google Colab (T4).") from e
    _torchaudio_compat()   # shim removed APIs before funasr's import chain
    _numpy2_compat()       # restore np.NaN / np.float_ aliases for NumPy 2.x
    _torch_load_compat()   # restore legacy torch.load default for trusted checkpoints
    # Dependency probe — step-specific install hint; `_funasr_auto` imports it.
    try:
        from funasr import AutoModel  # noqa: F401
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Step 4 needs FunASR for emotion2vec+ — module '{missing}' is not "
            "installed here.  Fix:  pip install funasr modelscope") from e

    diar = json.loads(DIARIZATION_JSON.read_text(encoding="utf-8"))
    cues = diar["cues"]
    # emotion2vec+ takes no language hint, so this is metadata only. It keeps
    # the `language` key Step 5 reads populated with something honest, instead
    # of the SenseVoice <|en|> tag that no longer exists.
    src_language = (PipelineState.load().artifacts.get("source_lang") or "auto")
    lang = "auto" if src_language == "auto" else src_language

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"🧠 Loading emotion2vec+ · {EMOTION2VEC_MODEL} · device={device}")
    log("   " + log_memory())

    t0 = time.time()
    # hub="hf" is REQUIRED. The modelscope resolution path queries a different
    # emotion2vec export that returns FOUR classes instead of nine, which would
    # silently fold 'disgusted'/'fearful'/'surprised' onto 'neutral'. There is
    # deliberately NO modelscope fallback here — it would succeed and hand back
    # the wrong label set. If HF is unreachable we fail loudly instead.
    model = _funasr_auto([(f"huggingface · {EMOTION2VEC_MODEL}",
                           {"model": EMOTION2VEC_MODEL, "hub": "hf"})],
                         device, log, "emotion2vec+ (Step 4)")

    mono, sr = _load_vocals_16k()
    grid: List[dict] = []
    fails = 0

    for cue in cues:
        a = float(cue["start"])
        b = max(float(cue["end"]), a + 0.2)          # sub-0.2 s guard
        seg = mono[int(a * sr): int(b * sr)]
        emotion = "neutral"
        if seg.size >= int(0.25 * sr):               # < 0.25 s → no signal
            try:
                res = model.generate(
                    input=np.ascontiguousarray(seg, dtype=np.float32),
                    granularity="utterance", extract_embedding=False)
                # labels[] and scores[] come back aligned. Take the argmax of
                # scores rather than blindly trusting labels[0], so a future
                # checkpoint that returns them unordered stays correct.
                labels = res[0].get("labels") if res else None
                scores = res[0].get("scores") if res else None
                if labels:
                    best = 0
                    if scores and len(scores) == len(labels):
                        best = int(np.argmax(np.asarray(scores, dtype=np.float32)))
                    emotion = _normalise_emotion(labels[best])
                else:
                    fails += 1
            except Exception:
                fails += 1
        grid.append({"index": cue["index"], "start": a, "end": b,
                     "speaker": cue.get("speaker", "Speaker1"),
                     "emotion": emotion, "language": lang})

    EMOTION_LOG_TXT.write_text(
        "\n".join(f"[{fmt_ts(g['start'])}] {g['speaker']} [{g['emotion']}]"
                  for g in grid),
        encoding="utf-8")
    EMOTION_GRID_JSON.write_text(json.dumps(grid, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    try:
        _write_csv(EMOTION_GRID_CSV,
                   ["Speaker ID", "Start_Time", "End_Time", "Emotion"],
                   [[g["speaker"], fmt_ts(g["start"]), fmt_ts(g["end"]),
                     g["emotion"]] for g in grid])
        log(f"🧾 CSV side-car → {EMOTION_GRID_CSV.name} ({len(grid)} row(s))")
    except Exception as e:
        log(f"   ⚠ CSV export skipped: {type(e).__name__}: {e}")

    # NOTE: the clone-prompt transcription pass that used to live here is gone.
    # It existed only because CosyVoice's zero-shot path wants its conditioning
    # text verbatim, and it ran a second SenseVoice pass purely to produce it.
    # Chatterbox clones from the audio prompt alone, so under 2.0 that artifact
    # had no consumer — while still costing one extra model pass per speaker.
    # `_ensure_prompt_transcripts()` now short-circuits instead of re-deriving.

    del model
    log("   " + clear_gpu_cache())

    dist = Counter(g["emotion"] for g in grid)
    log(f"✅ Step 4 [{EMOTION2VEC_MODEL.split('/')[-1]}] → emotion_log.txt + "
        f"emotion_grid.{{json,csv}} · {len(grid)} segment(s) in "
        f"{time.time() - t0:.1f}s"
        + (f" · {fails} decode fallback(s)" if fails else ""))
    log("   distribution: " + ", ".join(f"{k}×{v}" for k, v in dist.most_common()))
    return grid


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5 · SRT MAPPING & CONSOLIDATED SCRIPT ASSEMBLY  (pysrt)
# ═════════════════════════════════════════════════════════════════════════════

INDIC_LANGS = {"hi": "devanagari", "sa": "devanagari", "mr": "devanagari",
               "ne": "devanagari", "bn": "bengali", "gu": "gujarati",
               "pa": "gurmukhi", "ta": "tamil", "te": "telugu",
               "kn": "kannada", "ml": "malayalam"}


def transliterate_to_native(text: str, target_lang: str) -> str:
    """Roman (Latin) Indic text -> native script (Devanagari for hi) so
    the TTS front-end can pronounce it. No-op for non-Indic targets or
    native-script text. Fail-soft (returns the original on any error)."""
    scheme = INDIC_LANGS.get((target_lang or "").lower())
    if not scheme or not text:
        return text
    if any(ord(ch) > 0x0900 for ch in text):
        return text
    if not any(("a" <= ch.lower() <= "z") for ch in text):
        return text
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(text, sanscript.ITRANS, scheme)
    except Exception:
        return text


def _translit_on() -> bool:
    """User toggle (Tab 1) — default ON."""
    try:
        return PipelineState.load().artifacts.get("translit", "1") == "1"
    except Exception:
        return True


def step5_assemble_script(translated_srt_path: str,
                          log: Log,
                          force: bool = False) -> List[dict]:
    """
    Map the TRANSLATED SRT text onto the timestamp + speaker + emotion grid,
    producing the consolidated script dictionary array (final_script.json):

        [{ index, start, end, speaker, emotion, instruct, language,
           clone_prompt, original_text, translated_text }, ...]
    """
    if FINAL_SCRIPT_JSON.exists() and not force:
        log("↩ Step 5 cached (final_script.json) — skipping.")
        return json.loads(FINAL_SCRIPT_JSON.read_text(encoding="utf-8"))

    t_cues = load_srt(translated_srt_path)
    diar = json.loads(DIARIZATION_JSON.read_text(encoding="utf-8"))
    o_cues = diar["cues"]
    emotions = {g["index"]: g for g in
                json.loads(EMOTION_GRID_JSON.read_text(encoding="utf-8"))}
    speakers_info = diar.get("speakers", {})
    target_language = (PipelineState.load().artifacts.get("target_lang")
                       or "en")

    n = len(o_cues)
    log(f"🧩 Aligning {len(t_cues)} translated cues ⇄ {n} analysed cues …")

    rows: List[dict] = []
    for i, tc in enumerate(t_cues):
        oc = o_cues[min(i, n - 1)]
        # index-aligned first; if timestamps drifted, search a local window
        if abs(oc["start"] - tc["start"]) > 1.5:
            best_j, best_ov = min(i, n - 1), -1.0
            for j in range(max(0, i - 10), min(n, i + 11)):
                ov = _overlap(tc["start"], tc["end"],
                              o_cues[j]["start"], o_cues[j]["end"])
                if ov > best_ov:
                    best_ov, best_j = ov, j
            oc = o_cues[best_j]

        g = emotions.get(oc["index"], {})
        speaker = oc.get("speaker") or g.get("speaker") or "Speaker1"
        emotion = g.get("emotion", "neutral")
        spk = speakers_info.get(speaker, {})
        prompts = spk.get("clone_prompts") or []

        rows.append({
            "index": tc["index"],
            "start": round(tc["start"], 3),
            "end": round(tc["end"], 3),
            "speaker": speaker,
            "emotion": emotion,
            "instruct": EMOTION_INSTRUCT.get(emotion, "in a calm, neutral tone"),
            "language": g.get("language", "en"),
            "target_language": target_language,
            "clone_prompt": prompts[0] if prompts else "",
            "original_text": oc.get("text", ""),
            "translated_text": (
                transliterate_to_native(tc["text"], target_language)
                if _translit_on() else tc["text"]),
        })

    FINAL_SCRIPT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    log(f"✅ Step 5 → final_script.json · {len(rows)} consolidated row(s)")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  🩺 DIAGNOSTIC MODE — collect ALL errors in one run instead of halting
# ═════════════════════════════════════════════════════════════════════════════

def _render_diagnostic(errors: List[Tuple[str, str]]) -> str:
    """Render a summary report of every error collected during a run."""
    lines = ["═" * 62,
             " 🩺 DIAGNOSTIC REPORT — all errors collected this run",
             "═" * 62]
    for step, err in errors:
        lines.append(f"❌ {step}")
        lines.append(f"   └─ {err[:250]}")
    lines.append(f"→ {len(errors)} error(s) found. Fix and re-run — "
                 "cached steps auto-skip.")
    lines.append("═" * 62)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATORS — generator functions that stream a cumulative console log
#  to the Gradio UI while enforcing strict sequential execution.
# ═════════════════════════════════════════════════════════════════════════════

def run_import_and_analysis(media_path: str,
                            force: bool = False,
                            source_lang: str = "auto",
                            target_lang: str = "en",
                            translit: bool = True,
                            diagnostic: bool = False) -> Generator[str, None, None]:
    """TAB 1 · Steps 1–2: extract audio → Demucs vocal/music split.

    In diagnostic mode, errors are collected instead of halting — dependent
    steps are skipped and a full report prints at the end."""
    log = Log()
    state = PipelineState.load()
    media_path = str(media_path)      # Gradio may hand us NamedString
    src_name = Path(media_path).name
    errors: List[Tuple[str, str]] = []
    if diagnostic:
        yield log("🩺 DIAGNOSTIC MODE — errors collected, pipeline runs to finish")
    try:
        yield log("═" * 62)
        yield log(" 🔬 TAB 1 · IMPORT & ANALYSIS — sequential T4-safe run")
        yield log("═" * 62)
        yield log(f"📥 Media: {src_name}")
        yield log("   " + log_memory())

        # a NEW source file invalidates every cached step downstream
        if state.artifacts.get("source") not in (None, src_name):
            gone = state.invalidate_from("step1")
            yield log(f"♻ New source detected → resetting {len(gone)} cached step(s).")
        state.artifacts["source"] = src_name
        # remember the real path so Step 8 can remux video containers later
        state.artifacts["source_path"] = str(media_path)
        # dubbing language choices (consumed by Steps 4 & 7)
        state.artifacts["source_lang"] = str(source_lang or "auto")
        state.artifacts["target_lang"] = str(target_lang or "en")
        state.artifacts["translit"] = "1" if translit else "0"
        yield log(f"🌐 Languages · original: {state.artifacts['source_lang']}"
                  f" → dub: {state.artifacts['target_lang']}")

        # ── Step 1 · extract audio ─────────────────────────────────────────
        yield log("─" * 62)
        audio: Optional[Path] = None
        try:
            audio = step1_extract_audio(media_path, log, force=force)
            state.mark("step1", audio=audio.name)
            state.save()
            yield log("   " + log_memory())
        except Exception as e:
            if diagnostic:
                errors.append(("Step 1 · extract_audio", str(e)))
                yield log(f"❌ Step 1 failed: {e}")
            else:
                raise

        # ── Step 2 · separate vocals (depends on Step 1) ──────────────────
        if audio is not None:
            yield log("─" * 62)
            try:
                vocals, music = step2_separate_vocals(log, force=force)
                state.mark("step2", vocals=vocals.name, music=music.name)
                state.save()
                yield log("🎬 Analysis complete — switch to Tab 2 for speaker matching.")
            except Exception as e:
                if diagnostic:
                    errors.append(("Step 2 · separate_vocals", str(e)))
                    yield log(f"❌ Step 2 failed: {e}")
                else:
                    raise
        else:
            yield log("⏭ Step 2 · separate_vocals SKIPPED (Step 1 failed)")

        # ── diagnostic report ──────────────────────────────────────────────
        if diagnostic and errors:
            yield log(_render_diagnostic(errors))
        elif not errors:
            pass  # normal completion message already emitted
    except Exception as e:
        yield log(f"❌ Step failed: {e}")
        yield log("   Pipeline halted — fix the issue and re-run "
                  "(cached steps auto-skip).")
    finally:
        state.save()


def run_script_matching(hf_token: Optional[str],
                        original_srt_path: Optional[str],
                        translated_srt_path: Optional[str],
                        force: bool = False,
                        num_speakers: int = 0,
                        diagnostic: bool = False) -> Generator[str, None, None]:
    """TAB 2 · Steps 3–5: diarization → emotion scan → script assembly.

    In diagnostic mode, errors are collected instead of halting — dependent
    steps are skipped and a full report prints at the end."""
    log = Log()
    state = PipelineState.load()
    hf_token = str(hf_token) if hf_token else ""        # NamedString-safe
    original_srt_path = str(original_srt_path) if original_srt_path else None
    translated_srt_path = str(translated_srt_path) if translated_srt_path else None
    errors: List[Tuple[str, str]] = []
    if diagnostic:
        yield log("🩺 DIAGNOSTIC MODE — errors collected, pipeline runs to finish")
    try:
        yield log("═" * 62)
        yield log(" 🧬 TAB 2 · SCRIPT MATCHING & ASSEMBLY — steps 3 · 4 · 5")
        yield log("═" * 62)
        if not VOCALS_WAV.exists():
            raise RuntimeError("Run Tab 1 first — outputs/vocals.wav not found.")

        # ── Step 3 · diarization ───────────────────────────────────────────
        yield log("─" * 62)
        diar: Optional[dict] = None
        try:
            diar = step3_diarization(hf_token, original_srt_path, log, force=force, expected_speakers=num_speakers)
            state.mark("step3", speakers=str(len(diar.get("speakers", {}))))
            state.save()
            yield log("   " + log_memory())
        except Exception as e:
            if diagnostic:
                errors.append(("Step 3 · diarization", str(e)))
                yield log(f"❌ Step 3 failed: {e}")
            else:
                raise

        # ── Step 4 · emotion scan (depends on Step 3) ─────────────────────
        if diar is not None:
            yield log("─" * 62)
            try:
                step4_emotion_analysis(log, force=force)
                state.mark("step4")
                state.save()
                yield log("   " + log_memory())
            except Exception as e:
                if diagnostic:
                    errors.append(("Step 4 · emotion_scan", str(e)))
                    yield log(f"❌ Step 4 failed: {e}")
                else:
                    raise
        else:
            yield log("⏭ Step 4 · emotion_scan SKIPPED (Step 3 failed)")

        # ── Step 5 · script assembly (depends on Steps 3–4) ───────────────
        if diar is not None and EMOTION_GRID_JSON.exists():
            yield log("─" * 62)
            if not translated_srt_path or not Path(translated_srt_path).exists():
                if diagnostic:
                    errors.append(("Step 5 · script_assembly",
                                   "Translated SRT missing — upload it in Tab 1"))
                    yield log("❌ Step 5: Translated SRT missing — upload it in Tab 1")
                else:
                    raise RuntimeError("Translated SRT missing — upload it in Tab 1 "
                                       "to assemble the dubbing script.")
            else:
                try:
                    rows = step5_assemble_script(translated_srt_path, log, force=force)
                    state.mark("step5", rows=str(len(rows)))
                    state.save()
                    yield log("🧬 Script assembled — switch to Tab 3 to verify "
                              "render readiness.")
                except Exception as e:
                    if diagnostic:
                        errors.append(("Step 5 · script_assembly", str(e)))
                        yield log(f"❌ Step 5 failed: {e}")
                    else:
                        raise
        else:
            yield log("⏭ Step 5 · script_assembly SKIPPED (Step 3/4 failed)")

        # ── diagnostic report ──────────────────────────────────────────────
        if diagnostic and errors:
            yield log(_render_diagnostic(errors))
    except Exception as e:
        yield log(f"❌ Step failed: {e}")
        yield log("   Pipeline halted — fix the issue and re-run "
                  "(cached steps auto-skip).")
    finally:
        state.save()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 · render-readiness checklist
# ─────────────────────────────────────────────────────────────────────────────

def render_readiness() -> Dict[str, bool]:
    return {
        "step1_audio.wav — extracted master (Step 1)": AUDIO_WAV.exists(),
        "vocals.wav — Demucs vocal stem (Step 2)": VOCALS_WAV.exists(),
        "music.wav — instrumental stem (Step 2)": MUSIC_WAV.exists(),
        "diarization_map.json — speaker identities (Step 3)": DIARIZATION_JSON.exists(),
        "emotion_grid.json — paralinguistic scan (Step 4)": EMOTION_GRID_JSON.exists(),
        "final_script.json — consolidated dubbing script (Step 5)": FINAL_SCRIPT_JSON.exists(),
        "speaker_scripts.json — TTS-safe split (Step 6)": SPEAKER_SCRIPTS_JSON.exists(),
        "track_speakerN.wav — padded voice masters (Step 7)":
            bool(list(OUTPUTS_DIR.glob("track_speaker*.wav"))),
        "final_mix.wav — ducked master (Step 8)": FINAL_MIX_WAV.exists(),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 6 · SPLITTING ENGINE  (per-speaker channels, TTS-safe text only)
# ═════════════════════════════════════════════════════════════════════════════

# Patterns that must NEVER reach the TTS engine — otherwise the cloned voice
# would read speaker labels, rich emotion tags and timecodes out loud.
_TTS_SANITIZE_RES = [
    re.compile(r"<\|[^|>]{0,32}\|>"),                                    # <|en|> <|HAPPY|>
    re.compile(r"\[[^\]\n]{0,64}\]"),                                    # [calm] [00:50.000] [Music]
    re.compile(r"<[^>\n]{0,64}>"),                                       # <happy> <b> …html
    re.compile(r"\bSpeaker\s*\d+\s*[:\-–—]\s*", re.IGNORECASE),          # "Speaker1:"
    re.compile(r"\bSpeaker\s*\d+\b", re.IGNORECASE),                     # stray "Speaker2"
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\b"),         # 00:50.000 · 1:02:03,5
    re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"),  # SRT arrows
    re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE),                        # stray cue numbers
]


def sanitize_for_tts(text: str) -> str:
    """Strip speaker labels & timeline metadata — returns PURE spoken text."""
    out = str(text or "")
    for rx in _TTS_SANITIZE_RES:
        out = rx.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def step6_split_speaker_scripts(log: Log, force: bool = False) -> Dict[str, List[dict]]:
    """
    Clone the master consolidated array into separate per-speaker tracking
    dictionaries (Project_SpeakerN.txt) with every internal speaker label and
    time-metadata string stripped, so the TTS engine only ever sees the lines
    a human should hear.
    """
    if SPEAKER_SCRIPTS_JSON.exists() and not force:
        log("↩ Step 6 cached (speaker_scripts.json) — skipping.")
        return json.loads(SPEAKER_SCRIPTS_JSON.read_text(encoding="utf-8"))

    rows = json.loads(FINAL_SCRIPT_JSON.read_text(encoding="utf-8"))
    per_speaker: Dict[str, List[dict]] = {}
    for r in rows:
        spk = r.get("speaker") or "Speaker1"
        clean = sanitize_for_tts(r.get("translated_text", ""))
        if not clean:
            log(f"   ⚠ row {r.get('index')} ({spk}) empty after sanitising — skipped.")
            continue
        per_speaker.setdefault(spk, []).append({
            "index": r.get("index"),
            "start": float(r.get("start", 0.0)),
            "end": float(r.get("end", 0.0)),
            "emotion": r.get("emotion", "neutral"),
            "instruct": r.get("instruct", "in a calm, neutral tone"),
            "language": r.get("language", "en"),          # SOURCE language
            "target_language": r.get("target_language", ""),  # DUB language
            "text": clean,                       # ← the ONLY field TTS receives
        })

    for spk in per_speaker:                       # chronological inside channel
        per_speaker[spk].sort(key=lambda d: (d["start"], d["index"] or 0))

    for spk, lines in sorted(per_speaker.items()):
        txt_path = OUTPUTS_DIR / f"Project_{spk}.txt"
        # .txt holds PURE speakable lines — no labels, no timecodes
        txt_path.write_text("\n".join(l["text"] for l in lines), encoding="utf-8")
        log(f"   {txt_path.name}: {len(lines)} sanitised line(s)")

    SPEAKER_SCRIPTS_JSON.write_text(json.dumps(per_speaker, indent=2,
                                               ensure_ascii=False), encoding="utf-8")
    log(f"✅ Step 6 → speaker_scripts.json · {len(per_speaker)} speaker channel(s)")
    return per_speaker


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 7 · COSYVOICE 3.0 ZERO-SHOT TTS  (silence padding + overlap protection)
# ═════════════════════════════════════════════════════════════════════════════

def _cosyvoice_syspath() -> None:
    """
    CosyVoice ships as a SOURCE TREE (no setup.py — `pip install .` fails).
    Register the usual clone locations + the Matcha-TTS submodule on
    sys.path so `import cosyvoice` resolves. Idempotent.
    """
    import sys
    for base in ("/content/CosyVoice",                    # Colab bootstrap
                 str(Path.cwd() / "CosyVoice"),           # clone beside app
                 str(Path.home() / "CosyVoice")):         # home clone
        p = Path(base)
        if (p / "cosyvoice").is_dir():
            for entry in (p, p / "third_party" / "Matcha-TTS"):
                s = str(entry)
                if s not in sys.path:
                    sys.path.insert(0, s)
            return


def _load_cosyvoice(log: Log, prefer: str = ""):
    """
    Lazy CosyVoice loader (zero-shot mode). The engine ships inside the
    FunAudioLLM repo, so operators get a one-line bootstrap when missing.

    `prefer` pins a specific checkpoint — e.g. "iic/CosyVoice2-0.5B" for the
    default CosyVoice 2.0 engine. Empty keeps the historical 3 → 2 → 1 order.
    """
    _cosyvoice_syspath()
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "Step 7 needs the ML stack — 'torch' is not installed here. "
            "Local CPU fix:  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu  ·  Or run on "
            "Google Colab (T4).") from e
    # Multi-pass self-heal: import -> detect missing modules -> pip install
    # the RIGHT pip packages (module names often differ from package names)
    # -> retry. Up to 6 passes; stalls (no progress) exit early.
    _MOD2PKG = {"whisper": "openai-whisper", "cv2": "opencv-python",
                "sklearn": "scikit-learn", "yaml": "pyyaml",
                "matcha": "matcha-tts", "hyperpyyaml": "HyperPyYAML",
                "pil": "pillow", "audioop": "audioop-lts", "wget": "wget"}
    import sys as _sys
    errs, _CV, _prev = [], None, None
    for _attempt in range(6):
        errs = []
        for _cls in ("CosyVoice3", "CosyVoice2", "CosyVoice"):
            try:
                _mod = __import__("cosyvoice.cli.cosyvoice", fromlist=[_cls])
                _CV = getattr(_mod, _cls)
                break
            except Exception as e:      # ImportError OR deeper missing deps
                errs.append(f"{_cls}: {e}")
        if _CV is not None or _attempt == 5:
            break
        _missing = sorted({mm.group(1) for e in errs
                           for mm in [re.search(r"No module named '([^']+)'", str(e))]
                           if mm})
        if not _missing or _missing == _prev:
            break                        # nothing healable, or stalled
        _prev = _missing
        _pkgs = [_MOD2PKG.get(mn.lower(), mn) for mn in _missing]
        log("[i] auto-healing missing CosyVoice deps (pass "
            + str(_attempt + 1) + "/6): " + ", ".join(_missing) + " ...")
        subprocess.run([_sys.executable, "-m", "pip", "install", "-q", *_pkgs],
                       capture_output=True, text=True)
    if _CV is None:
        raise RuntimeError(
            "CosyVoice engine could not be imported. Deepest errors:\n  "
            + "\n  ".join(errs[-2:])
            + "\n\nBootstrap (Colab Cell 4 does this automatically):\n"
            "  git clone --recursive https://github.com/FunAudioLLM/"
            "CosyVoice /content/CosyVoice\n"
            "  then re-run Cell 4 (filtered requirements install).\n"
            "If Cell 4's install failed, its full pip output shows the cause.")

    # Deep deps: imported lazily during model load (not at class import)
    # - pre-heal them too so Step 7 never dies on 'No module named X'.
    if _CV is not None:
        import sys as _sys
        for _attempt in range(4):
            _deep = []
            for _d in ("conformer", "matcha", "hyperpyyaml",
                       "onnxruntime", "whisper"):
                try:
                    __import__(_d)
                except ModuleNotFoundError:
                    _deep.append(_d)
            if not _deep:
                break
            _pkgs = [_MOD2PKG.get(d.lower(), d) for d in _deep]
            log("[i] auto-healing deep CosyVoice deps: "
                + ", ".join(_deep) + " ...")
            subprocess.run([_sys.executable, "-m", "pip", "install", "-q",
                            *_pkgs], capture_output=True, text=True)
    from modelscope import snapshot_download
    # PAIRED engine+checkpoint selection: each CosyVoice class expects
    # its OWN checkpoint layout (CosyVoice3 wants cosyvoice3.yaml,
    # CosyVoice2 wants cosyvoice.yaml) - mixing them crashes with
    # 'yaml not found'. Try pairs in order; first success wins.
    _PAIRS = (("CosyVoice3", "iic/CosyVoice3-0.5B"),
              ("CosyVoice2", "iic/CosyVoice2-0.5B"),
              ("CosyVoice", "iic/CosyVoice-300M"))
    if prefer:               # the engine selector pins one checkpoint first
        _PAIRS = (tuple(p for p in _PAIRS if p[1] == prefer)
                  + tuple(p for p in _PAIRS if p[1] != prefer))
    _errs = []
    for _cls_name, _repo in _PAIRS:
        try:
            _mod = __import__("cosyvoice.cli.cosyvoice", fromlist=[_cls_name])
            _CVc = getattr(_mod, _cls_name)
        except Exception as e:
            _errs.append(f"{_cls_name}: import failed ({e})")
            continue
        try:
            model_dir = snapshot_download(_repo)
        except Exception as e:
            _errs.append(f"{_repo}: download failed ({str(e)[:120]})")
            log(f"[i] CosyVoice checkpoint {_repo} unavailable: "
                + str(e)[:120])
            continue
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"[GPU] Loading CosyVoice - {_repo} - device={device}")
        log("   " + log_memory())
        try:
            return _CVc(model_dir, load_trt=False, fp16=torch.cuda.is_available())
        except TypeError:                   # older constructor w/o trt/jit
            return _CVc(model_dir)
    raise RuntimeError(
        "CosyVoice could not load any engine/checkpoint pair. Errors:\n  "
        + "\n  ".join(_errs))


def _load_prompt_speech(path: str, target_sr: int = 16_000):
    """Clone-prompt wav → torch tensor (1, n) @16 kHz (CosyVoice expects 16k)."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, ch)
    wav = torch.from_numpy(np.ascontiguousarray(data.T)).mean(
        dim=0, keepdim=True)                             # → (1, n)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _ensure_prompt_transcripts(log: Log) -> Dict[str, str]:
    """
    speaker → transcript of its best clone prompt (zero-shot conditioning).

    Under 2.0 this is deliberately a NO-OP that never loads a model. The only
    engine that consumes prompt transcripts is CosyVoice's zero-shot path, and
    this build publishes Chatterbox alone (ENGINE_ALLOWLIST) — so re-deriving
    them would cost a full extra model load for an artifact nothing reads.

    A `clone_prompt_transcripts.json` left behind by a pipeline.py run is still
    honoured, so switching modules mid-project keeps CosyVoice viable.
    """
    if not PROMPT_TRANSCRIPTS_JSON.exists():
        return {}
    raw = json.loads(PROMPT_TRANSCRIPTS_JSON.read_text(encoding="utf-8"))
    return {k: re.sub(r"<\|[^|>]*\|>", " ", str(v)).strip()
            for k, v in raw.items()}


def _cosyvoice_speak(model, text: str, instruct: str, prompt_speech,
                     prompt_text: str):
    """
    Zero-shot clone + emotion control. Falls back gracefully across engine
    generations: instruct2 → zero_shot → cross_lingual.
    Returns (float32 mono numpy, sample_rate).
    """
    import torch
    import numpy as np
    sr = int(getattr(model, "sample_rate", 24_000))

    attempts = []
    # clone (zero-shot) FIRST for faithful speaker voice + emotion;
    # instruct2 is a delimited fallback; cross-lingual last.
    if prompt_text:
        attempts.append(lambda: model.inference_zero_shot(
            text, prompt_text, prompt_speech, stream=False))
    if instruct and hasattr(model, "inference_instruct2"):
        _ins = instruct if "<|endofprompt|>" in instruct else (
            instruct.rstrip(". ") + "<|endofprompt|>")
        attempts.append(lambda: model.inference_instruct2(
            text, _ins, prompt_speech, stream=False))
    if hasattr(model, "inference_cross_lingual"):
        attempts.append(lambda: model.inference_cross_lingual(
            text, prompt_speech, stream=False))

    last_err: Optional[Exception] = None
    for fn in attempts:
        try:
            chunks = list(fn())
            if not chunks:
                raise RuntimeError("engine returned no audio")
            y = torch.cat([c["tts_speech"] for c in chunks], dim=1) \
                .squeeze().cpu().numpy().astype(np.float32)
            return y, sr
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CosyVoice synthesis failed: {str(last_err)[:200]}")


# ═════════════════════════════════════════════════════════════════════════════
#  🎛 TTS ENGINE REGISTRY — selectable engines + one lightweight fallback
#     · CosyVoice 2.0  (DEFAULT)  → falls back to Edge-TTS
#     · CosyVoice 3.0             → TERMINAL: errors surfaced
#     · Chatterbox (Resemble AI)  → TERMINAL: errors surfaced
#     · Fish Audio S2-Pro         → TERMINAL: errors surfaced
#     · Edge-TTS                  → TERMINAL: errors surfaced
#  A TERMINAL engine must either deliver the real dubbed mix or raise a
#  detailed report — never silently substitute a different voice.
#  Edge-TTS is the ONLY lightweight tier (9 Hindi neural voices, both genders).
# ═════════════════════════════════════════════════════════════════════════════

# ── Build identity ────────────────────────────────────────────────────────────
# pipeline.py  → the original six-engine build  (PIPELINE_VARIANT "1.0")
# pipeline2.py → AutoDub Studio 2.0's DEDICATED copy — byte-for-byte identical
#                to this file apart from the header and these two constants
#                (PIPELINE_VARIANT "2.0", ENGINE_ALLOWLIST ("chatterbox",)).
# app.py picks between them with AUTODUB_PIPELINE, so the two can evolve
# independently without either file importing or referencing the other.
# Running `diff pipeline.py pipeline2.py` therefore shows exactly what makes
# 2.0 2.0 — nothing else.
PIPELINE_VARIANT = "2.0"

# Engines published when AUTODUB_TTS_ENGINES is unset. Wording kept accurate in
# BOTH pipeline files — only the value itself differs between them:
#   ()              → no baked filter: publish the full registry below
#   ("chatterbox",) → baked narrow: AutoDub Studio 2.0 is Chatterbox-only
ENGINE_ALLOWLIST: tuple = ("chatterbox",)

DEFAULT_TTS_ENGINE = "cosyvoice2"

TTS_ENGINES: Dict[str, dict] = {
    "cosyvoice2": {
        "label": "CosyVoice 2.0  (default · offline)",
        "kind": "cosyvoice", "repo": "iic/CosyVoice2-0.5B",
        "clone": True, "fallback": "edge", "vram": "~2 GB", "sr": 24_000,
    },
    # ── TERMINAL ENGINES ────────────────────────────────────────────────────
    # `"fallback": ""` means NO substitution. If such an engine cannot load, or
    # keeps failing mid-render, the pipeline ABORTS with a detailed report
    # instead of quietly swapping in a different voice. Silent substitution on
    # a premium engine hides the very bug we need to fix.
    "cosyvoice3": {
        "label": "CosyVoice 3.0  (offline)",
        "kind": "cosyvoice", "repo": "iic/CosyVoice3-0.5B",
        "clone": True, "fallback": "", "vram": "~2 GB", "sr": 24_000,
    },
    "chatterbox": {
        "label": "Chatterbox  (Resemble AI · offline)",
        "kind": "chatterbox",
        # VERIFIED against the released chatterbox-tts 0.1.7 wheel:
        #   ChatterboxMultilingualTTS.from_local(cls, ckpt_dir, device)
        #   ChatterboxMultilingualTTS.from_pretrained(cls, device)
        # Both take NO `repo_id` and NO `t3_model` — REPO_ID *and* the T3
        # checkpoint filename ("t3_mtl23ls_v2.safetensors") are hardcoded in
        # the package, so a T3 cannot be selected by argument in 0.1.7. The
        # finetune therefore has to be *present under that filename*; see
        # `_chatterbox_overlay_dir`.
        # "ResembleAI/Chatterbox-Multilingual-hi" is a v3 OVERLAY repo
        # (t3_hi.safetensors + s3gen_v3 only — no ve.pt/conds.pt). Hindi is
        # already one of the 23 languages in the base weights, so this is
        # opt-in and empty by default.
        "repo": "ResembleAI/chatterbox",
        "overlay_repo": "ResembleAI/Chatterbox-Multilingual-hi",
        "t3_model": "",            # e.g. "t3_hi.safetensors" (opt-in, see above)
        "clone": True, "fallback": "", "vram": "~2 GB", "sr": 24_000,
    },
    "fishs2": {
        "label": "Fish Audio S2-Pro  (heavy · offline)",
        "kind": "fish", "repo": "fishaudio/s2-pro",
        "clone": True, "fallback": "", "vram": "4-12 GB", "sr": 44_100,
    },
    "kokoro": {
        "label": "Kokoro-82M  (light · offline · no clone)",
        "kind": "kokoro", "repo": "hexgrad/Kokoro-82M",
        "clone": False, "fallback": "", "vram": "~1 GB", "sr": 24_000,
        "speed": 1.0,
    },
    "edge": {
        "label": "Edge-TTS  (lightweight · cloud)",
        "kind": "edge", "repo": "",
        "clone": False, "fallback": "", "vram": "0 (cloud)", "sr": 24_000,
    },
}

# ── Engine allow-list (launcher-controlled) ────────────────────────────────────
# A launcher can publish a SUBSET of the registry without forking this file by
# setting AUTODUB_TTS_ENGINES to a comma-separated list of engine ids:
#
#     os.environ["AUTODUB_TTS_ENGINES"] = "chatterbox"
#
# …which is exactly what the Chatterbox-only "AutoDub Studio 2.0" notebook does.
# Unset, empty, or consisting only of unknown ids publishes the FULL registry,
# so the original Colab_Runner.ipynb — which never sets it — behaves as before.
#
# The filter sits here, immediately after the dict literal and before any reader,
# because EVERY consumer indexes TTS_ENGINES: engine_choices(),
# get/set_tts_engine(), resolve_engine(), engine_fallback() and app.py's Tab 1
# radio + Tab 2 audition dropdown all derive from this one object. Narrowing it
# once keeps them consistent instead of each re-implementing the rule — and a
# request for an engine that was filtered out degrades to the active one
# (resolve_engine) rather than raising KeyError.
# AUTODUB_TTS_ENGINES always wins (a per-launch override); otherwise the build's
# own ENGINE_ALLOWLIST applies — () in pipeline.py, ("chatterbox",) in
# pipeline2.py. An empty result of both means no filter: publish everything.
_ENGINE_RAW = ((os.environ.get("AUTODUB_TTS_ENGINES") or "").strip()
               or ",".join(ENGINE_ALLOWLIST))
_AUTODUB_ALLOW = [e.strip() for e in _ENGINE_RAW.split(",")
                  if e.strip() and e.strip() in TTS_ENGINES]
if _AUTODUB_ALLOW:
    _allowed = {eid: TTS_ENGINES[eid] for eid in _AUTODUB_ALLOW}
    TTS_ENGINES.clear()          # mutate in place so earlier references stay
    TTS_ENGINES.update(_allowed) # valid, while honouring the requested order
    if DEFAULT_TTS_ENGINE not in TTS_ENGINES:
        DEFAULT_TTS_ENGINE = next(iter(TTS_ENGINES))
    print(f"[tts] engine allow-list → {', '.join(TTS_ENGINES)}")

_TTS_ENGINE = DEFAULT_TTS_ENGINE


def engine_choices() -> List[Tuple[str, str]]:
    """[(label, engine_id)] — ready for a Gradio dropdown/radio."""
    return [(spec["label"], eid) for eid, spec in TTS_ENGINES.items()]


def get_tts_engine() -> str:
    return _TTS_ENGINE if _TTS_ENGINE in TTS_ENGINES else DEFAULT_TTS_ENGINE


def language_choices(include_auto: bool = False) -> List[Tuple[str, str]]:
    """[(label, code)] the ACTIVE engine can speak — ready for a Gradio dropdown.

    Mirrors engine_choices(): one object decides, so switching engines
    re-derives it and the UI can never advertise a language the newly selected
    engine will reject at synthesis time. `include_auto` is for the source-side
    picker, where "let the ASR work it out" is a real option; the dub-side
    picker must name a language because that code is handed to generate().

    ⚠ Tuple order is (label, code) — NOT the (code, label) that
    DUBBING_LANGUAGES / ENGINE_LANGUAGES are stored in. Gradio unpacks every
    choice as `for _, value in choices` (gradio/components/dropdown.py →
    Dropdown.preprocess), so the SECOND element is the value the browser posts
    back and the first is only what gets painted on screen. Handing it
    (code, label) made each human-readable name a "legal" value while the code
    the UI actually sends was rejected with:

        Value: auto is not in the list of choices: ['🌐 Auto-detect', 'English']

    — the red "Error" banner in Tab 1. The swap below is therefore deliberate.

    An engine with no published set (edge, fishs2) falls back to the curated
    DUBBING_LANGUAGES rather than showing an empty menu.
    """
    langs = [(label, code) for code, label in
             (ENGINE_LANGUAGES.get(get_tts_engine()) or DUBBING_LANGUAGES)
             if code != "auto"]
    if include_auto:
        return [("🌐 Auto-detect", "auto")] + list(langs)
    return list(langs)


def set_tts_engine(engine_id: str) -> str:
    """Select the render engine; persisted so Steps 7 & 8 always agree."""
    global _TTS_ENGINE
    if engine_id in TTS_ENGINES:
        _TTS_ENGINE = engine_id
    st = PipelineState.load()
    st.artifacts["tts_engine"] = _TTS_ENGINE
    st.save()
    return _TTS_ENGINE


# Edge-TTS prosody steering (Tab 2). Edge has FIXED voices — no cloning — so
# pitch/rate/volume are the only way to nudge a line toward the original
# performance. Ranges are what the Edge endpoint accepts.
DEFAULT_EDGE_PROSODY: Dict[str, int] = {"pitch": 0, "rate": 0, "volume": 0}
_EDGE_PROSODY_RANGE = {"pitch": (-100, 100), "rate": (-50, 100),
                       "volume": (-100, 100)}


def get_engine_prosody() -> Dict[str, int]:
    """Current Edge-TTS prosody steering — {"pitch", "rate", "volume"}."""
    cur = dict(DEFAULT_EDGE_PROSODY)
    saved = PipelineState.load().artifacts.get("prosody") or {}
    for key in DEFAULT_EDGE_PROSODY:
        try:
            cur[key] = int(saved.get(key, 0))
        except (TypeError, ValueError):
            cur[key] = 0
    return cur


def set_engine_prosody(pitch: Optional[int] = None, rate: Optional[int] = None,
                       volume: Optional[int] = None) -> Dict[str, int]:
    """Set Edge-TTS prosody steering and persist it so Step 7 & 8 agree.

    `pitch` is in Hz, `rate` and `volume` in percent, all clamped to the ranges
    the Edge endpoint documents. Engines without prosody controls ignore this.
    """
    cur = get_engine_prosody()
    for key, val in (("pitch", pitch), ("rate", rate), ("volume", volume)):
        if val is None:
            continue
        lo, hi = _EDGE_PROSODY_RANGE[key]
        try:
            cur[key] = max(lo, min(hi, int(val)))
        except (TypeError, ValueError):
            pass
    st = PipelineState.load()
    st.artifacts["prosody"] = cur
    st.save()
    return cur


def engine_fallback(engine_id: str, spec: Optional[dict] = None) -> str:
    """The next engine down the chain, or "" when the engine is TERMINAL.

    Only CosyVoice 2.0 degrades (to Edge-TTS). Chatterbox, Fish S2-Pro,
    CosyVoice 3.0 and Edge-TTS itself are terminal: they must either produce
    the real dubbed mix or surface a detailed error — never a silent
    substitution.

    Prefers the live spec (which `resolve_engine` always populates) so a
    dynamically built engine still routes correctly, then falls back to the
    static registry for plain id lookups.
    """
    if isinstance(spec, dict) and "fallback" in spec:
        return str(spec.get("fallback") or "")
    return TTS_ENGINES.get(engine_id, {}).get("fallback", "")


def is_terminal_engine(engine_id: str, spec: Optional[dict] = None) -> bool:
    """True when `engine_id` has no fallback and must fail loudly."""
    return not engine_fallback(engine_id, spec)


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal-engine diagnostics
#  A fallback-less engine must either deliver the real dubbed mix or explain
#  itself. Everything below exists so the user never has to guess WHY an
#  engine died: the console, outputs/engine_error.log and tts_report.json all
#  carry the same detailed report.
# ─────────────────────────────────────────────────────────────────────────────

ENGINE_ERROR_LOG = OUTPUTS_DIR / "engine_error.log"


class TerminalEngineError(RuntimeError):
    """A fallback-less engine could not deliver audio.

    Carries the already-formatted report so callers can print it verbatim
    without re-deriving the diagnosis.
    """

    def __init__(self, report: str) -> None:
        super().__init__(report)
        self.report = report


def engine_env_snapshot() -> str:
    """One-line fingerprint of what is ACTUALLY installed right now."""
    import platform as _platform

    bits = [f"python {_platform.python_version()}"]
    for mod in ("torch", "numpy", "gradio", "librosa", "transformers"):
        try:
            m = __import__(mod)
            bits.append(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception:
            bits.append(f"{mod} MISSING")
    try:
        import torch
        if torch.cuda.is_available():
            try:
                free, _total = torch.cuda.mem_get_info()
                bits.append(f"cuda=True vram_free={free / 2 ** 30:.1f}GB")
            except Exception:
                bits.append("cuda=True")
        else:
            bits.append("cuda=False")
    except Exception:
        pass
    return " · ".join(bits)


def engine_error_report(engine_label: str, attempts: List[str],
                        detail: str = "", fixes: str = "") -> str:
    """Format + persist a detailed terminal-engine failure report.

    `attempts` are one-per-line causes in chronological order, `detail` is the
    full traceback (written to outputs/engine_error.log, never truncated).
    """
    lines = [f"❌ {engine_label} failed — TERMINAL engine (no fallback enabled)"]
    if attempts:
        lines.append("")
        for a in attempts:
            lines.append("  " + str(a).replace("\n", "\n      "))
    lines.append("")
    lines.append("  " + engine_env_snapshot())
    if fixes.strip():
        lines.append("")
        lines.append("  Fix:")
        for f in fixes.strip().splitlines():
            lines.append("    " + f.strip())
    if detail:
        lines.append("")
        lines.append(f"  Full traceback → {ENGINE_ERROR_LOG}")
    report = "\n".join(lines)
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        ENGINE_ERROR_LOG.write_text(
            f"{report}\n\n{'=' * 72}\nFULL TRACEBACK\n{'=' * 72}\n"
            f"{detail or '(none captured)'}\n",
            encoding="utf-8")
    except Exception:
        pass                                  # never mask the real failure
    return report


def _emotion_word(instruct: str) -> str:
    """Normalise an emotion2vec+ / table emotion label to a bare lowercase word.

    Accepts everything `EMOTION2VEC_MAP` produces ('angry'), a Chinese/short-
    English pair ('生气/angry'), or the instruct phrasing Step 7 passes in
    ('in an angry, tense tone') — this is the lookup `_emotion_to_exaggeration`
    runs before consulting `_CHATTER_EXAG`.
    """
    word = re.sub(r"[^A-Za-z]+", "", str(instruct or "")).lower()
    return word or "neutral"


def format_line(text: str, instruct: str, engine_id: str) -> str:
    """Inline emotion tag for engines that read one from the text itself.

    CosyVoice takes emotion as a separate `instruct` argument (see
    `format_instruct`); Edge-TTS has no emotion control; Chatterbox maps it to
    its `exaggeration` scalar. Only the Fish-style engines want the tag
    embedded in the line, using `[free-form bracket]` syntax.
    """
    if TTS_ENGINES.get(engine_id, {}).get("kind") == "fish":
        return f"[{_emotion_word(instruct)}] {text}".strip()
    return text


def format_instruct(instruct: str, engine_id: str) -> str:
    """The separate `instruct` string passed to CosyVoice-style engines."""
    if TTS_ENGINES.get(engine_id, {}).get("kind") == "cosyvoice":
        return f"Speak {_emotion_word(instruct)}."
    return ""


# SenseVoice 7-emotion → Chatterbox `exaggeration` (0-2). Chatterbox has no tag
# system; higher exaggeration = more dramatic AND faster speech, which our
# 1.15x time-stretch overlap guard then clamps back into the time slot.
_CHATTER_EXAG = {"happy": 0.70, "sad": 0.35, "angry": 0.80, "surprised": 0.75,
                 "neutral": 0.40, "fearful": 0.70, "disgusted": 0.65}


def _emotion_to_exaggeration(instruct: str) -> float:
    return float(_CHATTER_EXAG.get(_emotion_word(instruct), 0.50))


# ─────────────────────────────────────────────────────────────────────────────
#  Edge-TTS · the lightweight tier AND the sole fallback node (CosyVoice 2.0)
#  Distinct, gender-matched Indic voice per speaker (tiered by fidelity).
# ─────────────────────────────────────────────────────────────────────────────

# ── VERIFIED against the live endpoint (edge_tts.list_voices()) ──────────────
# Microsoft ships exactly TWO true Hindi neural voices, so a cast larger than
# 2 cannot be covered by hi-IN alone. Marathi and Nepali are written in
# Devanagari and read Hindi text natively → phonetically faithful stand-ins.
# Every TIER 3 locale uses a DIFFERENT script, so it will mispronounce
# Devanagari and is therefore only reached once Tiers 1-2 are exhausted.
_EDGE_TIER1 = ("hi-IN",)                        # true Hindi
_EDGE_TIER2 = ("mr-IN", "ne-NP")                # Devanagari-script Indic
_EDGE_TIER3 = ("gu-IN", "bn-IN", "bn-BD", "ta-IN", "ta-LK", "ta-MY", "ta-SG",
               "te-IN", "kn-IN", "ml-IN", "ur-IN", "ur-PK")
_EDGE_TIERS = (_EDGE_TIER1, _EDGE_TIER2, _EDGE_TIER3)

# How many leading voices per gender read Devanagari natively (Tiers 1 + 2).
_NATIVE_SCRIPT_VOICES = len(_EDGE_TIER1) + len(_EDGE_TIER2)

# Offline mirror of the same tiers — used only when the live list is
# unreachable. Order is significant: allocation takes from the FRONT, so the
# Devanagari-faithful voices are always cast before the foreign-script ones.
# NOTE: "hi-IN-SwaraNeural" is the first FEMALE POOL entry, not a default. It is
# only ever reached by `allocate_speaker_voices`, never hardcoded as a fallback.
_EDGE_PITCH_SUPPORT: Optional[bool] = None      # cached edge-tts >= 6.1 probe
_EDGE_FALLBACK_POOL = {
    "Female": ["hi-IN-SwaraNeural",
               "mr-IN-AarohiNeural", "ne-NP-HemkalaNeural",
               "gu-IN-DhwaniNeural", "bn-IN-TanishaaNeural",
               "bn-BD-NabanitaNeural", "ta-IN-PallaviNeural",
               "ta-LK-SaranyaNeural", "ta-MY-KaniNeural",
               "ta-SG-VenbaNeural", "te-IN-ShrutiNeural",
               "kn-IN-SapnaNeural", "ml-IN-SobhanaNeural",
               "ur-IN-GulNeural", "ur-PK-UzmaNeural"],
    "Male": ["hi-IN-MadhurNeural",
             "mr-IN-ManoharNeural", "ne-NP-SagarNeural",
             "gu-IN-NiranjanNeural", "bn-IN-BashkarNeural",
             "bn-BD-PradeepNeural", "ta-IN-ValluvarNeural",
             "ta-LK-KumarNeural", "ta-MY-SuryaNeural",
             "ta-SG-AnbuNeural", "te-IN-MohanNeural",
             "kn-IN-GaganNeural", "ml-IN-MidhunNeural",
             "ur-IN-SalmanNeural", "ur-PK-AsadNeural"],
}


def _is_native_script(voice: str) -> bool:
    """True when a voice reads Devanagari natively (hi-IN / mr-IN / ne-NP)."""
    v = str(voice or "").lower()
    return any(v.startswith(loc.lower())
               for tier in _EDGE_TIERS[:2] for loc in tier)


def _edge_voice_list(log: Log = _noop) -> Dict[str, List[str]]:
    """Live Indic voices from Edge, grouped by gender and TIER-ordered.

    Genders and names come from the service rather than being hardcoded, so the
    pool stays correct when Microsoft adds or re-genders a voice. The returned
    lists are ordered by locale tier (hi-IN → mr/ne → the rest), and allocation
    takes from the front, so the Devanagari-faithful voices are cast first.
    """
    try:
        import asyncio as _aio
        import concurrent.futures as _cf

        import edge_tts

        def _worker():
            loop = _aio.new_event_loop()
            _aio.set_event_loop(loop)
            try:
                return loop.run_until_complete(edge_tts.list_voices())
            finally:
                loop.close()

        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            voices = ex.submit(_worker).result(timeout=60)

        by_locale: Dict[str, List[Tuple[str, str]]] = {}
        for v in voices:
            by_locale.setdefault(str(v.get("Locale", "")), []).append(
                (str(v.get("ShortName", "")), str(v.get("Gender", "Female"))))

        pool: Dict[str, List[str]] = {}
        for tier in _EDGE_TIERS:                 # tier order == preference
            for loc in tier:
                for name, gender in by_locale.get(loc, []):
                    if name:
                        pool.setdefault(gender, []).append(name)
        pool = {g: list(dict.fromkeys(n)) for g, n in pool.items() if n}
        if pool:
            total = sum(len(x) for x in pool.values())
            detail = ", ".join(f"{g}:{len(v)}" for g, v in sorted(pool.items()))
            log(f"   Edge-TTS · {total} Indic voice(s) online ({detail}) · "
                f"{_NATIVE_SCRIPT_VOICES} Devanagari-faithful per gender")
            return pool
    except Exception as e:
        log(f"   ⚠ Edge voice list unavailable ({str(e)[:80]}) — using the "
            f"built-in Indic pool")
    return {g: list(v) for g, v in _EDGE_FALLBACK_POOL.items()}


def _speaker_gender(spk: str, info: Optional[dict] = None,
                    profiles: Optional[Dict[str, dict]] = None) -> str:
    """Authoritative gender for a speaker → "female" | "male" | "".

    Precedence: the user's Tab 2 override (`speaker_profiles.json`) first, then
    the acoustic estimate that `profile_speakers` wrote into
    `diarization_map.json`. An empty string means UNKNOWN and is a perfectly
    valid answer — the caller rotates voices rather than inventing a gender.
    """
    if profiles is None:
        profiles = load_speaker_profiles()
    override = str((profiles.get(spk) or {}).get("gender", ""))
    auto = str((info or {}).get("gender", ""))
    for raw in (override, auto):
        low = raw.strip().lower()
        if low.startswith("f"):
            return "female"
        if low.startswith("m"):
            return "male"
    return ""


def _interleave_voices(pool: Dict[str, List[str]]) -> List[str]:
    """Alternate genders so consecutive ungendered speakers don't all go female.

    The previous allocator fell back to `sorted(pool)`, and "Female" sorts
    before "Male" — which is precisely why an all-male cast came out sounding
    female. A Female/Male/Female/Male order makes rotation gender-neutral.
    """
    females = list(pool.get("Female") or [])
    males = list(pool.get("Male") or [])
    out: List[str] = []
    for i in range(max(len(females), len(males))):
        if i < len(females):
            out.append(females[i])
        if i < len(males):
            out.append(males[i])
    return out


def voice_cast_reason(spk: str, info: Optional[dict] = None,
                      profiles: Optional[Dict[str, dict]] = None) -> str:
    """Where a speaker's gender came from — shown in the console and Tab 2."""
    if profiles is None:
        profiles = load_speaker_profiles()
    if str((profiles.get(spk) or {}).get("gender", "")).strip():
        return "Tab 2 override"
    if str((info or {}).get("gender", "")).strip():
        return "auto-detected"
    return "undetermined"


def allocate_speaker_voices(speakers_info: Dict[str, dict],
                            log: Log = _noop,
                            pool: Optional[Dict[str, List[str]]] = None,
                            native: Optional[object] = None) -> Dict[str, str]:
    """Give every diarized speaker a DISTINCT, gender-matched voice.

    `pool` lets a non-Edge engine supply its own voice list (Kokoro does); when
    omitted the live Edge Indic pool is used. `native` is the predicate that
    decides whether to warn about a foreign script — Edge's Indic tiers need it,
    Kokoro's Hindi voices do not.

    Deterministic and stable: same-gender speakers are ranked by their measured
    median F0 (highest first) and handed the tier-ordered pool in that order, so
    a speaker keeps its voice even when other speakers are added or removed.

    A speaker whose gender is unknown ROTATES through a gender-interleaved pool
    instead of silently taking a female voice. Voices are never crossed between
    genders: when one gender's pool runs out, that gender's voices are recycled
    and the sharing is reported.
    """
    if pool is None:
        pool = _edge_voice_list(log)
    if native is None:
        native = _is_native_script
    cast: Dict[str, str] = {}
    if not pool:
        log("   ⚠ no voices available at all — casting impossible")
        return cast

    profiles = load_speaker_profiles()
    free = {g: list(v) for g, v in pool.items()}
    rotate = _interleave_voices(pool)          # gender-alternating rotation
    used: set = set()

    # Split the cast by what we actually know, then cast the KNOWN half first so
    # an undetermined speaker can never "steal" a gender-matched voice.
    known: List[Tuple[str, str, float]] = []
    unknown: List[str] = []
    for spk in sorted(speakers_info):
        info = speakers_info.get(spk) or {}
        g = _speaker_gender(spk, info, profiles)
        if g:
            known.append((spk, g, float(info.get("f0_hz") or 0.0)))
        else:
            unknown.append(spk)

    for gender in ("female", "male"):
        key = "Female" if gender == "female" else "Male"
        # Highest measured pitch first — a stable, reproducible rank.
        for spk, _g, _f0 in sorted((s for s in known if s[1] == gender),
                                   key=lambda s: -s[2]):
            if free.get(key):
                pick = free[key].pop(0)
            elif pool.get(key):                # recycle SAME gender, never cross
                pick = pool[key][len(used) % len(pool[key])]
                log(f"   ⚠ {spk}: {gender} voice pool exhausted — reusing "
                    f"{pick} (this speaker now SHARES a voice)")
            else:
                pick = ""
            if pick:
                used.add(pick)
                cast[spk] = pick

    # Unknown gender → gender-alternating rotation, never "female by default".
    for spk in unknown:
        pick = next((v for v in rotate if v not in used), "")
        if not pick and rotate:
            pick = rotate[len(used) % len(rotate)]
            log(f"   ⚠ {spk}: voice pool exhausted — reusing {pick} "
                f"(this speaker now SHARES a voice)")
        if pick:
            used.add(pick)
            cast[spk] = pick

    for spk in sorted(cast):
        info = speakers_info.get(spk) or {}
        gender = _speaker_gender(spk, info, profiles)
        reason = voice_cast_reason(spk, info, profiles)
        warn = "" if native(cast[spk]) else \
            "   ⚠ foreign script — may mispronounce Devanagari"
        log(f"   🎙 {spk} → {cast[spk]}  [{gender or 'gender ?'} · {reason}]{warn}")
    return cast


def _load_edge(log: Log):
    """Edge-TTS needs no weights — only the package. Uses zero GPU memory."""
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        log("[i] installing edge-tts ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "edge-tts"], capture_output=True, text=True)
        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Edge-TTS is not installed → pip install edge-tts") from e
    try:
        import librosa  # noqa: F401   (decodes the returned MP3)
    except ImportError as e:
        raise RuntimeError(
            "librosa is required to decode Edge-TTS audio → "
            "pip install librosa") from e
    log("✅ Edge-TTS ready — cloud neural Hindi voices, 0 GB VRAM.")
    return {"kind": "edge"}


def _edge_supports_pitch() -> bool:
    """True when this edge-tts build accepts Communicate(pitch=...).

    `pitch` was added in edge-tts 6.1. Older builds raise on the unexpected
    keyword, so we introspect the signature once instead of crashing mid-render.
    """
    global _EDGE_PITCH_SUPPORT
    if _EDGE_PITCH_SUPPORT is None:
        try:
            import inspect as _inspect

            import edge_tts
            _EDGE_PITCH_SUPPORT = "pitch" in _inspect.signature(
                edge_tts.Communicate.__init__).parameters
        except Exception:
            _EDGE_PITCH_SUPPORT = False
    return bool(_EDGE_PITCH_SUPPORT)


def _edge_speak(model, text: str, voice: str = "", rate: int = 0,
                volume: int = 0, pitch: int = 0, log: Log = _noop):
    """One Hindi line through the assigned Edge voice → (float32 mono, 24000).

    `pitch` (Hz), `rate` and `volume` (percent) are the Edge endpoint's prosody
    controls — they are what makes a fixed, non-cloning voice sit closer to the
    original performance. Pitch support was added in edge-tts 6.1, so it is
    probed at runtime and dropped rather than crashing on older builds.

    Runs the coroutine in a private thread with its own event loop so it can
    never collide with Gradio's running loop, and retries transient resets.
    """
    import asyncio as _aio
    import concurrent.futures as _cf
    import tempfile

    import librosa
    import numpy as np

    try:
        import edge_tts
    except ImportError as e:
        raise RuntimeError("Edge-TTS not installed → pip install edge-tts") from e

    # No silent default. A bare `hi-IN-SwaraNeural` here is what made an
    # all-male cast sound female: any casting failure fell through to this one
    # hardcoded female voice. Failing loudly is the honest behaviour.
    if not str(voice or "").strip():
        raise RuntimeError(
            "No Edge voice was assigned to this speaker — the voice cast is "
            "empty, so falling back to a hardcoded voice would silently "
            "mis-gender them. Re-run Step 3 (speaker profiling) then Step 7.")
    v = str(voice).strip()
    r = f"{int(rate):+d}%"
    vol = f"{int(volume):+d}%"
    pit = f"{int(pitch):+d}Hz"

    support = _edge_supports_pitch()
    if int(pitch) and not support:
        log("   ⚠ this edge-tts build has no `pitch` control (needs ≥ 6.1) — "
            "continuing without pitch steering.")
    last: Exception = RuntimeError("unknown error")

    for attempt in range(1, 4):
        p = Path(tempfile.mkstemp(suffix=".mp3")[1])
        try:
            def _worker() -> None:
                loop = _aio.new_event_loop()
                _aio.set_event_loop(loop)
                try:
                    kw: dict = {}
                    if support:
                        kw["pitch"] = pit
                    comm = edge_tts.Communicate(text, v, rate=r, volume=vol,
                                                **kw)
                    loop.run_until_complete(comm.save(str(p)))
                finally:
                    loop.close()

            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(_worker).result(timeout=90)
            if p.exists() and p.stat().st_size > 512:
                y, _sr = librosa.load(str(p), sr=24_000, mono=True)
                return np.asarray(y, dtype=np.float32), 24_000
            last = RuntimeError("endpoint returned empty audio")
        except Exception as e:
            last = e
        log(f"   edge attempt {attempt}/3 failed: {str(last)[:110]}")
        if attempt < 3:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Edge-TTS failed after 3 attempts: {str(last)[:180]}")


# ─────────────────────────────────────────────────────────────────────────────
#  Kokoro-82M (hexgrad) · the lightest OFFLINE tier — no cloning, ~1 GB
#  4 Hindi voices: hf_alpha / hf_beta (female), hm_omega / hm_psi (male).
#  Hindi G2P goes through misaki's espeak-ng backend, so the espeak-ng SYSTEM
#  package must be present (apt-get install espeak-ng on Colab).
#  TERMINAL engine: a silent swap to a cloud voice would defeat the point of
#  picking an offline engine, so failures are reported instead.
# ─────────────────────────────────────────────────────────────────────────────

# Kokoro's own Hindi voice ids. Only TWO per gender ship, so a cast larger than
# two per gender necessarily shares voices — that is reported, never hidden.
_KOKORO_HINDI_VOICES: Dict[str, List[str]] = {
    "Female": ["hf_alpha", "hf_beta"],
    "Male": ["hm_omega", "hm_psi"],
}

# A 0.95/0.05 mix is just a worse copy of one voice; blending only means
# something in the middle of the range, so the weight is clamped there.
_KOKORO_BLEND_MIN = 0.30
_KOKORO_BLEND_MAX = 0.70

DEFAULT_KOKORO_BLEND: Dict[str, object] = {"partner": "", "weight": 0.5}
DEFAULT_KOKORO_SPEED = 1.0
_KOKORO_SPEED_RANGE = (0.5, 1.5)

_KOKORO_FIX = """
pip install kokoro>=0.9.4 soundfile
apt-get install -y espeak-ng      # SYSTEM package — Kokoro's Hindi G2P needs it

`kokoro` performs grapheme-to-phoneme with misaki, whose Hindi backend is
espeak-ng. Without the espeak-ng BINARY the import succeeds but every line
fails, which is why it is installed explicitly rather than assumed.
"""


def _kokoro_voice_gender(name: str) -> str:
    """'female' / 'male' / '' from a Kokoro voice id (hf_alpha → female)."""
    low = str(name or "").strip().lower()
    if len(low) >= 2 and low[0] == "h":
        return {"f": "female", "m": "male"}.get(low[1], "")
    return ""


def get_kokoro_speed() -> float:
    """Kokoro speaking-rate multiplier (0.5–1.5)."""
    try:
        val = float((PipelineState.load().artifacts.get("kokoro") or {})
                    .get("speed", DEFAULT_KOKORO_SPEED))
    except (TypeError, ValueError):
        val = DEFAULT_KOKORO_SPEED
    lo, hi = _KOKORO_SPEED_RANGE
    return max(lo, min(hi, val))


def set_kokoro_speed(speed: float) -> float:
    """Set and persist the Kokoro speaking-rate multiplier (clamped)."""
    lo, hi = _KOKORO_SPEED_RANGE
    try:
        val = max(lo, min(hi, float(speed)))
    except (TypeError, ValueError):
        val = DEFAULT_KOKORO_SPEED
    st = PipelineState.load()
    blk = dict(st.artifacts.get("kokoro") or {})
    blk["speed"] = val
    st.artifacts["kokoro"] = blk
    st.save()
    return val


def get_kokoro_blend() -> Dict[str, object]:
    """Kokoro voice-blend config — {"partner": <voice id>, "weight": 0.3–0.7}."""
    cfg = dict(DEFAULT_KOKORO_BLEND)
    saved = (PipelineState.load().artifacts.get("kokoro") or {}).get("blend") or {}
    cfg["partner"] = str(saved.get("partner") or "")
    try:
        cfg["weight"] = float(saved.get("weight", 0.5))
    except (TypeError, ValueError):
        cfg["weight"] = 0.5
    return cfg


def set_kokoro_blend(partner: str = "", weight: Optional[float] = None) -> dict:
    """Enable/disable Kokoro voice blending and persist it.

    `partner` is a second Kokoro voice; every cast voice is mixed with it at
    `weight` (clamped to 0.30–0.70). A partner of a different gender is refused
    at synthesis time so a speaker can never be de-gendered by the blend.
    """
    cfg = get_kokoro_blend()
    cfg["partner"] = str(partner or "").strip()
    if weight is not None:
        try:
            cfg["weight"] = max(_KOKORO_BLEND_MIN,
                                min(_KOKORO_BLEND_MAX, float(weight)))
        except (TypeError, ValueError):
            pass
    st = PipelineState.load()
    blk = dict(st.artifacts.get("kokoro") or {})
    blk["blend"] = cfg
    st.artifacts["kokoro"] = blk
    st.save()
    return cfg


def _kokoro_voice_pool() -> Dict[str, List[str]]:
    """A copy of the static Kokoro Hindi pool, in gender/tier order."""
    return {g: list(v) for g, v in _KOKORO_HINDI_VOICES.items()}


def _kokoro_cast(speakers_info: Dict[str, dict], log: Log = _noop) -> Dict[str, str]:
    """Gender-matched Kokoro voice per speaker (same rules as Edge)."""
    # Every Kokoro Hindi voice is Devanagari-native, so there is nothing to warn
    # about — hence the always-true predicate.
    return allocate_speaker_voices(speakers_info, log,
                                   pool=_kokoro_voice_pool(),
                                   native=lambda _v: True)


# ── Phase D · auto-blend above capacity ──────────────────────────────────────
# Kokoro ships FOUR Hindi voices, so a balanced cast of ≤4 maps one-to-one and
# every speaker is already distinct. Past that a voice is RECYCLED and two
# speakers become audibly identical — the same failure the Edge pool has, just
# smaller. Rather than pretend a fifth voice exists, each recycled speaker
# keeps its base voice but mixes in the other voice of the same gender at a
# canonical weight no other speaker of that gender uses.

# Canonical weight = share of the gender's FIRST pool voice. Jumping to the
# extremes first (0.70 → 0.30 → 0.50) maximises the distance between
# consecutive speakers within the 0.30–0.70 clamp, where a 0.95/0.05 mix would
# just be a worse copy of one voice.
_BLEND_LADDER = (0.70, 0.30, 0.50, 0.60, 0.40, 0.65, 0.35, 0.55, 0.45,
                 0.62, 0.38, 0.54, 0.46)

KOKORO_HINDI_CAPACITY = sum(len(v) for v in _KOKORO_HINDI_VOICES.values())


def auto_kokoro_blends(cast: Dict[str, str],
                       speakers_info: Optional[Dict[str, dict]] = None,
                       log: Log = _noop) -> Dict[str, dict]:
    """Per-speaker blends that keep an over-capacity Kokoro cast distinguishable.

    Returns `{speaker: {"partner": <voice>, "weight": <0.30–0.70>}}` for the
    speakers whose voice had to be SHARED, or `{}` when no blending is needed.
    A speaker left OUT of the map renders on its pure voice.

    `weight` is chosen as a canonical share of the gender's first pool voice and
    then translated onto whatever voice the allocator actually assigned
    (`base * w + partner * (1-w)`). Translating matters: without it, a recycled
    speaker on `pool[1]` and a recycled speaker on `pool[0]` could both request
    the same ladder value and end up with the identical tensor.

    **Precedence** — a manually configured Tab 2 blend partner is an explicit
    instruction and wins; auto-blending is then off. If voices are still being
    shared in manual mode we say so, because one identical blend cannot separate
    two speakers.
    """
    cast = {k: v for k, v in (cast or {}).items() if v}
    if len(cast) < 2:
        return {}

    shared = sorted(v for v, n in Counter(cast.values()).items() if n > 1)
    manual = str((get_kokoro_blend() or {}).get("partner") or "").strip()
    if manual:
        log(f"   ⚖ manual Kokoro blend partner '{manual}' applies to every "
            f"speaker — auto-blend off.")
        if shared:
            log(f"   ⚠ {len(shared)} voice(s) are still shared; one identical "
                f"blend cannot separate those speakers. Clear the Tab 2 blend "
                f"partner to let Phase D spread them.")
        return {}

    if not shared:
        return {}

    blends: Dict[str, dict] = {}
    # Bucket by the gender of the ASSIGNED voice (always consistent with `cast`),
    # then rank highest F0 first — the allocator's own deterministic order.
    buckets: Dict[str, List[str]] = defaultdict(list)
    for spk, voice in cast.items():
        buckets[_kokoro_voice_gender(voice) or "?"].append(spk)

    for gender, spks in buckets.items():
        pool = _kokoro_voice_pool().get(str(gender).capitalize(), [])
        if len(pool) < 2:                       # nothing to mix this voice with
            continue
        spks.sort(key=lambda s: -float(
            ((speakers_info or {}).get(s) or {}).get("f0_hz") or 0.0))
        seen: set = set()
        step = 0
        for spk in spks:
            base = cast[spk]
            if base not in seen:
                seen.add(base)                  # first speaker on this voice → pure
                continue
            alpha = _BLEND_LADDER[step % len(_BLEND_LADDER)]
            step += 1
            if base == pool[0]:
                partner, weight = pool[1], alpha
            elif base == pool[1]:
                partner, weight = pool[0], 1.0 - alpha
            else:                               # voice outside the known pool
                continue
            blends[spk] = {"partner": partner,
                           "weight": round(max(_KOKORO_BLEND_MIN,
                                               min(_KOKORO_BLEND_MAX,
                                                   float(weight))), 3)}

    if blends:
        log(f"   🔀 Phase D auto-blend · {len(cast)} speakers > "
            f"{KOKORO_HINDI_CAPACITY} Kokoro Hindi voices — "
            f"{len(blends)} speaker(s) get a distinct same-gender mix.")
        for spk in sorted(blends):
            b = blends[spk]
            log(f"      {spk}: {cast[spk]} ⊕ {b['partner']} @ "
                f"{b['weight']:.2f}  [gender kept · now distinct]")
    elif shared:
        log(f"   ⚠ voices are shared but no blend could be built — the "
            f"speaker(s) on {', '.join(shared)} will sound alike.")
    return blends


def _espeak_status() -> Tuple[bool, str]:
    """Is the espeak-ng BINARY on PATH? Informational, not a hard gate."""
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return True, ""
    return False, ("espeak-ng not found on PATH; Kokoro's Hindi G2P falls back "
                   "to it and will fail if misaki cannot bundle one. Install "
                   "with:  apt-get install -y espeak-ng")


def _install_kokoro(log: Log) -> List[str]:
    """pip-install kokoro + soundfile, then apt-get the espeak-ng binary."""
    notes: List[str] = []
    for args in (("kokoro>=0.9.4", "soundfile"), ("misaki[en]",)):
        log(f"[i] pip install {' '.join(args)}")
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                *args], capture_output=True, text=True)
            rc, err = r.returncode, (r.stderr or r.stdout or "")
        except Exception as e:
            rc, err = -1, f"{type(e).__name__}: {e}"
        if rc:
            notes.append(f"[pip {' '.join(args)}] rc={rc} "
                         + " | ".join(err.strip().splitlines()[-4:])[:300])

    ok, _why = _espeak_status()
    if not ok:
        log("[i] apt-get install -y espeak-ng (Kokoro's Hindi G2P backend)")
        try:
            r = subprocess.run(["apt-get", "-qq", "-y", "install", "espeak-ng"],
                               capture_output=True, text=True)
            if r.returncode:
                notes.append(f"[apt-get espeak-ng] rc={r.returncode} "
                             + " | ".join((r.stderr or "").strip()
                                          .splitlines()[-3:])[:250])
        except Exception as e:
            notes.append(f"[apt-get espeak-ng] {type(e).__name__}: "
                         f"{str(e)[:150]}")
    return notes


def _load_kokoro(log: Log):
    """Load Kokoro-82M for Hindi — TERMINAL engine (no fallback)."""
    attempts: List[str] = []
    try:
        import kokoro  # noqa: F401
    except Exception as e:
        attempts.append(f"[import] {type(e).__name__}: {str(e)[:300]}")
        attempts.extend(_install_kokoro(log))
        try:
            import kokoro  # noqa: F401
        except Exception as e2:
            attempts.append(
                f"[import retry] {type(e2).__name__}: {str(e2)[:300]}")
            raise TerminalEngineError(engine_error_report(
                "Kokoro", attempts, detail=traceback.format_exc(),
                fixes=_KOKORO_FIX)) from e2

    ok, why = _espeak_status()
    if not ok:
        # Not fatal by itself — misaki may bundle espeak-ng — but it is the most
        # likely reason for row failures, so say it up front rather than at row 1.
        log(f"   ⚠ {why}")

    import torch

    from kokoro import KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[GPU] Loading Kokoro-82M · device={device} · lang_code='h' (Hindi)")
    log("   " + log_memory())
    try:
        pipe = KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M",
                         device=device)
    except Exception as e:
        attempts.append(f"[KPipeline] {type(e).__name__}: {str(e)[:500]}")
        raise TerminalEngineError(engine_error_report(
            "Kokoro", attempts, detail=traceback.format_exc(),
            fixes=_KOKORO_FIX)) from e

    log("✅ Kokoro-82M ready — 4 Hindi voices (hf_alpha · hf_beta · hm_omega · "
        "hm_psi), 24 kHz, ~1 GB VRAM.")
    return {"kind": "kokoro", "pipe": pipe, "cache": {}, "sr": 24_000}


def _to_numpy(x):
    """torch tensor / list / ndarray → ndarray, without importing torch eagerly."""
    import numpy as np
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _kokoro_extract_audio(chunk):
    """Audio array from a kokoro generator item, whichever API shape it uses.

    kokoro changed its public generator from `(graphemes, phonemes, audio)`
    tuples to a `KPipeline.Result` dataclass carrying `output.audio`. Accepting
    both means a version bump cannot silently degrade every line to silence.
    Returns None when the item carries no audio (a "quiet" pipeline).
    """
    if isinstance(chunk, (tuple, list)):
        return chunk[2] if len(chunk) >= 3 else None
    out = getattr(chunk, "output", None)
    if out is None:
        return None
    audio = getattr(out, "audio", None)
    return getattr(chunk, "audio", None) if audio is None else audio


def _kokoro_loaded(pipe, name: str, cache: Dict[str, object]):
    """`KPipeline.load_voice(name)`, cached per session.

    Kokoro exposes no `pipeline.voices` dict — `load_voice` is the accessor and
    it returns a torch tensor.
    """
    if name not in cache:
        cache[name] = pipe.load_voice(name)
    return cache[name]


def _kokoro_voice_pack(pipe, voice: str, blend: Optional[dict],
                       cache: Dict[str, object], log: Log = _noop):
    """Resolve a Kokoro voice — optionally a weighted BLEND of two — to a tensor.

    Blending is a weighted average of two `load_voice` tensors, which is exactly
    the mechanism Kokoro's authors document (`pipeline(text, voice=<tensor>)`).
    """
    name = str(voice or "").strip()
    if not name:
        return None

    cfg = dict(DEFAULT_KOKORO_BLEND)
    cfg.update(blend or {})
    partner = str(cfg.get("partner") or "").strip()
    try:
        w = float(cfg.get("weight", 0.5))
    except (TypeError, ValueError):
        w = 0.5
    w = max(_KOKORO_BLEND_MIN, min(_KOKORO_BLEND_MAX, w))

    if partner and partner != name:
        pg, ng = _kokoro_voice_gender(partner), _kokoro_voice_gender(name)
        if pg and ng and pg != ng:
            log(f"   ⚠ Kokoro blend partner '{partner}' is {pg} but '{name}' is "
                f"{ng} — blend skipped so the speaker keeps their gender.")
        else:
            key = f"{name}|{partner}|{w:.2f}"
            if key not in cache:
                a = _kokoro_loaded(pipe, name, cache)
                b = _kokoro_loaded(pipe, partner, cache)
                cache[key] = a * w + b * (1.0 - w)
            return cache[key]
    return _kokoro_loaded(pipe, name, cache)


def _kokoro_speak(model, text: str, voice: str = "", speed: float = 1.0,
                  blend: Optional[dict] = None, log: Log = _noop):
    """One Hindi line through the assigned Kokoro voice → (float32 mono, 24000).

    A consolidated row can exceed the model's ~400-character window, so kokoro's
    own generator may yield several chunks; they are concatenated with a short
    gap so the line still lands on the timeline as ONE clip.
    """
    import numpy as np

    if not str(text or "").strip():
        raise RuntimeError("Kokoro received an empty line.")

    pipe = (model or {}).get("pipe")
    if pipe is None:
        raise RuntimeError("Kokoro model was not loaded (no KPipeline).")

    pack = _kokoro_voice_pack(pipe, voice, blend,
                              model.setdefault("cache", {}), log)
    if pack is None:
        raise RuntimeError(
            "Kokoro has no voice for this speaker — the voice cast is empty. "
            "Re-run Step 3 (speaker profiling) then Step 7.")

    sr = int(model.get("sr", 24_000))
    pieces: List[np.ndarray] = []
    try:
        for chunk in pipe(str(text), voice=pack, speed=float(speed)):
            audio = _kokoro_extract_audio(chunk)
            if audio is None:
                continue
            arr = np.asarray(_to_numpy(audio), dtype=np.float32).reshape(-1)
            if arr.size:
                pieces.append(arr)
    except Exception as e:
        raise RuntimeError(
            f"Kokoro synthesis failed ({type(e).__name__}: {str(e)[:200]}). If "
            f"this mentions espeak/phonemes, the espeak-ng system package is "
            f"missing → apt-get install -y espeak-ng") from e

    if not pieces:
        raise RuntimeError(
            "Kokoro produced no audio for this line — its generator yielded "
            "nothing (G2P or voice load failed silently).")
    if len(pieces) == 1:
        return pieces[0], sr

    gap = np.zeros(int(0.06 * sr), dtype=np.float32)     # avoids join clicks
    merged: List[np.ndarray] = []
    for i, piece in enumerate(pieces):
        if i:
            merged.append(gap)
        merged.append(piece)
    return np.concatenate(merged).astype(np.float32), sr


# ─────────────────────────────────────────────────────────────────────────────
#  Chatterbox (Resemble AI) · clone-capable, ~2 GB
#  NOTE: every generated file carries a Resemble PerTh neural watermark.
# ─────────────────────────────────────────────────────────────────────────────

# Shown verbatim in every Chatterbox failure report. The `--no-deps` install is
# NOT a micro-optimisation: chatterbox-tts pins numpy<2, torch==2.6.0,
# torchaudio==2.6.0, transformers==5.2.0 and gradio==6.8.0, so a plain
# `pip install chatterbox-tts` downgrades NumPy, torch and Gradio out from
# under the running pipeline — which is fatal on Colab (NumPy 2.x) and breaks
# the Gradio UI we are rendering from.
# Shown verbatim in every Chatterbox failure report. The `--no-deps` install is
# NOT a micro-optimisation: chatterbox-tts pins numpy<2.0.0 (python<3.13),
# torch==2.6.0, torchaudio==2.6.0, transformers==5.2.0 and gradio==6.8.0, so a
# plain `pip install chatterbox-tts` rewrites NumPy / torch / Gradio out from
# under the running pipeline — which is fatal on Colab (NumPy 2.x) and breaks
# the Gradio UI we are rendering from.
_CHATTERBOX_FIX = """
pip install --no-deps chatterbox-tts
pip install resemble-perth s3tokenizer conformer safetensors omegaconf pyloudnorm pykakasi spacy-pkuseg diffusers "librosa>=0.10"

Do NOT run a plain `pip install chatterbox-tts`: it pins numpy<2,
torch==2.6.0, torchaudio==2.6.0, transformers==5.2.0 and gradio==6.8.0
and will downgrade NumPy / torch / Gradio out from under this pipeline.
`resemble-perth` is mandatory — chatterbox does a top-level `import perth`.

Chatterbox also imports torchaudio, transformers, einops, scipy, tokenizers
and huggingface_hub at module scope. Colab already ships all six, so they are
deliberately NOT in the list above (reinstalling them would risk the very
downgrade we are avoiding). Only add one by hand if the traceback names it.

If the retry said `Protobuf Gencode/Runtime ... gencode 6.31.1 runtime 5.29.6`:
    pip install -U "protobuf>=6.31.1"
The chatterbox dependencies above pull an `onnx` whose generated code was made
by protoc 6.31.x, and protobuf refuses NEW gencode on an OLDER runtime
("New Gencode + Old Runtime = Never Allowed"). Colab ships the 5.29.x runtime,
so the runtime has to come up to meet the gencode. Upgrading is safe: Python
gencode dating from 3.20.0 is guaranteed support through at least 8.x, so this
cannot strand any other package in the notebook.
"""


def _install_chatterbox(log: Log) -> List[str]:
    """Install chatterbox-tts WITHOUT letting it rewrite numpy/torch/gradio.

    Takes the package alone (`--no-deps`) then adds only its true runtime
    dependencies, so the interpreter we are already running inside keeps its
    NumPy 2.x / torch / Gradio versions. Returns pip notes for the failure
    report (empty when every step succeeded).
    """
    notes: List[str] = []
    steps = (
        ("--no-deps", "chatterbox-tts"),
        ("resemble-perth", "s3tokenizer", "conformer", "safetensors",
         "omegaconf", "pyloudnorm", "pykakasi", "spacy-pkuseg", "diffusers",
         "librosa>=0.10"),
    )
    for args in steps:
        log(f"[i] pip install {' '.join(args)}")
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                *args], capture_output=True, text=True)
            rc = r.returncode
            err = (r.stderr or r.stdout or "")
        except Exception as e:                # pip missing / no network
            rc, err = -1, f"{type(e).__name__}: {e}"
        if rc:
            tail = " | ".join(err.strip().splitlines()[-4:])
            notes.append(f"[pip {' '.join(args)}] rc={rc} {tail}"[:300])
    return notes


def _is_protobuf_gencode_error(exc: BaseException) -> bool:
    """True when `exc` is protobuf's gencode/runtime VersionError.

    Matching the message rather than the type alone is deliberate: the raise
    comes out of generated code, and the signature we care about is "protobuf"
    appearing together with "gencode" (or the runtime-too-old phrasing that
    follows it). That pair is what distinguishes onnx's 6.31.x gencode meeting
    Colab's 5.29.x runtime from every other way an import can fail. Anything
    else must NOT be repaired here — the engine keeps its normal failure path.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return "protobuf" in text and ("gencode" in text or "cannot be older" in text)


def _repair_protobuf_runtime(log: Log) -> List[str]:
    """Upgrade the protobuf runtime so it can load onnx's newer generated code.

    Chatterbox's dependencies bring in an `onnx` whose `_pb2` modules were
    generated by protoc 6.31.x, while Colab ships the 5.29.x runtime. Protobuf's
    guarantee is absolute — "New Gencode + Old Runtime = Never Allowed" — so the
    import raises VersionError before chatterbox can finish loading and the
    ENTIRE engine is reported as broken when in fact only the runtime is behind.

    This is called only after the real import has already failed with that
    exact error, which is what makes it safe: an earlier revision guessed the
    failing import with a `python -c "import onnx.onnx_ml_pb2"` probe, and
    because that module imports cleanly on its own the probe passed, returned
    early, and the repair never ran — the shipped fix text told people to run a
    command the code itself would not. Repairing off the observed exception
    cannot silently no-op.

    Upgrading the runtime rather than downgrading onnx is the direction
    protobuf supports: Python gencode from 3.20.0 onward stays supported on
    newer runtimes through at least 8.x, so nothing else in the notebook is
    stranded. The partly-imported modules are dropped from `sys.modules`
    afterwards so the retry loads them against the NEW runtime instead of
    replaying the failure already cached in this interpreter.

    Returns pip notes for the failure report (empty when the upgrade worked).
    """
    notes: List[str] = []
    log("[i] onnx's gencode is newer than the protobuf runtime — upgrading")
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                            "protobuf>=6.31.1"],
                           capture_output=True, text=True, timeout=600)
        rc, err = r.returncode, (r.stderr or r.stdout or "")
    except Exception as e:                    # pip missing / no network
        rc, err = -1, f"{type(e).__name__}: {e}"
    if rc:
        tail = " | ".join(err.strip().splitlines()[-4:])
        notes.append(f"[pip -U protobuf>=6.31.1] rc={rc} {tail}"[:300])
        return notes
    stale = ("onnx", "google.protobuf", "google._upb", "chatterbox")
    for name in [m for m in list(sys.modules)
                 if any(m == p or m.startswith(p + ".") for p in stale)]:
        sys.modules.pop(name, None)
    log("   ✓ protobuf runtime upgraded — retrying the import")
    return notes


# `from_local()` opens exactly these names and nothing else:
#   ve.pt · t3_mtl23ls_v2.safetensors · s3gen.pt · conds.pt (optional) ·
#   grapheme_mtl_merged_expanded_v1.json · Cangjie5_TC.json
# This is byte-for-byte the `allow_patterns` chatterbox-tts 0.1.7 uses in its
# own from_pretrained(), so the snapshot can never come up short a file the
# loader opens. The T3 name is hardcoded in the wheel — see the overlay helper.
_T3_FILENAME = "t3_mtl23ls_v2.safetensors"
_CHATTERBOX_FILES = ["ve.pt", _T3_FILENAME, "s3gen.pt", "conds.pt",
                     "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"]
# Missing any of these is fatal; conds.pt is a graceful `if exists` in the wheel.
_CHATTERBOX_REQUIRED = ["ve.pt", _T3_FILENAME, "s3gen.pt",
                        "grapheme_mtl_merged_expanded_v1.json"]


def _call_with_supported_kwargs(fn, *args, **kwargs):
    """Call `fn` passing only the keyword arguments it actually declares.

    Engine wheels drift under us: chatterbox-tts 0.1.7 ships
    `from_local(cls, ckpt_dir, device)`, while the unreleased master adds
    `t3_model=None`. A keyword the installed build never heard of is an
    instant TypeError *at model-load time* — which is exactly how a perfectly
    good install was being reported as a broken one. Filtering against the
    real signature keeps 0.1.7 working and lets a future release pick the
    value up for free.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):            # C builtin / exotic callable
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)             # declares **kwargs → pass all
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _chatterbox_overlay_dir(base: Path, finetune: Path, log: Log) -> Path:
    """A `from_local()`-ready dir = base checkpoint + the finetune T3.

    `from_local()` hardcodes the T3 filename, so a finetune cannot be handed
    over as an argument — it has to be *present under that name*. Overwriting
    the file inside the shared HuggingFace cache would silently change the
    voice of every later run, so we build a private directory of links instead
    (free, no multi-GB copy), degrading to hard links then a real copy when
    the platform refuses symlinks.
    """
    out = base.parent / f"{base.name}-{finetune.stem}"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(base.iterdir()):
        if not f.is_file() or f.name == _T3_FILENAME:
            continue
        dst = out / f.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(f.resolve())
        except OSError:
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)           # Windows / no privs
    target = out / _T3_FILENAME
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        target.symlink_to(finetune.resolve())
    except OSError:
        shutil.copy2(finetune, target)
    log(f"   ↗ overlay checkpoint → {out}")
    return out


def _chatterbox_ckpt_dir(log: Log, t3_model: str = "") -> Path:
    """Download the base multilingual weights (+ optional T3 finetune).

    We cannot select the T3 through `from_pretrained()`: in 0.1.7 it is
    `from_pretrained(cls, device)` — no `repo_id` and no `t3_model` — and the
    wheel hardcodes both REPO_ID and the T3 filename. Snapshotting explicitly
    is therefore the only way to merge the separate
    `Chatterbox-Multilingual-hi` finetune onto the base checkpoint.
    """
    from huggingface_hub import snapshot_download

    repo = str(TTS_ENGINES["chatterbox"].get("repo") or "ResembleAI/chatterbox")
    overlay = str(TTS_ENGINES["chatterbox"].get("overlay_repo") or "")
    token = os.environ.get("HF_TOKEN") or None

    log(f"[⇩] Chatterbox weights ← {repo}")
    ckpt_dir = Path(snapshot_download(repo_id=repo, token=token,
                                      allow_patterns=_CHATTERBOX_FILES))

    if t3_model and overlay:
        log(f"[⇩] Chatterbox T3 finetune ← {overlay} ({t3_model})")
        try:
            finetune = Path(snapshot_download(
                repo_id=overlay, token=token,
                allow_patterns=[t3_model])) / t3_model
            if not finetune.exists():
                raise FileNotFoundError(f"{t3_model} absent from {overlay}")
            ckpt_dir = _chatterbox_overlay_dir(ckpt_dir, finetune, log)
        except Exception as e:
            log(f"   ⚠ T3 finetune unavailable ({str(e)[:110]}) — using the "
                f"multilingual default T3 instead.")

    missing = [f for f in _CHATTERBOX_REQUIRED if not (ckpt_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Chatterbox checkpoint incomplete — missing "
                           f"{', '.join(missing)} in {ckpt_dir}")
    return ckpt_dir

def _load_chatterbox(log: Log):
    """Load Chatterbox Multilingual — TERMINAL engine (no fallback).

    Verified against the released chatterbox-tts 0.1.7 wheel
    (src/chatterbox/mtl_tts.py), not against master:
      · REPO_ID and the T3 checkpoint name are hardcoded in the package
      · `from_local(cls, ckpt_dir, device)` — NO `t3_model` keyword
      · `from_pretrained(cls, device)` — NO `repo_id`, NO `t3_model`; it reads
        HF_TOKEN from the environment itself before calling from_local
      · `generate()` returns a (1, N) torch tensor, `model.sr == S3GEN_SR`
        (24000), and "hi" is already one of its 23 languages
      · `__init__` builds `perth.PerthImplicitWatermarker()` and the module
        does a top-level `import perth` — so `resemble-perth` is mandatory

    The call goes through `_call_with_supported_kwargs` so that a build which
    *does* add `t3_model` (unreleased master) is used while 0.1.7 keeps
    working instead of dying on an unknown keyword.

    Any failure raises TerminalEngineError with the full diagnosis instead of
    silently falling back to a different voice.
    """
    attempts: List[str] = []

    def _import_chatterbox():
        """One import attempt.

        Wrapped in a helper so the recovery chain below reads as a sequence of
        attempts rather than three copies of the same `from` statement.
        """
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        return ChatterboxMultilingualTTS

    try:
        ChatterboxMultilingualTTS = _import_chatterbox()
    except Exception as e:
        attempts.append(f"[import] {type(e).__name__}: {str(e)[:300]}")
        attempts.extend(_install_chatterbox(log))
        try:
            ChatterboxMultilingualTTS = _import_chatterbox()
        except Exception as e2:
            attempts.append(
                f"[import retry] {type(e2).__name__}: {str(e2)[:300]}")
            if not _is_protobuf_gencode_error(e2):
                raise TerminalEngineError(engine_error_report(
                    "Chatterbox", attempts, detail=traceback.format_exc(),
                    fixes=_CHATTERBOX_FIX)) from e2
            # The dependency install worked; chatterbox itself never got a
            # chance to fail. protobuf refused to load onnx's newer gencode, so
            # raise the runtime to meet it and try once more. Repairing off the
            # exception we just caught — rather than probing for it in advance —
            # is what guarantees this path actually runs.
            attempts.extend(_repair_protobuf_runtime(log))
            try:
                ChatterboxMultilingualTTS = _import_chatterbox()
            except Exception as e3:
                attempts.append(
                    f"[import retry 2] {type(e3).__name__}: {str(e3)[:300]}")
                raise TerminalEngineError(engine_error_report(
                    "Chatterbox", attempts, detail=traceback.format_exc(),
                    fixes=_CHATTERBOX_FIX)) from e3
            attempts[:] = ["[i] chatterbox imported once the protobuf runtime "
                           "was upgraded to meet the gencode its onnx was "
                           "built with."]
        else:
            # The install fixed it, so the first error was a missing-package
            # blip and not the real fault. Demote it — leaving "No module named
            # 'chatterbox'" in the report sends people chasing a module that is
            # now perfectly importable.
            attempts[:] = ["[i] chatterbox was missing and was installed on the "
                           "fly; the import now succeeds."]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[GPU] Loading Chatterbox · device={device}")
    log("   " + log_memory())

    t3_model = str(TTS_ENGINES["chatterbox"].get("t3_model") or "")
    try:
        ckpt_dir = _chatterbox_ckpt_dir(log, t3_model)
        loader = getattr(ChatterboxMultilingualTTS, "from_local", None)
        if callable(loader):
            args = (ckpt_dir, device)
        else:                          # a future build without from_local
            loader = ChatterboxMultilingualTTS.from_pretrained
            args = (device,)
            attempts.append("[i] from_local absent — used from_pretrained")
        model = _call_with_supported_kwargs(
            loader, *args, t3_model=(t3_model or None))
    except Exception as e:
        attempts.append(f"[load] {type(e).__name__}: {str(e)[:300]}")
        raise TerminalEngineError(engine_error_report(
            "Chatterbox", attempts, detail=traceback.format_exc(),
            fixes=_CHATTERBOX_FIX)) from e

    log("✅ Chatterbox ready — multilingual (Hindi among 23 languages). "
        "Every generated file carries a Resemble PerTh watermark.")
    return model


def _chatterbox_speak(model, text: str, ref_audio: str = "", language: str = "hi",
                      exaggeration: float = 0.5, cfg_weight: float = 0.0,
                      log: Log = _noop):
    """One cloned line → (float32 mono, model.sr).

    `cfg_weight=0` is deliberate: our clone prompts are recorded in the SOURCE
    language but we synthesise Hindi, and Resemble advise 0 to avoid the output
    inheriting the reference clip's accent.
    """
    import numpy as np

    kwargs: dict = {"exaggeration": float(exaggeration),
                    "cfg_weight": float(cfg_weight)}
    try:
        out = model.generate(text, language_id=language,
                             audio_prompt_path=(ref_audio or None), **kwargs)
    except TypeError:
        kwargs.pop("cfg_weight", None)          # older build without cfg
        out = model.generate(text, language_id=language,
                             audio_prompt_path=(ref_audio or None), **kwargs)

    y = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
    return np.asarray(y, dtype=np.float32).squeeze(), int(
        getattr(model, "sr", 24_000))


# ─────────────────────────────────────────────────────────────────────────────
#  Fish Audio S2-Pro · heaviest engine (self-hosted, quantized)
#  bf16 needs ~21 GB → use NF4 (~12 GB) or the s2.cpp GGUF pair (~4-8 GB).
#  Validate first with:  python fish_s2_probe.py --all
#  Licence: Fish Audio Research License — non-commercial; ships attribution.
# ─────────────────────────────────────────────────────────────────────────────

FISH_ATTRIBUTION = ("This model is licensed under the Fish Audio Research "
                    "License, Copyright © 39 AI, INC. All Rights Reserved.")


def _load_fishs2(log: Log):
    """Load self-hosted S2-Pro.

    Two supported shapes:
      1. an `s2.cpp` / official REST server already listening (S2_CPP_URL)
      2. the fish-speech python package with local (quantized) weights
    """
    url = (os.environ.get("S2_CPP_URL") or "").strip()
    if url:
        log(f"✅ Fish S2-Pro via local /v1/tts server → {url}")
        log(f"   {FISH_ATTRIBUTION}")
        return {"kind": "fish", "mode": "server", "url": url.rstrip("/")}

    try:
        import fish_speech  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Fish Speech is not installed. Self-hosting S2-Pro needs either:\n"
            "  · an s2.cpp server  → export S2_CPP_URL=http://127.0.0.1:8080\n"
            "  · or  git clone https://github.com/fishaudio/fish-speech "
            "&& pip install -e .\n"
            "Run `python fish_s2_probe.py --all` first to validate this GPU."
        ) from e

    log(f"[GPU] Fish S2-Pro weights={TTS_ENGINES['fishs2']['repo']}")
    log("   " + log_memory())
    log(f"   {FISH_ATTRIBUTION}")
    return {"kind": "fish", "mode": "local"}


def _fishs2_speak(model, text: str, ref_audio: str = "", log: Log = _noop,
                  **kw):
    """One S2-Pro line through a local /v1/tts server → (float32, 44.1 kHz).

    The server returns WAV, so the real sample rate is read from the file
    rather than assumed. The emotion tag must already be inline in `text`.
    """
    import tempfile

    import soundfile as sf
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("pip install requests") from e

    url = str(model.get("url", "")).rstrip("/")
    if not url:
        raise RuntimeError(
            "Fish S2-Pro in-process synthesis is not wired for this build.\n"
            "Use an s2.cpp server instead:\n"
            "  1. build https://github.com/mach92432/s2.cpp\n"
            "  2. start it with the q4_k_m GGUF pair (~4 GB VRAM)\n"
            "  3. export S2_CPP_URL=http://127.0.0.1:8080\n"
            "Run `python fish_s2_probe.py --all` first to validate this GPU "
            "(S2-Pro needs 24 GB at bf16).")

    payload: dict = {"text": text, "format": "wav", "sample_rate": 44100}
    if ref_audio:
        payload["reference_audio"] = str(ref_audio)

    r = requests.post(f"{url}/v1/tts", json=payload, timeout=900)
    r.raise_for_status()

    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    tmp.write_bytes(r.content)
    y, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    return y.mean(axis=1).astype("float32"), int(sr)


# ─────────────────────────────────────────────────────────────────────────────
#  voice preview — audition a casting / prosody choice BEFORE a full render
# ─────────────────────────────────────────────────────────────────────────────

PREVIEW_WAV = OUTPUTS_DIR / "voice_preview.wav"
_PREVIEW_ENGINES = ("edge", "kokoro")           # fixed-voice engines only
_PREVIEW_TEXT = "नमस्ते, यह मेरी आवाज़ का नमूना है।"


def _speaker_info(spk: str) -> dict:
    """`diarization_map.json` entry for one speaker ({} when unknown)."""
    try:
        diar = json.loads(DIARIZATION_JSON.read_text(encoding="utf-8"))
        return (diar.get("speakers", {}) or {}).get(spk) or {}
    except Exception:
        return {}


def _preview_voice_for(eid: str, speaker: str, log: Log) -> str:
    """Cast ONE speaker in isolation, for a preview."""
    if not speaker:
        return ""
    info = _speaker_info(speaker)
    if eid == "kokoro":
        return _kokoro_cast({speaker: info}, log).get(speaker, "")
    return allocate_speaker_voices({speaker: info}, log).get(speaker, "")


def preview_voice(text: str = "", engine_id: str = "", speaker: str = "",
                  voice: str = "", pitch: int = 0, rate: int = 0,
                  volume: int = 0, speed: float = 1.0,
                  blend_partner: str = "", blend_weight: float = 0.5,
                  log: Log = _noop) -> Tuple[Optional[str], str]:
    """Synthesize one short line with the current casting → (wav path, message).

    Only the FIXED-VOICE engines (Edge, Kokoro) are previewable. For a cloning
    engine the voice *is* the speaker's own recording, so there is nothing to
    audition without loading multi-gigabyte weights — we say so instead of
    pretending otherwise.
    """
    eid = engine_id if engine_id in TTS_ENGINES else get_tts_engine()
    spec = resolve_engine(eid)
    kind = spec["kind"]

    if kind not in _PREVIEW_ENGINES:
        return None, (f"Preview covers fixed-voice engines only "
                      f"({', '.join(_PREVIEW_ENGINES)}). '{eid}' clones each "
                      f"speaker's own recording, so there is no separate voice "
                      f"to audition.")

    sample = str(text or "").strip() or _PREVIEW_TEXT
    v = str(voice or "").strip() or _preview_voice_for(eid, speaker, log)
    if not v:
        return None, ("No voice to preview — diarize in Tab 2 first, or pass a "
                      "voice explicitly. A preview with a hardcoded default "
                      "would misrepresent the render.")

    try:
        model = spec["load"](log)
        if kind == "edge":
            y, sr = _edge_speak(model, sample, voice=v, rate=int(rate),
                                volume=int(volume), pitch=int(pitch), log=log)
            note = f"pitch {int(pitch):+d} Hz · rate {int(rate):+d}%"
        else:
            blend = {"partner": str(blend_partner or "").strip(),
                     "weight": float(blend_weight)}
            y, sr = _kokoro_speak(model, sample, voice=v, speed=float(speed),
                                  blend=blend, log=log)
            note = (f"speed ×{float(speed):.2f}"
                    + (f" · blended with {blend['partner']}"
                       if blend["partner"] else ""))
    except TerminalEngineError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Preview failed ({type(e).__name__}: {str(e)[:200]})"

    try:
        import soundfile as sf
        sf.write(str(PREVIEW_WAV), y, sr, subtype="FLOAT")
    except Exception as e:
        return None, f"Preview rendered but could not be saved: {e}"

    return str(PREVIEW_WAV), (f"Preview · {v} · {len(y) / sr:.2f}s · {sr} Hz · "
                              f"{note}")


# ─────────────────────────────────────────────────────────────────────────────
#  engine resolution — binds an id to its (loader, speaker) pair
# ─────────────────────────────────────────────────────────────────────────────

def resolve_engine(engine_id: str = "") -> dict:
    """Return the spec + callables for `engine_id` (defaults to the selection)."""
    eid = engine_id if engine_id in TTS_ENGINES else get_tts_engine()
    spec: dict = dict(TTS_ENGINES[eid])
    spec["id"] = eid
    kind = spec["kind"]

    if kind == "edge":
        spec["load"] = _load_edge
        spec["speak"] = _edge_speak
        spec["instruct"] = False
        # Edge has no cloning, so pitch/rate/volume are its only expressive
        # controls — read them from the persisted Tab 2 state.
        spec["prosody"] = get_engine_prosody()
    elif kind == "chatterbox":
        spec["load"] = _load_chatterbox
        spec["speak"] = _chatterbox_speak
        spec["instruct"] = False
    elif kind == "kokoro":
        spec["load"] = _load_kokoro
        spec["speak"] = _kokoro_speak
        spec["instruct"] = False
        # Persisted so a re-render reproduces the same timbre.
        spec["speed"] = get_kokoro_speed()
        spec["blend"] = get_kokoro_blend()
    elif kind == "fish":
        spec["load"] = _load_fishs2
        spec["speak"] = _fishs2_speak
        spec["instruct"] = False
    else:                                   # cosyvoice family
        spec["load"] = (lambda log, _r=spec["repo"]: _load_cosyvoice(log, prefer=_r))
        spec["speak"] = _cosyvoice_speak
        spec["instruct"] = True
    return spec






# ─────────────────────────────────────────────────────────────────────────────
#  engine dispatch + failover  (Edge-TTS is the terminal node)
# ─────────────────────────────────────────────────────────────────────────────

class EngineRuntime:
    """Holds the active engine + its model, and performs the failover to Edge.

    Failover happens at two points, and ONLY for engines that declare a
    `fallback`:
      · load time  — the usual failure (OOM, weights missing, not installed)
      · 3 consecutive row failures — the engine loaded but cannot produce audio

    Every other engine (Chatterbox, Fish S2-Pro, CosyVoice 3.0, Edge-TTS) is
    TERMINAL. Those raise `TerminalEngineError` with a full report instead of
    substituting a different voice, because a silent substitution is what made
    the previous Chatterbox failure so hard to diagnose.
    """

    def __init__(self, engine: dict, log: Log, dub_lang: str = "hi") -> None:
        self.log = log
        self.spec = engine
        self.label = str(engine.get("label") or engine.get("id") or "engine")
        self.model = None
        self.failed = 0
        self.errors: List[str] = []      # every failed row, oldest first
        self.fell_back_from = ""
        # The language we dub INTO (not the source language detected per line).
        # Multi-lingual engines such as Chatterbox must be told this explicitly.
        self.dub_lang = str(dub_lang or "hi").lower()

    @property
    def id(self) -> str:
        return self.spec["id"]

    @property
    def kind(self) -> str:
        return self.spec["kind"]

    def _switch(self, fb_id: str, why: str) -> None:
        self.log(f"⚠ {why}")
        self.log(f"   ↳ falling back to the lightweight engine: {fb_id}")
        self.fell_back_from = self.id
        self.spec = resolve_engine(fb_id)
        self.failed = 0
        self.model = self.spec["load"](self.log)

    def load(self) -> None:
        try:
            self.model = self.spec["load"](self.log)
        except TerminalEngineError:
            raise                                   # loader already reported
        except Exception as e:
            causes = [f"[load] {type(e).__name__}: {str(e)[:700]}"]
            fb_id = engine_fallback(self.id, self.spec)
            if not fb_id:
                raise TerminalEngineError(engine_error_report(
                    self.label, causes, detail=traceback.format_exc())) from e
            self._switch(fb_id, f"{self.id} could not load ({str(e)[:160]})")

    def speak(self, spk, row, prompt_speech, transcript, voice_cast,
              blends: Optional[Dict[str, dict]] = None):
        """Synthesize one row → (clip, sr).

        Fallback engines switch tier after 3 consecutive failures. TERMINAL
        engines have nowhere to go, so 3 consecutive failures abort the render
        with a full report — never a silent voice substitution, and never a
        silently-empty master track.

        `blends` is a per-speaker override (Phase D auto-blend). It wins over
        the engine-wide blend from Tab 2 for the speakers it names.
        """
        try:
            clip, sr = self._dispatch(spk, row, prompt_speech, transcript,
                                      voice_cast, blends)
            self.failed = 0
            return clip, sr
        except Exception as e:
            self.failed += 1
            note = (f"[row {row.get('index')} · {spk}] "
                    f"{type(e).__name__}: {str(e)[:300]}")
            self.errors.append(note)
            fb_id = engine_fallback(self.id, self.spec)

            if not fb_id:                            # terminal → explain, abort
                if self.failed < 3:
                    raise
                causes = ([f"{self.failed} consecutive failures — giving up"] +
                          self.errors[-6:])
                if self.errors[:-6]:
                    causes.insert(1, f"... {len(self.errors) - 6} earlier "
                                     f"failure(s) omitted")
                raise TerminalEngineError(engine_error_report(
                    self.label, causes, detail=traceback.format_exc())) from e

            if fb_id != self.id:
                self._switch(fb_id, f"{self.id} failed {self.failed}x in a row "
                                    f"({str(e)[:90]})")
                return self._dispatch(spk, row, prompt_speech, transcript,
                                      voice_cast, blends)
            raise

    def _dispatch(self, spk, row, prompt_speech, transcript, voice_cast,
                  blends: Optional[Dict[str, dict]] = None):
        text = str(row["text"])
        instruct = str(row.get("instruct", "") or "")
        sid = self.id
        if self.kind == "cosyvoice":
            return self.spec["speak"](self.model,
                                      format_line(text, instruct, sid),
                                      format_instruct(instruct, sid),
                                      prompt_speech, transcript)
        if self.kind == "chatterbox":
            return self.spec["speak"](
                self.model, text, ref_audio=prompt_speech,
                language=(row.get("target_language") or self.dub_lang),
                exaggeration=_emotion_to_exaggeration(instruct))
        if self.kind == "kokoro":
            # Per-speaker auto-blend (Phase D) wins over the engine-wide
            # Tab 2 blend for the speakers it names; everyone else keeps the
            # global setting (or none at all).
            return self.spec["speak"](
                self.model, text, voice=voice_cast.get(spk, ""),
                speed=float(self.spec.get("speed", 1.0)),
                blend=(blends or {}).get(spk) or self.spec.get("blend") or {})
        if self.kind == "edge":
            return self.spec["speak"](self.model, text,
                                      voice=voice_cast.get(spk, ""),
                                      **(self.spec.get("prosody") or {}))
        # fish-style: emotion travels inline in the text
        return self.spec["speak"](self.model,
                                  format_line(text, instruct, sid),
                                  ref_audio=prompt_speech)


def step7_synthesize(log: Log, force: bool = False,
                     fit_max_speed: float = 1.35,
                     fit_min_speed: float = 0.90,
                     fit_stretch_short: bool = False) -> dict:
    """
    Engine-agnostic synthesis onto per-speaker silent master timelines:
      · one master numpy ZERO-SIGNAL array per speaker, exactly matching the
        duration of the original media (the master audio clock)
      · every row is cloned with the speaker's clean voice profile + emotion
        paralinguistic tags (English · Hindi · Spanish)
      · TWO-WAY FIT — each clip is fitted onto its own SRT slot with librosa's
        PHASE-VOCODER time-stretch (pitch-preserving):
          · over-long  → compressed, capped at `fit_max_speed` (default 1.35×),
                         then hard-trimmed with a 30 ms fade as a last resort
          · short      → optionally stretched, floored at `fit_min_speed`
                         (default 0.90×). OFF by default.

    `fit_stretch_short` is off on purpose: a line that ends 0.4 s inside its
    slot leaves 0.4 s of natural silence, whereas slowing the speech to consume
    it sounds slurred. Only compression is audibly a *fix*, so only compression
    runs unprompted. Both caps are plain arguments, so Tab 2/3 can expose them
    as knobs without touching this function.

    (Phase vocoder, NOT WSOLA: `librosa.effects.time_stretch` is phase-vocoder
    based. It is transparent to roughly 1.3× and gets metallic beyond that,
    which is why the cap is a tunable rather than a hidden constant.)
    """
    import numpy as np
    import soundfile as sf

    engine = resolve_engine()                 # selected in Tab 1
    eid = engine["id"]

    track_files = sorted(OUTPUTS_DIR.glob("track_speaker*.wav"))
    if TTS_REPORT_JSON.exists() and track_files and not force:
        cached = json.loads(TTS_REPORT_JSON.read_text(encoding="utf-8"))
        # Re-render when the engine changed — otherwise switching engines would
        # silently reuse the previous engine's tracks.
        if cached.get("engine", "") == eid:
            log(f"↩ Step 7 cached [{eid}] — skipping (force to redo).")
            return cached
        log(f"♻ Engine changed → {eid} "
            f"(was {cached.get('engine', 'unknown')}) — re-rendering Step 7.")
        force = True

    if not FINAL_SCRIPT_JSON.exists():
        raise RuntimeError("Run Tab 2 first — final_script.json missing.")

    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("librosa is missing → pip install librosa") from e

    scripts = step6_split_speaker_scripts(log, force=False)
    n_rows = sum(len(v) for v in scripts.values())
    if not n_rows:
        raise RuntimeError("No speakable rows survived sanitisation.")

    duration = _wav_duration(AUDIO_WAV)          # master clock

    speakers_info = json.loads(DIARIZATION_JSON.read_text(encoding="utf-8")) \
        .get("speakers", {})
    transcripts = _ensure_prompt_transcripts(log)

    # ── engine bring-up (degrades to the lightweight tier on failure) ─────
    dub_lang = (PipelineState.load().artifacts.get("target_lang") or "hi")
    rt = EngineRuntime(engine, log, dub_lang=dub_lang)
    rt.load()
    eid = rt.id
    if rt.kind == "cosyvoice":
        sr = int(getattr(rt.model, "sample_rate", 24_000))
    elif rt.kind == "chatterbox":
        sr = int(getattr(rt.model, "sr", 24_000))
    else:
        sr = int(rt.spec.get("sr", 24_000))
    if rt.fell_back_from:
        log(f"   active engine: {eid}  (fell back from {rt.fell_back_from})")
        engine = rt.spec

    # Non-cloning engines have FIXED voices → cast a distinct, gender-matched
    # voice per speaker so one speaker never silently absorbs another's.
    voice_cast: Dict[str, str] = {}
    blends: Dict[str, dict] = {}
    if rt.kind == "edge":
        voice_cast = allocate_speaker_voices(speakers_info, log)
    elif rt.kind == "kokoro":
        voice_cast = _kokoro_cast(speakers_info, log)
        # Phase D — past Kokoro's 4 Hindi voices a cast has to share, so give
        # every recycled speaker a distinct same-gender mix.
        blends = auto_kokoro_blends(voice_cast, speakers_info, log)

    # ── the auto-padding system: silent zero-signal master per speaker ────
    n_total = int(round(duration * sr))
    masters: Dict[str, np.ndarray] = {
        spk: np.zeros(n_total, dtype=np.float32) for spk in scripts}
    log(f"🎚 {len(masters)} silent master track(s) · {duration:.2f}s @ {sr} Hz each")

    # global chronological order → 'next sequential line start' for any row
    timeline = sorted(({"spk": spk, **l} for spk, lines in scripts.items()
                       for l in lines),
                      key=lambda d: (d["start"], d.get("index") or 0))

    report = {"engine": eid, "sample_rate": sr,
              "duration_s": round(duration, 3), "rows": []}
    t0 = time.time()
    failed_rows = 0
    for i, row in enumerate(timeline, 1):
        spk = row["spk"]
        start = float(row["start"])
        # time slice available before the NEXT sequential line starts
        avail = ((float(timeline[i]["start"]) - start) if i < len(timeline)
                 else (duration - start))
        avail = max(avail, 0.5)                  # never less than ½ s of room
        max_samples = int(avail * sr)

        # A clone prompt is only required by engines that CLONE. Edge-TTS (and
        # Kokoro) synthesise from a fixed, pre-trained voice, so demanding a
        # prompt from them used to silently skip every row and hand Step 8 an
        # all-silent master.
        info = speakers_info.get(spk, {})
        prompt_rel = (info.get("clone_prompts") or [""])[0]
        if not prompt_rel and bool(rt.spec.get("clone", False)):
            failed_rows += 1
            log(f"   ⚠ row {row.get('index')} — no clone prompt for {spk}; "
                f"skipped (engine '{eid}' clones and needs one).")
            continue
        prompt_speech = str(BASE_DIR / prompt_rel) if prompt_rel else ""

        try:
            clip, clip_sr = rt.speak(spk, row, prompt_speech,
                                     transcripts.get(spk, ""), voice_cast,
                                     blends)
            # normalise every engine's output into the master sample rate
            if clip_sr and int(clip_sr) != sr:
                clip = _resample_1d(clip, int(clip_sr), sr)
            clip = np.asarray(clip, dtype=np.float32).reshape(-1)
        except TerminalEngineError:
            # No fallback exists and the engine is failing systematically —
            # abort the whole render so Step 8 never mixes a silent track.
            raise
        except Exception as e:
            failed_rows += 1
            log(f"   ⚠ row {row.get('index')} synthesis failed: "
                f"{type(e).__name__}: {str(e)[:200]}")
            continue

        gen_s = clip.size / sr
        # ── TWO-WAY fit onto this line's own slot ────────────────────────────
        # ratio > 1 ⇒ the clip overruns its slot (would bleed into the next
        # speaker's window) ⇒ compress. ratio < 1 ⇒ it underruns ⇒ optionally
        # stretch, clamped at fit_min_speed so a short line is never stretched
        # into slurred speech just to close a silence.
        ratio = (clip.size / max_samples) if max_samples else 1.0
        speed, trimmed, mode = 1.0, False, "fit"
        if ratio > 1.0:
            speed = float(min(fit_max_speed, ratio))
            mode = "compressed"
        elif ratio < 1.0 and fit_stretch_short:
            speed = float(max(fit_min_speed, ratio))
            mode = "stretched"
        if abs(speed - 1.0) > 1e-3:
            clip = librosa.effects.time_stretch(y=clip, rate=speed)
        if clip.size > max_samples:          # last resort: hard snap + fade
            fade = int(0.03 * sr)
            clip = clip[:max_samples]
            if clip.size > fade:
                clip[-fade:] *= np.linspace(1.0, 0.0, fade).astype(np.float32)
            trimmed, mode = True, "trimmed"

        pos = int(start * sr)
        end = min(pos + clip.size, n_total)
        seg = clip[: end - pos].astype(np.float32)
        f = int(0.005 * sr)                      # 5 ms de-click fades
        if seg.size > 2 * f:
            seg[:f] *= np.linspace(0.0, 1.0, f).astype(np.float32)
            seg[-f:] *= np.linspace(1.0, 0.0, f).astype(np.float32)
        masters[spk][pos:end] += seg

        report["rows"].append({"index": row.get("index"), "speaker": spk,
                               "start": round(start, 3),
                               "slot_s": round(avail, 3),
                               "available_s": round(avail, 3),
                               "generated_s": round(gen_s, 3),
                               "speed": round(speed, 3),
                               "mode": mode, "trimmed": trimmed})
        if abs(speed - 1.0) > 1e-3 or i % 25 == 0 or i == len(timeline):
            note = ("" if abs(speed - 1.0) <= 1e-3
                    else f" · {mode} ×{speed:.2f}")
            if trimmed:
                note += " + trimmed"
            log(f"   {i}/{len(timeline)} · row {row.get('index')} [{spk}] "
                f"{gen_s:.2f}s → slot {avail:.2f}s{note}")

    rt_errors = list(rt.errors)
    rt.model = None
    del rt
    log("   " + clear_gpu_cache())

    # ── HARD GUARD: never hand a silent track to Step 8 ───────────────────
    # If the engine produced nothing, the masters below would be all-zero and
    # Step 8 would still "succeed", shipping a final mix with no voices in it.
    # Fail loudly with the accumulated per-row causes instead.
    if not report["rows"]:
        raise TerminalEngineError(engine_error_report(
            f"Step 7 · {eid}",
            [f"0 of {len(timeline)} row(s) produced audio "
             f"({failed_rows} row(s) raised)"] + rt_errors[-6:],
            fixes="Every row failed, so no dubbed track was written.\n"
                  "Re-run with a different engine, or read the detail above."))
    if failed_rows:
        log(f"   ⚠ {failed_rows}/{len(timeline)} row(s) skipped — "
            f"{len(report['rows'])} rendered.")
        for note in rt_errors[-3:]:
            log(f"      {note}")

    for spk, master in masters.items():
        sf.write(str(OUTPUTS_DIR / f"track_{spk.lower()}.wav"),
                 master, sr, subtype="FLOAT")
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["failed_rows"] = failed_rows

    # ── per-line fit accounting (auditable, not just logged) ─────────────────
    rows = report["rows"]
    comp = [r for r in rows if r.get("mode") == "compressed"]
    stretch = [r for r in rows if r.get("mode") == "stretched"]
    trim = [r for r in rows if r.get("trimmed")]
    rates = sorted(r["speed"] for r in rows if r["speed"] > 1.0)
    report["fit"] = {
        "fit_max_speed": fit_max_speed, "fit_min_speed": fit_min_speed,
        "stretch_short": bool(fit_stretch_short),
        "n_rows": len(rows), "n_compressed": len(comp),
        "n_stretched": len(stretch), "n_trimmed": len(trim),
        "max_rate": round(rates[-1], 3) if rates else 1.0,
        "median_rate": round(rates[len(rates) // 2], 3) if rates else 1.0,
    }
    TTS_REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    try:
        _write_csv(LINE_FIT_CSV,
                   ["Row", "Speaker", "Start", "Slot_s", "Generated_s",
                    "Rate", "Mode", "Trimmed"],
                   [[r["index"], r["speaker"], fmt_ts(r["start"]),
                     f"{r['slot_s']:.3f}", f"{r['generated_s']:.3f}",
                     f"{r['speed']:.3f}", r["mode"],
                     "yes" if r["trimmed"] else "no"] for r in rows])
    except Exception as e:
        log(f"   ⚠ row-fit CSV skipped: {type(e).__name__}: {e}")

    log(f"✅ Step 7 [{eid}] → {len(masters)} padded track(s) · "
        f"{len(rows)} row(s) rendered in {report['elapsed_s']}s · "
        f"{len(comp)} compressed · {len(stretch)} stretched · "
        f"{len(trim)} hard-trimmed")
    if report["fit"]["max_rate"] > 1.0:
        log(f"   fit: worst ×{report['fit']['max_rate']:.2f} · "
            f"median ×{report['fit']['median_rate']:.2f} "
            f"(caps ×{fit_max_speed:.2f} … ×{fit_min_speed:.2f}, "
            f"short-line stretch {'ON' if fit_stretch_short else 'off'})")
    return report


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 8 · MASTER MIXDOWN + LOOKAHEAD DUCKING  (numpy + FFmpeg remux)
# ═════════════════════════════════════════════════════════════════════════════

def _resample_1d(data, sr_in: int, sr_out: int):
    """1-D resample — torchaudio when available, numpy-linear fallback else."""
    import numpy as np
    data = np.asarray(data, dtype=np.float32)
    if sr_in == sr_out:
        return data
    try:
        import torch
        import torchaudio
        t = torch.from_numpy(data)[None, :]
        t = torchaudio.functional.resample(t, sr_in, sr_out)
        return t.numpy()[0].astype(np.float32)
    except ImportError:                          # torch-free fallback path
        n_out = int(round(data.size * sr_out / sr_in))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_out = np.linspace(0.0, data.size - 1, n_out)
        return np.interp(x_out, np.arange(data.size, dtype=np.float64),
                         data).astype(np.float32)


def step8_mixdown(log: Log, force: bool = False, duck_db: float = -6.0,
                  lookahead_ms: int = 120, attack_ms: int = 90,
                  release_ms: int = 300, frame_ms: int = 10) -> dict:
    """
    Automated audio mixdown routing engine:
      1. layers every padded per-speaker master directly over one another
      2. imports the Demucs instrumental bed from Step 2
      3. DUCK ENGINE FIX — a lookahead ducking curve attenuates the music by
         `duck_db` (smooth linear attack/release) whenever any voice layer
         carries signal, and releases it back during silence
      4. exports final_mix.wav and — when the original input was a video —
         wraps the master audio behind the original video stream with an
         overwrite copy command (no video re-encode) → final_dubbed.mp4
    """
    import numpy as np
    import soundfile as sf

    result = {"mix": str(FINAL_MIX_WAV), "video": ""}
    if FINAL_MIX_WAV.exists() and not force:
        log("↩ Step 8 mix cached (final_mix.wav) — verifying export only.")
    else:
        track_files = sorted(OUTPUTS_DIR.glob("track_speaker*.wav"))
        if not track_files:
            raise RuntimeError("No padded speaker tracks — run Step 7 first.")
        if not MUSIC_WAV.exists():
            raise RuntimeError("Instrumental bed missing — run Tab 1 (Step 2).")

        t0 = time.time()
        music, sr = sf.read(str(MUSIC_WAV), dtype="float32", always_2d=True)
        n = music.shape[0]
        log(f"🎼 Instrumental bed · {n / sr:.2f}s @ {sr} Hz stereo")

        # ── 1 · layer every speaker track directly over one another ────────
        voices = np.zeros(n, dtype=np.float32)
        for tp in track_files:
            v, vsr = sf.read(str(tp), dtype="float32", always_2d=True)
            v = v.mean(axis=1)
            if vsr != sr:
                v = _resample_1d(v, vsr, sr)
            end = min(n, v.size)
            voices[:end] += v[:end]
            log(f"   ➕ {tp.name} layered")

        # ── 2 · voice-activity envelope (frame_ms resolution) ──────────────
        fl = max(1, int(sr * frame_ms / 1000))
        n_frames = n // fl
        if n_frames == 0:
            raise RuntimeError("Audio too short to mix.")
        frames = voices[: n_frames * fl].reshape(n_frames, fl)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        threshold = max(1e-4, 0.06 * float(rms.max()))
        active = (rms > threshold).astype(np.float32)

        duck_gain = float(10 ** (duck_db / 20.0))     # −6 dB → 0.5012
        target = np.where(active > 0, duck_gain, 1.0).astype(np.float32)

        # ── 3 · smooth linear attack/release (slew-rate limited) ───────────
        af = max(1, int(attack_ms / frame_ms))
        rf = max(1, int(release_ms / frame_ms))
        max_down = (1.0 - duck_gain) / af
        max_up = (1.0 - duck_gain) / rf
        gain = np.empty_like(target)
        g = 1.0
        for i in range(target.size):
            g += float(np.clip(target[i] - g, -max_down, max_up))
            gain[i] = g

        # lookahead pre-duck: shift the curve earlier so the dip lands
        # BEFORE each voice onset
        k = min(max(0, int(lookahead_ms / frame_ms)), target.size - 1)
        if k:
            gain[:-k] = gain[k:]
            gain[-k:] = gain[-k]

        gain_samples = np.repeat(gain, fl)[:n]
        if gain_samples.size < n:
            pad = np.full(n - gain_samples.size, gain[-1], dtype=np.float32)
            gain_samples = np.concatenate([gain_samples, pad])

        final = music * gain_samples[:, None] + voices[:, None]
        peak = float(np.abs(final).max())
        if peak > 0.99:
            final *= 0.99 / peak
            log(f"   🔉 peak normalised ({peak:.2f} → 0.99)")
        sf.write(str(FINAL_MIX_WAV), final.astype(np.float32), sr,
                 subtype="PCM_16")
        log(f"✅ Step 8 → final_mix.wav · duck {duck_db:+.0f} dB · "
            f"lookahead {lookahead_ms} ms · attack {attack_ms} ms · "
            f"release {release_ms} ms · {time.time() - t0:.1f}s")

    # ── 4 · wrap the master audio back into the original video container ──
    state = PipelineState.load()
    src = state.artifacts.get("source_path", "")
    if src and Path(src).suffix.lower() in VIDEO_EXTS and Path(src).exists():
        log("🎥 Original source was video — remuxing (stream copy, no re-encode) …")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", src, "-i", str(FINAL_MIX_WAV),
               "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
               str(FINAL_VIDEO_MP4)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            result["video"] = str(FINAL_VIDEO_MP4)
            log(f"🎬 Final dubbed video → {FINAL_VIDEO_MP4.name}")
        else:
            log(f"⚠ remux failed ({proc.stderr.strip()[-160:]}) — "
                f"audio-only master kept.")
    else:
        log("ℹ source was audio-only — delivering the master mix without video.")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 ORCHESTRATOR — Steps 6 · 7 · 8
# ═════════════════════════════════════════════════════════════════════════════

def run_rendering(force: bool = False,
                  diagnostic: bool = False) -> Generator[str, None, None]:
    """TAB 3 · split → CosyVoice TTS → ducked mixdown → video remux.

    In diagnostic mode, errors are collected instead of halting — dependent
    steps are skipped and a full report prints at the end."""
    log = Log()
    state = PipelineState.load()
    errors: List[Tuple[str, str]] = []
    if diagnostic:
        yield log("🩺 DIAGNOSTIC MODE — errors collected, pipeline runs to finish")
    try:
        yield log("═" * 62)
        yield log(" 🚀 TAB 3 · RENDERING ENGINE — steps 6 · 7 · 8")
        yield log("═" * 62)
        if not FINAL_SCRIPT_JSON.exists():
            raise RuntimeError("Run Tab 2 first — final_script.json missing.")

        # ── VRAM pre-flight ───────────────────────────────────────────────
        # Steps 1–5 tear each model down as they finish, but an aborted or
        # diagnostic-mode run can leave weights resident, and PyTorch's
        # caching allocator never returns cached blocks on its own. Step 7's
        # TTS engine is the single largest allocation of the pipeline, so
        # flush whatever Tabs 1–2 left behind and report it — an OOM during
        # the engine load otherwise surfaces as a dead process, which the
        # browser can only report as "Connection to the server was lost".
        yield log("🧹 VRAM pre-flight — releasing anything Tabs 1–2 left behind")
        yield log("   " + clear_gpu_cache())
        yield log("   " + log_memory())

        # ── Step 6 · split speaker scripts ─────────────────────────────────
        yield log("─" * 62)
        scripts_ok = False
        try:
            step6_split_speaker_scripts(log, force=force)
            state.mark("step6")
            state.save()
            scripts_ok = True
        except Exception as e:
            if diagnostic:
                errors.append(("Step 6 · split_speaker_scripts", str(e)))
                yield log(f"❌ Step 6 failed: {e}")
            else:
                raise

        # ── Step 7 · CosyVoice TTS (depends on Step 6) ─────────────────────
        if scripts_ok:
            yield log("─" * 62)
            try:
                step7_synthesize(log, force=force)
                state.mark("step7")
                state.save()
                yield log("   " + log_memory())
            except Exception as e:
                if diagnostic:
                    errors.append((f"Step 7 · {get_tts_engine()}", str(e)))
                    yield log(f"❌ Step 7 failed: {e}")
                else:
                    raise
        else:
            yield log(f"⏭ Step 7 · {get_tts_engine()} SKIPPED (Step 6 failed)")

        # ── Step 8 · mixdown (depends on Steps 2, 6, 7) ────────────────────
        if scripts_ok and TTS_REPORT_JSON.exists():
            yield log("─" * 62)
            try:
                result = step8_mixdown(log, force=force)
                state.mark("step8",
                           output="video" if result.get("video") else "audio-only")
                state.save()
                yield log("🏁 Render complete — download the master below.")
            except Exception as e:
                if diagnostic:
                    errors.append(("Step 8 · mixdown", str(e)))
                    yield log(f"❌ Step 8 failed: {e}")
                else:
                    raise
        else:
            yield log("⏭ Step 8 · mixdown SKIPPED (Step 6/7 failed)")

        # ── diagnostic report ──────────────────────────────────────────────
        if diagnostic and errors:
            yield log(_render_diagnostic(errors))
    except Exception as e:
        yield log(f"❌ Step failed: {e}")
        yield log("   Pipeline halted — fix the issue and re-run "
                  "(cached steps auto-skip).")
    finally:
        state.save()
