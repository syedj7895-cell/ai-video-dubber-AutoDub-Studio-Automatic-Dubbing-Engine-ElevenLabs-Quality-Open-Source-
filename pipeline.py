# ═══════════════════════════════════════════════════════════════════════════
#   AUTOMATIC DUBBING ENGINE — pipeline.py
#   Core file-routing layer · sequential GPU execution · OOM defense
#   Target runtime: Google Colab Free T4 (16 GB VRAM) — CPU-safe fallbacks
# ═══════════════════════════════════════════════════════════════════════════
#
#   EXECUTION MAP
#   ────────────
#     Step 1 · extract_audio        FFmpeg / MoviePy ......... CPU
#     Step 2 · separate_vocals      Demucs v4 htdemucs ....... GPU → flush
#     Step 3 · diarize + mine       Pyannote 3.1 ............. GPU → flush
#     Step 4 · emotion scan         SenseVoice-Small ......... GPU → flush
#     Step 5 · assemble script      pysrt merge .............. CPU
#
#   DESIGN RULE — exactly ONE heavy model lives on the GPU at any moment.
#   After each model finishes, `clear_gpu_cache()` releases the model,
#   garbage-collects, and empties the CUDA cache before the next stage
#   boots. This is the defense against Colab T4 Out-Of-Memory kills.
#
#   All heavy libraries (torch / demucs / pyannote / funasr) are imported
#   LAZILY inside their step functions, so:
#     · importing this module is instant and dependency-light,
#     · the UI launches even on machines without torch,
#     · no GPU memory is touched until a step actually runs.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
import time
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
STATE_JSON = OUTPUTS_DIR / "state.json"                   # pipeline state

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

# SenseVoice language tag → readable name (bonus metadata for the grid)
LANG_MAP = {"zh": "Chinese", "en": "English", "yue": "Cantonese",
            "ja": "Japanese", "ko": "Korean"}

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

def step2_separate_vocals(log: Log, force: bool = False) -> Tuple[Path, Path]:
    """
    Load Demucs v4 (htdemucs), split the master audio into a clean vocal
    stem and a background instrumental stem, write both to disk as .wav,
    then IMMEDIATELY tear the model down with clear_gpu_cache().
    """
    if VOCALS_WAV.exists() and MUSIC_WAV.exists() and not force:
        log("↩ Step 2 cached (vocals.wav + music.wav) — skipping.")
        return VOCALS_WAV, MUSIC_WAV

    try:
        import torch
        import soundfile as sf
        from demucs.api import Separator
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Step 2 needs the ML stack — module '{missing}' is not installed here. "
            "Local CPU fix:  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu  then  "
            "pip install demucs  ·  Or run on Google Colab (T4) where the "
            "stack ships preinstalled.") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"🧠 Loading Demucs v4 · model=htdemucs · device={device}")
    log("   " + log_memory())

    t0 = time.time()
    # shifts=0 → single pass (2× faster, T4-friendly); split=True keeps VRAM flat
    sep = Separator(model="htdemucs", device=device, shifts=0,
                    overlap=0.25, split=True, progress=False)
    log("🎚 Separating stems (chunked streaming — OOM-safe) …")
    _wav, sources = sep.separate_audio_file(str(AUDIO_WAV))
    sr = sep.samplerate  # 44 100 Hz

    vocals = sources["vocals"]
    music = sources["drums"].cpu() + sources["bass"].cpu() + sources["other"].cpu()
    sf.write(str(VOCALS_WAV), vocals.cpu().numpy().T, sr)   # (frames, channels)
    sf.write(str(MUSIC_WAV), music.numpy().T, sr)

    dur = _wav_duration(VOCALS_WAV)
    # 🔥 full model teardown before anything else loads
    del sources, vocals, music, sep
    log("   " + clear_gpu_cache())
    log(f"✅ Step 2 → vocals.wav + music.wav  ({dur:.1f}s in {time.time() - t0:.1f}s)")
    return VOCALS_WAV, MUSIC_WAV


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 · SPEAKER DIARIZATION + CLONE-PROMPT MINING  (Pyannote 3.1)
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


def step3_diarization(hf_token: Optional[str],
                      original_srt_path: Optional[str],
                      log: Log,
                      force: bool = False) -> dict:
    """
    Pyannote 3.1 diarization of the clean vocal track, cross-referenced with
    the user's ORIGINAL SRT timeline (max-overlap speaker vote per cue), then
    clone-prompt mining + full GPU teardown.
    """
    if DIARIZATION_JSON.exists() and not force:
        log("↩ Step 3 cached (diarization_map.json) — skipping.")
        return json.loads(DIARIZATION_JSON.read_text(encoding="utf-8"))

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Step 3 needs the ML stack — module '{missing}' is not installed here. "
            "Local CPU fix:  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu  then  "
            "pip install pyannote.audio  ·  Or run on Google Colab (T4).") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"🧠 Loading Pyannote 3.1 · {PYANNOTE_MODEL} · device={device}")

    kwargs = {"use_auth_token": hf_token.strip()} if (hf_token or "").strip() else {}
    try:
        pipe = Pipeline.from_pretrained(PYANNOTE_MODEL, **kwargs)
    except Exception as e:
        raise RuntimeError(
            "Pyannote failed to load. Checklist: ① visit huggingface.co/"
            "pyannote/speaker-diarization-3.1 AND …/segmentation-3.1 and accept "
            "the user conditions with the SAME account that owns the token; "
            "② paste the token in Tab 1 ▸ Advanced. Detail: " + str(e)[:300]
        ) from e

    pipe.to(torch.device(device))
    log("🎙 Diarizing the isolated vocal track …")
    t0 = time.time()
    diar = pipe(str(VOCALS_WAV))
    turns = [{"start": float(t.start), "end": float(t.end), "raw": label}
             for t, _, label in diar.itertracks(yield_label=True)]
    del pipe, diar
    log("   " + clear_gpu_cache())

    if not turns:
        raise RuntimeError("Diarization returned zero speaker turns.")
    n_raw = len({t["raw"] for t in turns})
    log(f"   {len(turns)} raw turns · {n_raw} distinct voices · {time.time() - t0:.1f}s")

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

    result = {"model": PYANNOTE_MODEL, "speakers": speakers, "cues": cue_rows}
    DIARIZATION_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    log(f"✅ Step 3 → diarization_map.json · {len(speakers)} identity tag(s) mapped")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 · PARALINGUISTIC EMOTION SCAN  (Alibaba SenseVoice-Small / FunASR)
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
    Run SenseVoice-Small EXCLUSIVELY over the timeline-matched voice segments
    (the diarized SRT cue windows) to tag each with one of the 7 canonical
    emotions. Produces:
      · outputs/emotion_log.txt   — spec format:  [00:50.000] Speaker1 [calm]
      · outputs/emotion_grid.json — machine grid consumed by Step 5
    """
    if EMOTION_GRID_JSON.exists() and not force:
        log("↩ Step 4 cached (emotion_grid.json) — skipping.")
        return json.loads(EMOTION_GRID_JSON.read_text(encoding="utf-8"))

    try:
        import torch
        from funasr import AutoModel
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Step 4 needs the ML stack — module '{missing}' is not installed here. "
            "Local CPU fix:  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu  then  "
            "pip install funasr modelscope  ·  Or run on Google Colab (T4).") from e

    diar = json.loads(DIARIZATION_JSON.read_text(encoding="utf-8"))
    cues = diar["cues"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"🧠 Loading SenseVoice-Small · {SENSEVOICE_MODEL} · device={device}")
    log("   " + log_memory())

    t0 = time.time()
    model = AutoModel(model=SENSEVOICE_MODEL,
                      vad_model="fsmn-vad",
                      vad_kwargs={"max_single_segment_time": 30_000},
                      device=device,
                      disable_update=True,
                      disable_pbar=True,
                      trust_remote_code=True)

    mono, sr = _load_vocals_16k()
    tag_re = re.compile(r"<\|([A-Z_]+)\|>")
    grid: List[dict] = []
    fails = 0

    for cue in cues:
        a = float(cue["start"])
        b = max(float(cue["end"]), a + 0.2)          # sub-0.2 s guard
        seg = mono[int(a * sr): int(b * sr)]
        if seg.size < int(0.25 * sr):                # < 0.25 s → no signal
            emotion, lang = "neutral", "en"
        else:
            try:
                res = model.generate(input=seg, cache={},
                                     language="auto", use_itn=False,
                                     merge_vad=False)
                raw = res[0].get("text", "") if res else ""
                tags = tag_re.findall(raw)
                emotion = next((SENSEVOICE_EMO_MAP[t] for t in tags
                                if t in SENSEVOICE_EMO_MAP), "neutral")
                lang = next((t for t in tags if t in LANG_MAP), "en")
            except Exception:
                emotion, lang, fails = "neutral", "en", fails + 1
        grid.append({"index": cue["index"], "start": a, "end": b,
                     "speaker": cue.get("speaker", "Speaker1"),
                     "emotion": emotion, "language": lang})

    EMOTION_LOG_TXT.write_text(
        "\n".join(f"[{fmt_ts(g['start'])}] {g['speaker']} [{g['emotion']}]"
                  for g in grid),
        encoding="utf-8")
    EMOTION_GRID_JSON.write_text(json.dumps(grid, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    # bonus (feeds Step 7) · transcribe each speaker's cleanest clone prompt
    # so CosyVoice zero-shot cloning gets its exact conditioning transcript.
    try:
        prompt_rels = {spk: (info.get("clone_prompts") or [""])[0]
                       for spk, info in diar.get("speakers", {}).items()
                       if (info.get("clone_prompts") or [""])[0]}
        if prompt_rels:
            log("🈳 Transcribing clone prompts (zero-shot conditioning text) …")
            ptexts: Dict[str, str] = {}
            for spk, rel in sorted(prompt_rels.items()):
                try:
                    res = model.generate(input=str(BASE_DIR / rel), cache={},
                                         language="auto", use_itn=False)
                    raw = res[0].get("text", "") if res else ""
                    ptexts[spk] = tag_re.sub("", raw).strip()
                    log(f"   {spk}: '{ptexts[spk][:48]}'")
                except Exception as e:
                    ptexts[spk] = ""
                    log(f"   ⚠ {spk}: transcription failed ({str(e)[:80]})")
            PROMPT_TRANSCRIPTS_JSON.write_text(
                json.dumps(ptexts, indent=2, ensure_ascii=False),
                encoding="utf-8")
    except Exception as e:
        log(f"   ⚠ prompt transcription skipped: {e}")

    del model
    log("   " + clear_gpu_cache())

    dist = Counter(g["emotion"] for g in grid)
    log(f"✅ Step 4 → emotion_log.txt · {len(grid)} segments "
        f"in {time.time() - t0:.1f}s" + (f" · {fails} decode fallbacks" if fails else ""))
    log("   distribution: " + ", ".join(f"{k}×{v}" for k, v in dist.most_common()))
    return grid


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5 · SRT MAPPING & CONSOLIDATED SCRIPT ASSEMBLY  (pysrt)
# ═════════════════════════════════════════════════════════════════════════════

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
            "clone_prompt": prompts[0] if prompts else "",
            "original_text": oc.get("text", ""),
            "translated_text": tc["text"],
        })

    FINAL_SCRIPT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    log(f"✅ Step 5 → final_script.json · {len(rows)} consolidated row(s)")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATORS — generator functions that stream a cumulative console log
#  to the Gradio UI while enforcing strict sequential execution.
# ═════════════════════════════════════════════════════════════════════════════

def run_import_and_analysis(media_path: str,
                            force: bool = False) -> Generator[str, None, None]:
    """TAB 1 · Steps 1–2: extract audio → Demucs vocal/music split."""
    log = Log()
    state = PipelineState.load()
    src_name = Path(media_path).name
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

        yield log("─" * 62)
        audio = step1_extract_audio(media_path, log, force=force)
        state.mark("step1", audio=audio.name)
        state.save()
        yield log("   " + log_memory())

        yield log("─" * 62)
        vocals, music = step2_separate_vocals(log, force=force)
        state.mark("step2", vocals=vocals.name, music=music.name)
        state.save()
        yield log("🎬 Analysis complete — switch to Tab 2 for speaker matching.")
    except Exception as e:
        yield log(f"❌ Step failed: {e}")
        yield log("   Pipeline halted — fix the issue and re-run "
                  "(cached steps auto-skip).")
    finally:
        state.save()


def run_script_matching(hf_token: Optional[str],
                        original_srt_path: Optional[str],
                        translated_srt_path: Optional[str],
                        force: bool = False) -> Generator[str, None, None]:
    """TAB 2 · Steps 3–5: diarization → emotion scan → script assembly."""
    log = Log()
    state = PipelineState.load()
    try:
        yield log("═" * 62)
        yield log(" 🧬 TAB 2 · SCRIPT MATCHING & ASSEMBLY — steps 3 · 4 · 5")
        yield log("═" * 62)
        if not VOCALS_WAV.exists():
            raise RuntimeError("Run Tab 1 first — outputs/vocals.wav not found.")

        yield log("─" * 62)
        diar = step3_diarization(hf_token, original_srt_path, log, force=force)
        state.mark("step3", speakers=str(len(diar.get("speakers", {}))))
        state.save()
        yield log("   " + log_memory())

        yield log("─" * 62)
        step4_emotion_analysis(log, force=force)
        state.mark("step4")
        state.save()
        yield log("   " + log_memory())

        yield log("─" * 62)
        if not translated_srt_path or not Path(translated_srt_path).exists():
            raise RuntimeError("Translated SRT missing — upload it in Tab 1 "
                               "to assemble the dubbing script.")
        rows = step5_assemble_script(translated_srt_path, log, force=force)
        state.mark("step5", rows=str(len(rows)))
        state.save()
        yield log("🧬 Script assembled — switch to Tab 3 to verify render readiness.")
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
            "language": r.get("language", "en"),
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

def _load_cosyvoice(log: Log):
    """
    Lazy CosyVoice 3.0 loader (zero-shot mode). The engine ships inside the
    FunAudioLLM repo, so operators get a one-line bootstrap when missing.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "Step 7 needs the ML stack — 'torch' is not installed here. "
            "Local CPU fix:  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu  ·  Or run on "
            "Google Colab (T4).") from e
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3 as _CV        # CosyVoice 3.0
    except ImportError:
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as _CV    # 2.x fallback
        except ImportError:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice as _CV
            except ImportError as e:
                raise RuntimeError(
                    "CosyVoice engine not found. Bootstrap once with:\n"
                    "  git clone https://github.com/FunAudioLLM/CosyVoice && "
                    "cd CosyVoice && git submodule update --init --recursive && "
                    "pip install -r requirements.txt && pip install -e .\n"
                    "…then re-run the render.") from e

    from modelscope import snapshot_download
    model_dir = None
    for repo in ("iic/CosyVoice3-0.5B", "iic/CosyVoice2-0.5B"):
        try:
            model_dir = snapshot_download(repo)
            break
        except Exception:
            continue
    if model_dir is None:
        raise RuntimeError("CosyVoice checkpoint download failed "
                           "(check network / ModelScope access).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"🧠 Loading CosyVoice · {Path(model_dir).name} · device={device}")
    log("   " + log_memory())
    try:
        return _CV(model_dir, load_trt=False, fp16=torch.cuda.is_available())
    except TypeError:                       # older constructor without trt/jit
        return _CV(model_dir)


def _load_prompt_speech(path: str, target_sr: int = 16_000):
    """Clone-prompt wav → torch tensor (1, n) @16 kHz (CosyVoice expects 16k)."""
    import torch
    import torchaudio
    wav, sr = torchaudio.load(str(path))                 # (channels, n)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _ensure_prompt_transcripts(log: Log) -> Dict[str, str]:
    """
    speaker → transcript of its best clone prompt (zero-shot conditioning).
    Normally produced during Step 4; re-derives with a quick SenseVoice pass
    when only the cached path was taken.
    """
    if PROMPT_TRANSCRIPTS_JSON.exists():
        return json.loads(PROMPT_TRANSCRIPTS_JSON.read_text(encoding="utf-8"))
    diar = json.loads(DIARIZATION_JSON.read_text(encoding="utf-8"))
    todo = {spk: (info.get("clone_prompts") or [""])[0]
            for spk, info in diar.get("speakers", {}).items()
            if (info.get("clone_prompts") or [""])[0]}
    if not todo:
        return {}

    log("🈳 Transcribing clone prompts for zero-shot conditioning (SenseVoice) …")
    import torch
    from funasr import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel(model=SENSEVOICE_MODEL, vad_model="fsmn-vad",
                      vad_kwargs={"max_single_segment_time": 30_000},
                      device=device, disable_update=True, disable_pbar=True,
                      trust_remote_code=True)
    tag_re = re.compile(r"<\|[^|>]*\|>")
    out: Dict[str, str] = {}
    for spk, rel in sorted(todo.items()):
        try:
            res = model.generate(input=str(BASE_DIR / rel), cache={},
                                 language="auto", use_itn=False)
            raw = res[0].get("text", "") if res else ""
            out[spk] = tag_re.sub("", raw).strip()
            log(f"   {spk}: '{out[spk][:48]}'")
        except Exception as e:
            out[spk] = ""
            log(f"   ⚠ {spk}: transcription failed ({str(e)[:80]})")
    del model
    log("   " + clear_gpu_cache())
    PROMPT_TRANSCRIPTS_JSON.write_text(json.dumps(out, indent=2,
                                                  ensure_ascii=False),
                                       encoding="utf-8")
    return out


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
    if instruct and hasattr(model, "inference_instruct2"):
        attempts.append(lambda: model.inference_instruct2(
            text, instruct, prompt_speech, stream=False))
    if prompt_text:
        attempts.append(lambda: model.inference_zero_shot(
            text, prompt_text, prompt_speech, stream=False))
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


def step7_synthesize(log: Log, force: bool = False,
                     max_speedup: float = 1.15) -> dict:
    """
    CosyVoice 3.0 zero-shot synthesis onto per-speaker silent master timelines:
      · one master numpy ZERO-SIGNAL array per speaker, exactly matching the
        duration of the original media (the master audio clock)
      · every row is cloned with the speaker's clean voice profile + emotion
        paralinguistic tags (English · Hindi · Spanish)
      · OVERPRESSURE FIX — if a generated clip would bleed into the next
        sequential line, it is time-stretched with librosa (non-pitch-shifting,
        up to 1.15×) so it snaps inside its slot without moving the clock.
    """
    import numpy as np
    import soundfile as sf

    track_files = sorted(OUTPUTS_DIR.glob("track_speaker*.wav"))
    if TTS_REPORT_JSON.exists() and track_files and not force:
        log("↩ Step 7 cached — skipping synthesis (force to redo).")
        return json.loads(TTS_REPORT_JSON.read_text(encoding="utf-8"))

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
    model = _load_cosyvoice(log)
    sr = int(getattr(model, "sample_rate", 24_000))

    # ── the auto-padding system: silent zero-signal master per speaker ────
    n_total = int(round(duration * sr))
    masters: Dict[str, np.ndarray] = {
        spk: np.zeros(n_total, dtype=np.float32) for spk in scripts}
    log(f"🎚 {len(masters)} silent master track(s) · {duration:.2f}s @ {sr} Hz each")

    # global chronological order → 'next sequential line start' for any row
    timeline = sorted(({"spk": spk, **l} for spk, lines in scripts.items()
                       for l in lines),
                      key=lambda d: (d["start"], d.get("index") or 0))
    speakers_info = json.loads(DIARIZATION_JSON.read_text(encoding="utf-8")) \
        .get("speakers", {})
    transcripts = _ensure_prompt_transcripts(log)

    report = {"sample_rate": sr, "duration_s": round(duration, 3), "rows": []}
    t0 = time.time()
    for i, row in enumerate(timeline, 1):
        spk = row["spk"]
        start = float(row["start"])
        # time slice available before the NEXT sequential line starts
        avail = ((float(timeline[i]["start"]) - start) if i < len(timeline)
                 else (duration - start))
        avail = max(avail, 0.5)                  # never less than ½ s of room
        max_samples = int(avail * sr)

        info = speakers_info.get(spk, {})
        prompt_rel = (info.get("clone_prompts") or [""])[0]
        if not prompt_rel:
            log(f"   ⚠ row {row.get('index')} — no clone prompt for {spk}; skipped.")
            continue

        try:
            prompt_speech = _load_prompt_speech(str(BASE_DIR / prompt_rel))
            clip, _ = _cosyvoice_speak(model, row["text"],
                                       f"Speak {row['instruct']}.",
                                       prompt_speech, transcripts.get(spk, ""))
        except Exception as e:
            log(f"   ⚠ row {row.get('index')} synthesis failed: {str(e)[:120]}")
            continue

        gen_s = clip.size / sr
        speed, trimmed = 1.0, False
        if clip.size > max_samples:
            # overpressure fix · non-pitch-shifting time stretch (≤ 1.15×)
            speed = float(min(max_speedup, clip.size / max_samples))
            clip = librosa.effects.time_stretch(y=clip, rate=speed)
            if clip.size > max_samples:          # last resort: hard snap + fade
                fade = int(0.03 * sr)
                clip = clip[:max_samples]
                if clip.size > fade:
                    clip[-fade:] *= np.linspace(1.0, 0.0, fade).astype(np.float32)
                trimmed = True

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
                               "available_s": round(avail, 3),
                               "generated_s": round(gen_s, 3),
                               "speed": round(speed, 3), "trimmed": trimmed})
        if speed > 1.0 or i % 25 == 0 or i == len(timeline):
            note = (f" · stretched ×{speed:.2f}" + (" + trimmed" if trimmed else "")
                    if speed > 1.0 else "")
            log(f"   {i}/{len(timeline)} · row {row.get('index')} [{spk}] "
                f"{gen_s:.2f}s → slot {avail:.2f}s{note}")

    del model
    log("   " + clear_gpu_cache())

    for spk, master in masters.items():
        sf.write(str(OUTPUTS_DIR / f"track_{spk.lower()}.wav"),
                 master, sr, subtype="FLOAT")
    report["elapsed_s"] = round(time.time() - t0, 1)
    TTS_REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    stretched = sum(1 for r in report["rows"] if r["speed"] > 1.0)
    log(f"✅ Step 7 → {len(masters)} padded track(s) · "
        f"{len(report['rows'])} row(s) rendered in {report['elapsed_s']}s · "
        f"{stretched} overlap-protection stretch(es)")
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

def run_rendering(force: bool = False) -> Generator[str, None, None]:
    """TAB 3 · split → CosyVoice TTS → ducked mixdown → video remux."""
    log = Log()
    state = PipelineState.load()
    try:
        yield log("═" * 62)
        yield log(" 🚀 TAB 3 · RENDERING ENGINE — steps 6 · 7 · 8")
        yield log("═" * 62)
        if not FINAL_SCRIPT_JSON.exists():
            raise RuntimeError("Run Tab 2 first — final_script.json missing.")

        yield log("─" * 62)
        step6_split_speaker_scripts(log, force=force)
        state.mark("step6")
        state.save()

        yield log("─" * 62)
        step7_synthesize(log, force=force)
        state.mark("step7")
        state.save()
        yield log("   " + log_memory())

        yield log("─" * 62)
        result = step8_mixdown(log, force=force)
        state.mark("step8", output="video" if result.get("video") else "audio-only")
        state.save()
        yield log("🏁 Render complete — download the master below.")
    except Exception as e:
        yield log(f"❌ Step failed: {e}")
        yield log("   Pipeline halted — fix the issue and re-run "
                  "(cached steps auto-skip).")
    finally:
        state.save()
