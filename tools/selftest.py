# ═══════════════════════════════════════════════════════════════════════════
#  SELF-TEST — pure-Python layers of the dubbing pipeline (no GPU / models)
#  Validates: timestamp format · pysrt parsing · interval math · orchestrator
#  guards · Step-5 assembly · TTS sanitiser · Step-6 splitting · Step-8
#  mixdown with lookahead ducking (synthetic audio).
#
#  Run:  python tools/selftest.py
# ═══════════════════════════════════════════════════════════════════════════

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline  # noqa: E402

results = []


def check(name: str, cond: bool) -> None:
    results.append(bool(cond))
    print(f"{'✅' if cond else '❌'} {name}")


def skip(name: str, why: str) -> None:
    """Report a check that cannot run in this build — deliberately NOT counted
    as a pass, so an obfuscated build can never inflate the score."""
    print(f"⏭  {name}  ({why})")


# ── 1 · timestamp formatting (spec log format) ──────────────────────────────
check("fmt_ts(50.0) == '00:50.000'", pipeline.fmt_ts(50.0) == "00:50.000")
check("fmt_ts rolls hours (01:02:03.500)", pipeline.fmt_ts(3723.5) == "01:02:03.500")

# ── 2 · SRT parsing via pysrt ───────────────────────────────────────────────
sample = ("1\n00:00:01,000 --> 00:00:03,500\nHello there.\n\n"
          "2\n00:00:04,000 --> 00:00:06,000\nGeneral Kenobi!\n")
srt_path = pipeline.OUTPUTS_DIR / "_selftest_orig.srt"
srt_path.write_text(sample, encoding="utf-8")
cues = pipeline.load_srt(srt_path)
check("pysrt parsed 2 cues", len(cues) == 2)
check("cue1 start == 1.0 s", abs(cues[0]["start"] - 1.0) < 1e-6)
check("cue2 text preserved", cues[1]["text"] == "General Kenobi!")

# ── 3 · interval math (speaker cross-referencing primitives) ────────────────
check("overlap math 0-5 ∩ 3-8 == 2", pipeline._overlap(0, 5, 3, 8) == 2.0)
check("merge intervals", pipeline._merge_intervals(
    [(0, 2), (2, 5), (9, 10)]) == [[0, 5], [9, 10]])

# ── 4 · orchestrator guard: Tab 2 refused before Tab 1 ─────────────────────
captured = list(pipeline.run_script_matching("", None, None))
check("Tab-2 guard fires without Tab-1 artifacts",
      any("Run Tab 1 first" in ln for ln in captured))

# ── 5 · Step 5 assembly over a synthetic diarization + emotion grid ────────
diar = {
    "model": "selftest",
    "speakers": {"Speaker1": {
        "raw_label": "SPEAKER_00", "total_speech_s": 4.5, "cue_count": 2,
        "clone_prompts": ["outputs/speaker1_clone_prompt.wav"]}},
    "cues": [
        {"index": 1, "start": 1.0, "end": 3.5, "speaker": "Speaker1",
         "raw": "SPEAKER_00", "text": "Hello there."},
        {"index": 2, "start": 4.0, "end": 6.0, "speaker": "Speaker1",
         "raw": "SPEAKER_00", "text": "General Kenobi!"},
    ],
}
grid = [
    {"index": 1, "start": 1.0, "end": 3.5, "speaker": "Speaker1",
     "emotion": "happy", "language": "en"},
    {"index": 2, "start": 4.0, "end": 6.0, "speaker": "Speaker1",
     "emotion": "angry", "language": "en"},
]
pipeline.DIARIZATION_JSON.write_text(json.dumps(diar), encoding="utf-8")
pipeline.EMOTION_GRID_JSON.write_text(json.dumps(grid), encoding="utf-8")

trans = ("1\n00:00:01,000 --> 00:00:03,500\nBonjour.\n\n"
         "2\n00:00:04,000 --> 00:00:06,000\nKenobi !\n")
tsrt = pipeline.OUTPUTS_DIR / "_selftest_trans.srt"
tsrt.write_text(trans, encoding="utf-8")

rows = pipeline.step5_assemble_script(str(tsrt), pipeline.Log(), force=True)
check("step5 assembled 2 rows", len(rows) == 2)
check("row1 speaker mapped", rows[0]["speaker"] == "Speaker1")
check("row1 emotion merged", rows[0]["emotion"] == "happy")
check("row2 emotion merged", rows[1]["emotion"] == "angry")
check("row1 translated text", rows[0]["translated_text"] == "Bonjour.")
check("row1 instruct hint", "happy" in rows[0]["instruct"])
check("row1 clone prompt linked",
      rows[0]["clone_prompt"].endswith("speaker1_clone_prompt.wav"))
check("final_script.json written", pipeline.FINAL_SCRIPT_JSON.exists())

# ── 6 · TTS sanitiser (labels & timecodes never reach the voice) ───────────
dirty = ("[00:12.500] Speaker1: <|en|><|HAPPY|> 1\n00:00:01,000 --> 00:00:03,500 "
         "Bonjour <b>le</b> monde!")
clean = pipeline.sanitize_for_tts(dirty)
check("sanitiser removes timecodes", "00:00" not in clean and "12.500" not in clean)
check("sanitiser removes speaker labels", "Speaker" not in clean)
check("sanitiser removes rich tags", "<|" not in clean and "<b>" not in clean)
check("sanitiser keeps spoken words", "Bonjour" in clean and "monde" in clean)

# ── 7 · Step 6 splitting (per-speaker, TTS-safe channels) ──────────────────
dirty_rows = [
    {"index": 1, "start": 1.0, "end": 3.5, "speaker": "Speaker1",
     "emotion": "happy", "instruct": "in a happy, upbeat tone", "language": "en",
     "clone_prompt": "outputs/speaker1_clone_prompt.wav",
     "original_text": "a", "translated_text": "[happy] Speaker1: Bonjour le monde!"},
    {"index": 2, "start": 4.0, "end": 6.0, "speaker": "Speaker2",
     "emotion": "angry", "instruct": "in an angry, tense tone", "language": "en",
     "clone_prompt": "outputs/speaker2_clone_prompt.wav",
     "original_text": "b", "translated_text": "00:04.000 General Kenobi!"},
    {"index": 3, "start": 6.5, "end": 8.0, "speaker": "Speaker1",
     "emotion": "neutral", "instruct": "in a calm, neutral tone", "language": "en",
     "clone_prompt": "outputs/speaker1_clone_prompt.wav",
     "original_text": "c", "translated_text": "Au revoir."},
]
pipeline.FINAL_SCRIPT_JSON.write_text(json.dumps(dirty_rows), encoding="utf-8")
scripts = pipeline.step6_split_speaker_scripts(pipeline.Log(), force=True)
check("step6 produced 2 speaker channels", sorted(scripts) == ["Speaker1", "Speaker2"])
p1 = (pipeline.OUTPUTS_DIR / "Project_Speaker1.txt").read_text(encoding="utf-8")
p2 = (pipeline.OUTPUTS_DIR / "Project_Speaker2.txt").read_text(encoding="utf-8")
check("Project_Speaker1.txt exists with 2 lines", len(p1.splitlines()) == 2)
check("Project txt has NO labels/timecodes",
      "Speaker" not in p1 and "Speaker" not in p2
      and "[happy]" not in p1 and "00:04" not in p2)
check("txt keeps pure spoken lines", "Au revoir." in p1 and "Kenobi!" in p2)
check("speaker_scripts.json structure", scripts["Speaker1"][0]["emotion"] == "happy")

# ── 8 · Tab-3 orchestrator guard ────────────────────────────────────────────
pipeline.FINAL_SCRIPT_JSON.unlink(missing_ok=True)   # restore no-artifact state
captured = list(pipeline.run_rendering(force=True))
check("Tab-3 guard fires without prerequisites",
      any("Run Tab 2 first" in ln for ln in captured))

# ── 9 · Step 8 mixdown + lookahead ducking (synthetic audio, torch-free) ───
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = 44_100
dur = 2.0
t_axis = np.linspace(0.0, dur, int(SR * dur), endpoint=False)
# loud music bed (0.5) + quiet voice (0.05) ⇒ total-RMS drop during voice
# proves the −6 dB duck curve engaged
music = (0.50 * np.sin(2 * np.pi * 220.0 * t_axis))[:, None] * np.ones((1, 2))
music = music.astype(np.float32)
sf.write(str(pipeline.MUSIC_WAV), music, SR, subtype="FLOAT")

# voice burst 0–0.9 s, saved at 16 kHz to exercise the resample path
tv = np.linspace(0.0, 0.9, int(16_000 * 0.9), endpoint=False)
voice = (0.05 * np.sin(2 * np.pi * 300.0 * tv)).astype(np.float32)
track = pipeline.OUTPUTS_DIR / "track_speaker1.wav"
sf.write(str(track), voice, 16_000, subtype="FLOAT")

result = pipeline.step8_mixdown(pipeline.Log(), force=True)
check("step8 wrote final_mix.wav", pipeline.FINAL_MIX_WAV.exists())
check("step8 reports audio-only delivery",
      result.get("mix", "").endswith("final_mix.wav")
      and result.get("video", "") == "")

mix, msr = sf.read(str(pipeline.FINAL_MIX_WAV), dtype="float32", always_2d=True)
def _rms(seg):
    return float(np.sqrt((seg ** 2).mean()) + 1e-12)
during_voice = mix[int(0.25 * msr):int(0.85 * msr)]     # ducked music + voice
after_voice = mix[int(1.30 * msr):int(1.90 * msr)]      # music back at unity
check("ducking engaged during voice (−6 dB floor)",
      _rms(during_voice) < _rms(after_voice) * 0.95)
check("music recovered after silence (unity gain region)",
      _rms(after_voice) > 0.25)

# ── 10 · terminal-engine contract — no fallback ⇒ loud, detailed failure ───
check("cosyvoice2 still degrades to edge",
      pipeline.engine_fallback("cosyvoice2") == "edge")
for _eid in ("cosyvoice3", "chatterbox", "fishs2", "edge"):
    check(f"{_eid} is terminal",
          pipeline.engine_fallback(_eid) == ""
          and pipeline.is_terminal_engine(_eid))
check("cosyvoice2 is not terminal",
      not pipeline.is_terminal_engine("cosyvoice2"))

elog = pipeline.ENGINE_ERROR_LOG
elog.unlink(missing_ok=True)
rep = pipeline.engine_error_report(
    "UnitTest Engine", ["[import] ImportError: boom"], detail="TRACEBACK-MARKER")
check("error report names the engine + terminal state",
      "UnitTest Engine" in rep and "TERMINAL" in rep)
check("error report lists every cause in order",
      "[import] ImportError: boom" in rep)
check("error report fingerprints the live environment",
      "python " in rep and "numpy" in rep and "torch" in rep)
check("full traceback persisted to outputs/engine_error.log",
      elog.exists() and "TRACEBACK-MARKER" in elog.read_text(encoding="utf-8"))


class _BoomEngine:
    """A loader/speaker that always fails — a stand-in for a broken engine."""

    @staticmethod
    def load(log):
        raise RuntimeError("simulated weights-not-found")

    @staticmethod
    def speak(model, text, **kw):
        raise RuntimeError("simulated synthesis failure")


_terminal_spec = {"id": "_unit_terminal", "label": "UnitTest Terminal",
                  "kind": "unit", "fallback": "",
                  "load": _BoomEngine.load, "speak": _BoomEngine.speak}

rt_term = pipeline.EngineRuntime(_terminal_spec, pipeline.Log())
try:
    rt_term.load()
    check("terminal engine load failure raises", False)
except pipeline.TerminalEngineError as exc:
    check("terminal engine load failure raises TerminalEngineError", True)
    check("...and the report carries the real cause",
          "weights-not-found" in str(exc))

rt_term.model = object()
abort_msg = ""
for _ in range(4):
    try:
        rt_term.speak("Speaker1", {"index": 7, "text": "namaste"}, "", "", {})
    except pipeline.TerminalEngineError as exc:
        abort_msg = str(exc)
        break
    except Exception:
        pass
check("terminal engine aborts after 3 consecutive row failures",
      bool(abort_msg) and rt_term.failed >= 3)
check("abort report names the failing row", "[row 7" in abort_msg)
check("abort report says how many failures accumulated",
      "consecutive failures" in abort_msg)
check("terminal engine never silently substitutes a voice",
      rt_term.spec["id"] == "_unit_terminal")

# ── 11 · Chatterbox checkpoint resolution (mocked HF hub, offline) ─────────
# Regression guard for the reported bug: from_pretrained() has no `repo_id`
# argument and hardcodes REPO_ID, so we drive snapshot_download ourselves.
try:
    import tempfile
    import huggingface_hub

    _base = ["ve.pt", "conds.pt", "s3gen.pt", "t3_cfg.pt",
             "grapheme_mtl_merged_expanded_v1.json", "t3_mtl23ls_v2.safetensors"]
    _hub_calls = []
    _orig_sd = huggingface_hub.snapshot_download
    _spec = pipeline.TTS_ENGINES["chatterbox"]
    _saved = (_spec["repo"], _spec["overlay_repo"])

    def _fake_sd(repo_id, token=None, allow_patterns=None):
        _hub_calls.append((repo_id, token, tuple(allow_patterns or ())))
        d = Path(tempfile.mkdtemp())
        for f in (_base if repo_id == "unit/base" else ["t3_hi.safetensors"]):
            (d / f).write_bytes(b"x")
        return str(d)

    huggingface_hub.snapshot_download = _fake_sd
    _spec["repo"], _spec["overlay_repo"] = "unit/base", "unit/overlay"

    _spec["t3_model"] = ""
    _d1 = pipeline._chatterbox_ckpt_dir(pipeline.Log(), "")
    check("chatterbox default path requests the base repo (not the overlay)",
          _hub_calls[0][0] == "unit/base")
    check("chatterbox default downloads the files from_local() reads",
          {"ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
           "t3_mtl23ls_v2.safetensors"} <= set(_hub_calls[0][2]))
    check("chatterbox passes our HF token through to snapshot_download",
          _hub_calls[0][1] is None)          # ambient credential when unset

    _hub_calls.clear()
    _d2 = pipeline._chatterbox_ckpt_dir(pipeline.Log(), "t3_hi.safetensors")
    check("Hindi finetune merges the overlay repo on top of the base",
          [c[0] for c in _hub_calls] == ["unit/base", "unit/overlay"]
          and (_d2 / "t3_hi.safetensors").exists())

    # overlay unavailable ⇒ degrade to the multilingual default, never crash
    def _no_overlay(repo_id, token=None, allow_patterns=None):
        d = Path(tempfile.mkdtemp())
        if repo_id == "unit/base":
            for f in _base:
                (d / f).write_bytes(b"x")
        return str(d)

    huggingface_hub.snapshot_download = _no_overlay
    _d3 = pipeline._chatterbox_ckpt_dir(pipeline.Log(), "t3_hi.safetensors")
    check("missing overlay degrades to the default T3 instead of dangling",
          not (_d3 / "t3_hi.safetensors").exists())

    # incomplete base checkpoint must be reported, not silently used
    huggingface_hub.snapshot_download = lambda *a, **k: str(tempfile.mkdtemp())
    try:
        pipeline._chatterbox_ckpt_dir(pipeline.Log(), "")
        check("incomplete checkpoint is rejected", False)
    except RuntimeError as exc:
        check("incomplete checkpoint is rejected",
              "incomplete" in str(exc) and "ve.pt" in str(exc))

    huggingface_hub.snapshot_download = _orig_sd
    _spec["repo"], _spec["overlay_repo"] = _saved
    _spec["t3_model"] = ""
except ImportError as exc:                    # huggingface_hub absent
    print(f"⏭ chatterbox checkpoint checks skipped ({exc})")

# ── 12 · automatic gender detection (validated on synthetic voices) ────────
# These are the exact regressions found while building this: plain YIN labelled
# PURE SILENCE as "female, 400 Hz" and WHITE NOISE as "male, 73 Hz", both at
# full confidence. pYIN's voiced/unvoiced flag rejects every non-voice input.
try:
    import librosa  # noqa: F401

    _GSR = 16_000

    def _voice(f0, dur=1.5, bright=1200.0, seed=0):
        """Harmonic glottal source + spectral tilt + syllable-rate envelope."""
        rng = np.random.default_rng(seed)
        n = int(_GSR * dur)
        t = np.arange(n) / _GSR
        sig = np.zeros(n)
        k = 1
        while k * f0 < 7000:
            f = k * f0
            sig += (np.exp(-f / bright) / k) * \
                np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
            k += 1
        sig *= 0.4 / (np.abs(sig).max() + 1e-9)
        env = np.clip((0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) * 1.6 - 0.3,
                      0.0, 1.0)
        return (sig * env).astype(np.float32)

    for _nm, _sig, _want in (
            ("male 110 Hz", _voice(110), "male"),
            ("male 140 Hz", _voice(140), "male"),
            ("female 235 Hz", _voice(235, bright=2600), "female"),
            ("female 280 Hz", _voice(280, bright=3000), "female"),
            ("silence", np.zeros(int(_GSR * 1.5), dtype=np.float32), ""),
            ("white noise",
             np.random.default_rng(3).normal(0, 0.1, int(_GSR * 1.5)).astype(np.float32), ""),
            ("0.2 s clip (no evidence)", _voice(130, dur=0.2), "")):
        _g = pipeline._estimate_gender(_sig, _GSR)["gender"]
        check(f"gender: {_nm} → {_want or 'undetermined'}", _g == _want)
    check("gender estimate reports a pitch and a confidence",
          pipeline._estimate_gender(_voice(120), _GSR)["f0_hz"] > 100.0)
except ImportError as exc:                    # librosa absent
    print(f"⏭ gender-detection checks skipped ({exc})")

# ── 13 · gender-matched voice casting (the "everyone sounds female" bug) ────
# The old allocator fell through to `sorted(pool)`, and "Female" sorts before
# "Male", so an all-male cast was handed female voices. It also ended in a
# hardcoded `hi-IN-SwaraNeural` default.
import inspect  # noqa: E402

_orig_pool = pipeline._edge_voice_list
_orig_profs = pipeline.load_speaker_profiles
_FIXTURE = {"Female": ["F1", "F2", "F3"], "Male": ["M1", "M2", "M3"]}

pipeline._edge_voice_list = lambda log=None: {k: list(v) for k, v in _FIXTURE.items()}
pipeline.load_speaker_profiles = lambda: {}

_cast = pipeline.allocate_speaker_voices({
    "S1": {"gender": "male", "f0_hz": 105.0},
    "S2": {"gender": "male", "f0_hz": 130.0},
    "S3": {"gender": "female", "f0_hz": 230.0},
}, pipeline.Log())
check("gendered speakers get distinct voices", len(set(_cast.values())) == 3)
check("male speakers get male voices",
      _cast["S1"].startswith("M") and _cast["S2"].startswith("M"))
check("female speakers get female voices", _cast["S3"].startswith("F"))
check("higher-pitched same-gender speaker takes the first voice",
      _cast["S2"] == "M1" and _cast["S1"] == "M2")

_cast2 = pipeline.allocate_speaker_voices({
    "A": {"gender": ""}, "B": {"gender": ""}, "C": {"gender": ""}}, pipeline.Log())
check("ungendered speakers alternate gender — never female-by-default",
      [s[0] for s in (_cast2[k] for k in sorted(_cast2))] == ["F", "M", "F"])

pipeline.load_speaker_profiles = lambda: {"S1": {"gender": "female"}}
_cast3 = pipeline.allocate_speaker_voices({"S1": {"gender": "male"}}, pipeline.Log())
check("Tab 2 override wins over the acoustic estimate",
      _cast3["S1"].startswith("F"))
pipeline.load_speaker_profiles = lambda: {}

check("_speaker_gender prefers the override over the estimate",
      pipeline._speaker_gender("X", {"gender": "male"}) == "male")
check("_speaker_gender returns '' when nothing is known",
      pipeline._speaker_gender("X", {"gender": ""}) == "")
check("_interleave_voices alternates genders",
      pipeline._interleave_voices({"Female": ["F1", "F2"], "Male": ["M1"]})
      == ["F1", "M1", "F2"])

def _code_only(fn) -> "str | None":
    """Source of `fn` with comments stripped, or None when there is no source.

    Phase 6 (`build_secure.py`) compiles pipeline.py into a native extension,
    and compiled code has no retrievable source — `inspect.getsource` raises.
    Returning None lets the source-scanning checks SKIP instead of either
    crashing or passing vacuously against an empty string.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):          # built-in / extension module
        return None
    return "\n".join(ln.split("#")[0] for ln in src.splitlines())


_edge_src_parts = [_code_only(pipeline.allocate_speaker_voices),
                   _code_only(pipeline._edge_speak)]
_edge_speak_src = _edge_src_parts[1]
if any(p is None for p in _edge_src_parts):
    _why = "pipeline is a compiled extension — no source to scan"
    skip("no hardcoded female voice left in the Edge casting path", _why)
    skip("_edge_speak refuses an empty voice instead of defaulting", _why)
else:
    _edge_src = "".join(_edge_src_parts)
    check("no hardcoded female voice left in the Edge casting path",
          "SwaraNeural" not in _edge_src)
    check("_edge_speak refuses an empty voice instead of defaulting",
          "raise RuntimeError" in _edge_speak_src)


pipeline._edge_voice_list = _orig_pool
pipeline.load_speaker_profiles = _orig_profs

# ── 14 · Edge prosody steering (pitch / rate / volume) ─────────────────────
_state_bak = pipeline.STATE_JSON.read_bytes() if pipeline.STATE_JSON.exists() else None
pipeline.STATE_JSON.unlink(missing_ok=True)
check("prosody defaults to neutral",
      pipeline.get_engine_prosody() == {"pitch": 0, "rate": 0, "volume": 0})
_clamped = pipeline.set_engine_prosody(pitch=999, rate=5, volume=-999)
check("prosody is clamped to the Edge endpoint ranges",
      _clamped == {"pitch": 100, "rate": 5, "volume": -100})
check("prosody persists", pipeline.get_engine_prosody()["pitch"] == 100)
check("partial updates leave the other fields alone",
      pipeline.set_engine_prosody(rate=10)
      == {"pitch": 100, "rate": 10, "volume": -100})
check("the edge engine spec carries the persisted prosody",
      pipeline.resolve_engine("edge")["prosody"]["rate"] == 10)
check("cloning engines have no prosody block",
      "prosody" not in pipeline.resolve_engine("cosyvoice2"))
check("prosody can be reset to neutral",
      pipeline.set_engine_prosody(pitch=0, rate=0, volume=0)
      == {"pitch": 0, "rate": 0, "volume": 0})
if _state_bak is not None:
    pipeline.STATE_JSON.write_bytes(_state_bak)
else:
    pipeline.STATE_JSON.unlink(missing_ok=True)

# ── 15 · Kokoro engine plumbing (fake pipeline — no model download) ────────
# Exercises BOTH generator shapes kokoro has shipped ((gs, ps, audio) tuples and
# the newer KPipeline.Result dataclass), plus blending and the gender guard.
from types import SimpleNamespace  # noqa: E402

_KVALS = {"hf_alpha": 1.0, "hf_beta": 2.0, "hm_omega": 10.0, "hm_psi": 20.0}


class _FakePipe:
    """Stand-in for KPipeline: load_voice → vector, __call__ → generator."""

    def __init__(self, shape: str = "result", chunks: int = 2) -> None:
        self.shape, self.chunks = shape, chunks
        self.loaded: list = []
        self.calls: list = []

    def load_voice(self, name):
        self.loaded.append(name)
        return np.full((4,), _KVALS.get(name, 0.0), dtype=np.float32)

    def __call__(self, text, voice=None, speed=1.0, **_kw):
        self.calls.append({"text": text, "voice": voice, "speed": speed})
        for i in range(0 if self.shape == "quiet" else self.chunks):
            wav = np.full(2400, 0.1 * (i + 1), dtype=np.float32)
            if self.shape == "tuple":
                yield (f"gs{i}", f"ps{i}", wav)
            else:
                yield SimpleNamespace(graphemes=f"gs{i}", phonemes=f"ps{i}",
                                      output=SimpleNamespace(audio=wav))


for _shape in ("result", "tuple"):
    _pipe = _FakePipe(_shape)
    _mdl = {"kind": "kokoro", "pipe": _pipe, "cache": {}, "sr": 24_000}
    _wave, _sr = pipeline._kokoro_speak(_mdl, "नमस्ते दोस्त", voice="hf_alpha")
    check(f"kokoro ({_shape} API) returns audio at 24 kHz",
          _sr == 24_000 and _wave.dtype == np.float32 and _wave.size > 4800)
    check(f"kokoro ({_shape} API) joins multi-chunk lines into one clip",
          _wave.size == 2400 + 1440 + 2400)      # 2 chunks + 0.06 s gap
    check(f"kokoro ({_shape} API) forwards text as a plain string",
          _pipe.calls[0]["text"] == "नमस्ते दोस्त")

_quiet = _FakePipe("quiet")
try:
    pipeline._kokoro_speak({"pipe": _quiet, "cache": {}, "sr": 24_000},
                           "x", voice="hf_alpha")
    check("kokoro refuses a silent generator", False)
except RuntimeError as exc:
    check("kokoro refuses a silent generator",
          "produced no audio" in str(exc))

try:
    pipeline._kokoro_speak({"pipe": _FakePipe(), "cache": {}, "sr": 24_000},
                           "x", voice="")
    check("kokoro refuses an empty voice cast", False)
except RuntimeError as exc:
    check("kokoro refuses an empty voice cast",
          "voice cast is empty" in str(exc))

# blending: weighted average of two SAME-GENDER voice tensors
_bp = _FakePipe()
_blend = {"partner": "hf_beta", "weight": 0.70}
_pack = pipeline._kokoro_voice_pack(_bp, "hf_alpha", _blend, {})
check("kokoro blends two same-gender voices (weighted average)",
      np.allclose(_pack, 0.70 * 1.0 + 0.30 * 2.0))

_clamped = pipeline._kokoro_voice_pack(_FakePipe(), "hf_alpha",
                                       {"partner": "hf_beta", "weight": 0.99}, {})
check("kokoro blend weight is clamped to ≤0.70",
      np.allclose(_clamped, 0.70 * 1.0 + 0.30 * 2.0))

_logs: list = []
_cross = pipeline._kokoro_voice_pack(_FakePipe(), "hf_alpha",
                                     {"partner": "hm_omega", "weight": 0.5},
                                     {}, lambda m: _logs.append(str(m)))
check("kokoro refuses a cross-gender blend partner",
      np.allclose(_cross, 1.0) and any("blend skipped" in m for m in _logs))

_cache: dict = {}
_cp = _FakePipe()
pipeline._kokoro_voice_pack(_cp, "hf_alpha", {}, _cache)
pipeline._kokoro_voice_pack(_cp, "hf_alpha", {}, _cache)
check("kokoro caches load_voice per session",
      _cp.loaded.count("hf_alpha") == 1)

_sp = _FakePipe()
pipeline._kokoro_speak({"pipe": _sp, "cache": {}, "sr": 24_000}, "x",
                       voice="hm_psi", speed=0.8)
check("kokoro forwards the speaking rate", _sp.calls[0]["speed"] == 0.8)

# one more quiet shape: a Result whose output carries no audio
class _QuietPipe(_FakePipe):
    def __call__(self, text, voice=None, speed=1.0, **_kw):
        self.calls.append({"text": text, "voice": voice, "speed": speed})
        yield SimpleNamespace(graphemes="g", phonemes="p",
                              output=SimpleNamespace(audio=None))


try:
    pipeline._kokoro_speak({"pipe": _QuietPipe(), "cache": {}, "sr": 24_000},
                           "x", voice="hf_alpha")
    check("kokoro detects a Result chunk with null audio", False)
except RuntimeError as exc:
    check("kokoro detects a Result chunk with null audio",
          "produced no audio" in str(exc))

# the shipped pool must agree with the gender encoded in each voice id
_KPOOL = pipeline._kokoro_voice_pool()
check("kokoro Hindi pool is gender-consistent",
      all(pipeline._kokoro_voice_gender(v)
          == ("female" if g == "Female" else "male")
          for g, vs in _KPOOL.items() for v in vs))

_kcast = pipeline.allocate_speaker_voices(
    {"A": {"gender": "male", "f0_hz": 100.0},
     "B": {"gender": "female", "f0_hz": 240.0}}, pipeline.Log(),
    pool=pipeline._kokoro_voice_pool(), native=lambda _v: True)
check("kokoro casting is gender-matched",
      _kcast["A"].startswith("hm_") and _kcast["B"].startswith("hf_"))

# ── Phase D · auto-blend above capacity ─────────────────────────────────────
# A helper: the effective share of the gender's FIRST pool voice in the final
# tensor `base*w + partner*(1-w)`. Two speakers are distinguishable only if
# these differ — that is the whole point of the policy.
def _alpha0(base, partner, w, pool):
    return (w if base == pool[0] else 0.0) + \
           ((1.0 - w) if partner == pool[0] else 0.0)


_F_POOL = ["hf_alpha", "hf_beta"]
_M_POOL = ["hm_omega", "hm_psi"]

# Reset any blend a previous check persisted — Phase D must start neutral.
pipeline.set_kokoro_blend("", None)

_si4 = {"F1": {"f0_hz": 220.0}, "F2": {"f0_hz": 200.0},
        "M1": {"f0_hz": 110.0}, "M2": {"f0_hz": 100.0}}
_cast4 = {"F1": "hf_alpha", "F2": "hf_beta",
          "M1": "hm_omega", "M2": "hm_psi"}

check("Kokoro Hindi capacity is 4 voices", pipeline.KOKORO_HINDI_CAPACITY == 4)
check("a cast within capacity needs no auto-blend",
      pipeline.auto_kokoro_blends(_cast4, _si4) == {})

_si5 = dict(_si4, F3={"f0_hz": 180.0})
_cast5 = dict(_cast4, F3="hf_alpha")           # F3 must SHARE F1's voice
_bl5 = pipeline.auto_kokoro_blends(_cast5, _si5)
check("only the recycled speaker is blended",
      set(_bl5) == {"F3"} and "F1" not in _bl5 and "F2" not in _bl5)
check("blend partner is the OTHER voice of the same gender",
      _bl5.get("F3", {}).get("partner") == "hf_beta")
check("auto-blend weight stays inside the 0.30–0.70 clamp",
      _bl5 and pipeline._KOKORO_BLEND_MIN <= _bl5["F3"]["weight"]
      <= pipeline._KOKORO_BLEND_MAX)

# 5 female speakers → 3 of them must share; every resulting mix must differ.
_si6 = {"F1": {"f0_hz": 230.0}, "F2": {"f0_hz": 215.0}, "F3": {"f0_hz": 200.0},
        "F4": {"f0_hz": 185.0}, "F5": {"f0_hz": 170.0}}
_cast6 = {"F1": "hf_alpha", "F2": "hf_beta", "F3": "hf_alpha",
          "F4": "hf_beta", "F5": "hf_alpha"}
_bl6 = pipeline.auto_kokoro_blends(_cast6, _si6)
_alphas = []
for spk, voice in _cast6.items():
    b = _bl6.get(spk)
    _alphas.append(_alpha0(voice, b["partner"], b["weight"], _F_POOL)
                   if b else (1.0 if voice == _F_POOL[0] else 0.0))
check("5 speakers → 3 auto-blends issued", len(_bl6) == 3)
check("every speaker resolves to a DISTINCT mix", len(set(_alphas)) == 5)
check("auto-blend never crosses genders",
      all(pipeline._kokoro_voice_gender(v["partner"]) == "female"
          for v in _bl6.values()))

# Precedence: an explicit Tab 2 partner is an instruction, so it wins.
pipeline.set_kokoro_blend("hf_beta", 0.5)
try:
    _manual = pipeline.auto_kokoro_blends(_cast5, _si5)
    check("manual Tab 2 blend partner beats Phase D auto-blend", _manual == {})
finally:
    pipeline.set_kokoro_blend("", None)

# Full-resolution check: the blend the dispatcher forwards for a shared speaker
# actually changes the tensor mixture (the mechanism Kokoro documents).
_seen_k = {}
_orig_kospeak, _orig_kload = pipeline._kokoro_speak, pipeline._load_kokoro


def _probe_kospeak(model, text, voice="", speed=1.0, blend=None, log=None, **kw):
    _seen_k["blend"] = blend or {}
    return [0.0] * 24000, 24000


try:
    pipeline._kokoro_speak, pipeline._load_kokoro = _probe_kospeak, (lambda log=None: "k")
    # resolve_engine() binds spec["speak"] from the module globals, so it must
    # be called AFTER the patch above for the probe to receive the call.
    _rt = pipeline.EngineRuntime(pipeline.resolve_engine("kokoro"),
                                 pipeline.Log())
    _rt.model = "k"
    _rt._dispatch("F3", {"index": 1, "text": "नमस्ते"}, "", "",
                  _cast6, _bl6)
    check("dispatcher forwards the per-speaker auto-blend",
          _seen_k.get("blend", {}).get("partner") == "hf_beta")
    _seen_k.clear()
    _rt._dispatch("F1", {"index": 1, "text": "नमस्ते"}, "", "", _cast6, _bl6)
    check("an unblended speaker forwards NO auto-blend",
          not _seen_k.get("blend", {}).get("partner"))
finally:
    pipeline._kokoro_speak, pipeline._load_kokoro = _orig_kospeak, _orig_kload

# ── voice preview (Tab 2 audition) ─────────────────────────────────────────
# Refusal paths must not touch the network or load weights.
_p, _m = pipeline.preview_voice(engine_id="chatterbox")
check("preview refuses a cloning engine",
      _p is None and "clones each speaker's own recording" in _m)
_p, _m = pipeline.preview_voice(engine_id="edge")
check("preview refuses rather than fall back to a default voice",
      _p is None and "No voice to preview" in _m)

# Success paths: stub the loaders/synthesizers so no network is touched.
_orig_load_edge, _orig_edge_speak = pipeline._load_edge, pipeline._edge_speak
_orig_load_koko, _orig_koko_speak = pipeline._load_kokoro, pipeline._kokoro_speak
_seen = {}


def _fake_load_edge(log=None):
    return "edge-client"


def _fake_edge_speak(model, text, voice="", rate=0, volume=0, pitch=0,
                     log=None, **kw):
    _seen.update(model=model, text=text, voice=voice, rate=rate,
                 volume=volume, pitch=pitch)
    return [0.0] * 8000, 16000          # 0.5 s @ 16 kHz


def _fake_load_koko(log=None):
    return "kokoro-pipe"


def _fake_koko_speak(model, text, voice="", speed=1.0, blend=None,
                     log=None, **kw):
    _seen.update(model=model, text=text, voice=voice, speed=speed, blend=blend)
    return [0.0] * 12000, 24000         # 0.5 s @ 24 kHz


try:
    pipeline._load_edge, pipeline._edge_speak = _fake_load_edge, _fake_edge_speak
    pipeline._load_kokoro = _fake_load_koko
    pipeline._kokoro_speak = _fake_koko_speak

    _seen.clear()
    path, msg = pipeline.preview_voice(engine_id="edge", voice="hi-IN-MadhurNeural",
                                       pitch=-30, rate=25, volume=10)
    check("edge preview synthesizes the audition line",
          path is not None and _seen.get("voice") == "hi-IN-MadhurNeural")
    check("edge preview forwards pitch/rate/volume",
          _seen.get("pitch") == -30 and _seen.get("rate") == 25
          and _seen.get("volume") == 10)
    check("edge preview writes a playable wav",
          path is not None and Path(path).exists()
          and Path(path).stat().st_size > 0)

    _seen.clear()
    path, msg = pipeline.preview_voice(engine_id="kokoro", voice="hm_omega",
                                       speed=1.15, blend_partner="hm_psi",
                                       blend_weight=0.6)
    check("kokoro preview forwards speed + blend",
          path is not None and _seen.get("speed") == 1.15
          and _seen.get("blend", {}).get("partner") == "hm_psi")
    check("kokoro preview reports voice and duration in the message",
          path is not None and "hm_omega" in msg and "s ·" in msg)
finally:
    pipeline._load_edge, pipeline._edge_speak = _orig_load_edge, _orig_edge_speak
    pipeline._load_kokoro, pipeline._kokoro_speak = _orig_load_koko, _orig_koko_speak
    pipeline.PREVIEW_WAV.unlink(missing_ok=True)

# ── Phase 6 · build_secure.py (Cython obfuscation builder) ───────────────────
import importlib.util as _ilu
import tempfile as _tf


def _raises(exc, fn) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _is_valid(src: str) -> bool:
    try:
        compile(src, "<embedded>", "exec")
        return True
    except SyntaxError:
        return False


_bs_spec = _ilu.spec_from_file_location(
    "build_secure", Path(__file__).resolve().parents[1] / "build_secure.py")
_bs = _ilu.module_from_spec(_bs_spec)
_bs_spec.loader.exec_module(_bs)

check("build_secure REQUIRED_API names all exist on pipeline",
      all(hasattr(pipeline, n) for n in _bs.REQUIRED_API))
check("build_secure's embedded verifier is syntactically valid Python",
      _is_valid(_bs._VERIFY_SRC))
check("verifier's marker survives embedding", _bs.MARKER in _bs._VERIFY_SRC)

# The scrub safety contract: neither flag alone nor both may scrub unverified.
check("--keep-source never scrubs", not _bs.can_scrub(True, False))
check("--no-verify can NEVER scrub", not _bs.can_scrub(False, True))
check("--keep-source + --no-verify never scrubs", not _bs.can_scrub(True, True))
check("default flags are the only ones that allow scrubbing",
      _bs.can_scrub(False, False))

with _tf.TemporaryDirectory() as _td:
    _t = Path(_td)
    check("no extension present → _binary_of returns None",
          _bs._binary_of(_t, "pipeline") is None)
    (_t / "pipeline.cp310-win_amd64.pyd").write_bytes(b"MZ")
    check("a planted extension is discovered",
          _bs._binary_of(_t, "pipeline") is not None)
    check("unrelated extensions are ignored",
          _bs._binary_of(_t, "other") is None)

    # The verifier hides the source then restores it; this is that restore.
    _src = _t / "pipeline.py"
    _bak = _t / f"pipeline.py{_bs.HIDDEN_SUFFIX}"
    _src.write_text("X = 1\n", encoding="utf-8")
    _src.replace(_bak)
    check("hidden source starts out hidden", not _src.exists())
    check("_ensure_restored puts the source back",
          _bs._ensure_restored(_t, "pipeline") and _src.exists())
    check("_ensure_restored leaves no backup behind", not _bak.exists())
    check("_ensure_restored is a no-op when the source is present",
          _bs._ensure_restored(_t, "pipeline") is False)

    check("preflight refuses a missing target",
          _raises(SystemExit, lambda: _bs.preflight(_t, "no_such_module", True)))
    check("preflight accepts an existing target in dry-run mode",
          _bs.preflight(_t, "pipeline", True) == _src)


# ── cleanup self-test artifacts (leave a pristine tree) ────────────────────
for p in (srt_path, tsrt, pipeline.DIARIZATION_JSON, pipeline.EMOTION_GRID_JSON,
          pipeline.FINAL_SCRIPT_JSON, pipeline.STATE_JSON,
          pipeline.SPEAKER_SCRIPTS_JSON, pipeline.FINAL_MIX_WAV,
          pipeline.MUSIC_WAV, track, elog,
          pipeline.OUTPUTS_DIR / "Project_Speaker1.txt",
          pipeline.OUTPUTS_DIR / "Project_Speaker2.txt"):
    p.unlink(missing_ok=True)

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)