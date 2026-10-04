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
# Regression guard for the reported bug: 0.1.7's from_pretrained() takes
# neither `repo_id` nor `t3_model`, so we drive snapshot_download ourselves and
# present a finetune under the filename the wheel hardcodes.
try:
    import tempfile
    import huggingface_hub

    _hub_calls = []
    _orig_sd = huggingface_hub.snapshot_download
    _spec = pipeline.TTS_ENGINES["chatterbox"]
    _saved = (_spec["repo"], _spec["overlay_repo"])

    def _fake_sd(repo_id, token=None, allow_patterns=None):
        _hub_calls.append((repo_id, token, tuple(allow_patterns or ())))
        d = Path(tempfile.mkdtemp())
        for f in (pipeline._CHATTERBOX_FILES if repo_id == "unit/base"
                  else ["t3_hi.safetensors"]):
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
    check("Hindi finetune pulls the overlay repo on top of the base",
          [c[0] for c in _hub_calls] == ["unit/base", "unit/overlay"])
    check("finetune is served under the filename 0.1.7 hardcodes",
          (_d2 / pipeline._T3_FILENAME).exists()
          and not (_d2 / "t3_hi.safetensors").exists())
    check("finetune lands in its own dir — the base cache is untouched",
          _d2 != _d1
          and not (_d1 / pipeline._T3_FILENAME).samefile(
              _d2 / pipeline._T3_FILENAME))

    # overlay unavailable ⇒ degrade to the multilingual default, never crash
    def _no_overlay(repo_id, token=None, allow_patterns=None):
        d = Path(tempfile.mkdtemp())
        if repo_id == "unit/base":
            for f in pipeline._CHATTERBOX_FILES:
                (d / f).write_bytes(b"x")
        return str(d)

    huggingface_hub.snapshot_download = _no_overlay
    _d3 = pipeline._chatterbox_ckpt_dir(pipeline.Log(), "t3_hi.safetensors")
    check("missing overlay degrades to the default T3 instead of dangling",
          (_d3 / pipeline._T3_FILENAME).exists()
          and not (_d3 / "t3_hi.safetensors").exists())

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

# ── 11b · Chatterbox LOAD, end-to-end against a 0.1.7-shaped stub ──────────
# This reproduces the exact reported failure — an installed, importable
# chatterbox whose from_local(cls, ckpt_dir, device) has no `t3_model` — and
# proves the whole load path now completes instead of aborting the run.
import types as _types

_cb_calls = []


class _StubMTL:
    @classmethod
    def from_local(cls, ckpt_dir, device):
        _cb_calls.append(("from_local", str(ckpt_dir), device))
        return {"loaded": True}


_cb_pkg = _types.ModuleType("chatterbox")
_cb_mod = _types.ModuleType("chatterbox.mtl_tts")
_cb_mod.ChatterboxMultilingualTTS = _StubMTL
_cb_pkg.mtl_tts = _cb_mod
_mods = ("chatterbox", "chatterbox.mtl_tts", "torch")
_saved_mods = {k: sys.modules.get(k) for k in _mods}
if _saved_mods["torch"] is None:              # this box may have no torch
    _fake_torch = _types.ModuleType("torch")
    _fake_torch.cuda = _types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = _fake_torch
sys.modules["chatterbox"] = _cb_pkg
sys.modules["chatterbox.mtl_tts"] = _cb_mod

_orig_ckpt_fn = pipeline._chatterbox_ckpt_dir
pipeline._chatterbox_ckpt_dir = lambda log, t3_model="": Path("unit-ckpt")
try:
    _loaded = pipeline._load_chatterbox(pipeline.Log())
    check("load_chatterbox succeeds against a 0.1.7-style from_local",
          _loaded == {"loaded": True})
    check("…calling from_local(ckpt_dir, device) and never t3_model",
          len(_cb_calls) == 1 and _cb_calls[0][:2] == ("from_local",
                                                       "unit-ckpt")
          and len(_cb_calls[0]) == 3)
finally:
    pipeline._chatterbox_ckpt_dir = _orig_ckpt_fn
    for _k, _v in _saved_mods.items():
        if _v is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _v

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


# ── Chatterbox call plumbing · the 0.1.7 `t3_model` regression ─────────────
# The released wheel declares from_local(cls, ckpt_dir, device). Calling it with
# `t3_model=` raised "unexpected keyword argument" and killed every Chatterbox
# run even though the install was perfect. These pin the fix.
import numpy as _np


class _CB017:
    """chatterbox-tts 0.1.7 — the signature that actually ships."""

    @classmethod
    def from_local(cls, ckpt_dir, device):
        return {"ckpt": str(ckpt_dir), "device": device}


class _CBMaster:
    """Unreleased master — from_local grew a t3_model keyword."""

    seen = "unset"

    @classmethod
    def from_local(cls, ckpt_dir, device, t3_model=None):
        cls.seen = t3_model
        return {"t3": t3_model}


class _CBSoppy:
    """A build that swallows anything (**kwargs)."""

    seen = None

    @classmethod
    def from_local(cls, ckpt_dir, device, **kw):
        cls.seen = kw
        return kw


check("adaptive call drops t3_model for the real 0.1.7 from_local",
      pipeline._call_with_supported_kwargs(
          _CB017.from_local, "ckpt", "cuda",
          t3_model="t3_hi.safetensors") == {"ckpt": "ckpt", "device": "cuda"})
_CBMaster.seen = "unset"
pipeline._call_with_supported_kwargs(_CBMaster.from_local, "c", "cuda",
                                     t3_model="t3_hi.safetensors")
check("adaptive call still forwards t3_model to a master-style signature",
      _CBMaster.seen == "t3_hi.safetensors")
_CBSoppy.seen = None
pipeline._call_with_supported_kwargs(_CBSoppy.from_local, "c", "cuda",
                                     t3_model="x", other=1)
check("adaptive call forwards everything to a **kwargs signature",
      _CBSoppy.seen == {"t3_model": "x", "other": 1})
check("adaptive call survives a callable with no readable signature",
      isinstance(pipeline._call_with_supported_kwargs(type, "X", (), {}), type))

check("T3 filename matches the name hardcoded in the wheel",
      pipeline._T3_FILENAME == "t3_mtl23ls_v2.safetensors")
check("t3_cfg.pt is NOT requested (0.1.7's from_local never reads it)",
      "t3_cfg.pt" not in pipeline._CHATTERBOX_FILES)
check("every mandatory file appears in the snapshot list",
      set(pipeline._CHATTERBOX_REQUIRED) <= set(pipeline._CHATTERBOX_FILES))

with _tf.TemporaryDirectory() as _td:
    _t = Path(_td)
    _base = _t / "base"
    _base.mkdir()
    for _n in ("ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
               "Cangjie5_TC.json", "conds.pt"):
        (_base / _n).write_bytes(b"BASE:" + _n.encode())
    (_base / pipeline._T3_FILENAME).write_bytes(b"BASE:v2-t3")
    _fine = _t / "t3_hi.safetensors"
    _fine.write_bytes(b"HINDI-FINETUNE")

    _out = pipeline._chatterbox_overlay_dir(_base, _fine, lambda _m: None)

    check("overlay dir carries every base checkpoint file",
          all((_out / _n).exists() for _n in
              ("ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
               "Cangjie5_TC.json", "conds.pt")))
    check("overlay swaps the finetune in under the HARDCODED T3 name",
          (_out / pipeline._T3_FILENAME).read_bytes() == b"HINDI-FINETUNE")
    check("overlay dir is exactly from_local-ready (no missing/extra files)",
          sorted(p.name for p in _out.iterdir())
          == sorted(pipeline._CHATTERBOX_FILES))
    check("overlay never pollutes the shared base cache",
          (_base / pipeline._T3_FILENAME).read_bytes() == b"BASE:v2-t3")
    check("base cache still holds exactly its own files",
          sorted(p.name for p in _base.iterdir())
          == sorted(pipeline._CHATTERBOX_FILES))


class _LegacyGen:
    """A build whose generate() has no cfg_weight → the TypeError-retry path."""

    sr = 24_000

    def __init__(self):
        self.got = {}

    def generate(self, text, language_id, audio_prompt_path=None,
                 exaggeration=0.5):
        self.got = {"text": text, "lang": language_id,
                    "ref": audio_prompt_path, "exag": exaggeration}
        return [[0.0, 0.1, -0.1]]


_g = _LegacyGen()
_y, _sr = pipeline._chatterbox_speak(_g, "नमस्ते", ref_audio="r.wav",
                                     language="hi", exaggeration=0.7)
check("speak() retries without cfg_weight on a legacy generate()",
      _g.got.get("exag") == 0.7 and _sr == 24_000)
check("speak() returns mono float32 audio",
      _y.shape == (3,) and _y.dtype == _np.float32)


# ── 13 · engine allow-list — AutoDub Studio 2.0 publishes Chatterbox only ───
# AUTODUB_TTS_ENGINES narrows the registry at IMPORT time, so the already
# imported `pipeline` above can never observe it: the real mechanism has to be
# probed in a fresh interpreter, which is also exactly how the notebook's launch
# cell reaches `python app.py`.
import os as _os
import subprocess as _sp

_ROOT = str(Path(__file__).resolve().parents[1])
_PROBE = (
    "import json, sys\n"
    "sys.path.insert(0, %r)\n"
    "import pipeline as p\n"
    "import app\n"
    "print(json.dumps({\n"
    "  'engines': list(p.TTS_ENGINES),\n"
    "  'default': p.DEFAULT_TTS_ENGINE,\n"
    "  'active': p.get_tts_engine(),\n"
    "  'choices': [e for _, e in p.engine_choices()],\n"
    "  'fb': p.engine_fallback('chatterbox'),\n"
    "  'resolve': p.resolve_engine('cosyvoice2')['id'],\n"
    "  'set': p.set_tts_engine('cosyvoice2'),\n"
    "  'info': app._engine_info(),\n"
    "  'lic': app._engine_licence(),\n"
    "}))\n" % _ROOT)


def _probe(value):
    """Run the probe above with AUTODUB_TTS_ENGINES forced to `value`."""
    env = dict(_os.environ)
    env["AUTODUB_TTS_ENGINES"] = value
    proc = _sp.run([sys.executable, "-c", _PROBE], capture_output=True,
                   text=True, env=env, cwd=_ROOT, timeout=180)
    if proc.returncode != 0:
        return None, proc.stderr[-1200:]
    return json.loads(proc.stdout.strip().splitlines()[-1]), ""


_d, _err = _probe("chatterbox")
if _d is None:
    check("allow-list probe runs in a fresh interpreter", False)
    print(_err)
else:
    check("allow-list publishes chatterbox only",
          _d["engines"] == ["chatterbox"])
    check("…and makes it both the default and the active engine",
          _d["default"] == _d["active"] == "chatterbox")
    check("Tab 1's selector offers exactly one engine",
          _d["choices"] == ["chatterbox"])
    check("chatterbox stays TERMINAL under the allow-list", _d["fb"] == "")
    check("asking for a filtered-out engine degrades instead of raising",
          _d["resolve"] == _d["set"] == "chatterbox")
    check("Tab 1 copy names Chatterbox and no removed engine",
          "Chatterbox" in _d["info"] and "CosyVoice" not in _d["info"])
    check("licence notice covers only published engines",
          "PerTh" in _d["lic"] and "Fish" not in _d["lic"]
          and "Microsoft" not in _d["lic"])

# An empty/unset value must publish the FULL registry — that is what keeps the
# original Colab_Runner.ipynb (which never sets the variable) unchanged.
_e, _err = _probe("")
if _e is None:
    check("no allow-list → full registry still loads", False)
    print(_err)
else:
    check("unset allow-list keeps all six engines (original Colab_Runner)",
          _e["engines"] == list(pipeline.TTS_ENGINES) and len(_e["engines"]) == 6)
    check("…and the default is still cosyvoice2",
          _e["default"] == "cosyvoice2")
    check("Tab 1 copy falls back to the six-engine wording",
          "CosyVoice" in _e["info"])


# ── 14 · dedicated pipeline — AutoDub Studio 2.0 runs pipeline2.py ───────────
# app.py binds whichever module AUTODUB_PIPELINE names, and that binding happens
# at app import time, so it too can only be observed from OUTSIDE the process.
import importlib as _il


def _exec_body(path, drop=("PIPELINE_VARIANT =", "ENGINE_ALLOWLIST:")):
    """Executable lines only (comments/blank stripped), minus the identity pair."""
    out = []
    for _l in Path(path).read_text(encoding="utf-8").splitlines():
        _s = _l.strip()
        if not _s or _s.startswith("#"):
            continue
        if _s.startswith(drop):
            continue
        out.append(_s)
    return out


check("pipeline.py identifies as the 1.0 build",
      pipeline.PIPELINE_VARIANT == "1.0"
      and pipeline.ENGINE_ALLOWLIST == ()
      and len(pipeline.TTS_ENGINES) == 6)

_p2 = _il.import_module("pipeline2")
check("pipeline2.py imports and identifies as the 2.0 build",
      _p2.PIPELINE_VARIANT == "2.0"
      and _p2.ENGINE_ALLOWLIST == ("chatterbox",))
check("pipeline2 is Chatterbox-only with NO env var set (baked, not inherited)",
      list(_p2.TTS_ENGINES) == ["chatterbox"]
      and _p2.DEFAULT_TTS_ENGINE == "chatterbox"
      and _p2.engine_fallback("chatterbox") == "")
check("pipeline2 exposes build_secure's whole public API",
      all(hasattr(_p2, n) for n in _bs.REQUIRED_API))
check("pipeline2 differs from pipeline ONLY in its identity constants",
      _exec_body(pipeline.__file__) == _exec_body(_p2.__file__))
check("…so the 2.0 variant is a real duplicate, not a divergent fork",
      len(_exec_body(_p2.__file__)) > 1000)

_SEL = (
    "import json, os, sys\n"
    "sys.path.insert(0, %r)\n"
    "try:\n"
    "    import app\n"
    "    p = app.pipeline\n"
    "    out = {'module': p.__name__, 'variant': p.PIPELINE_VARIANT,\n"
    "           'engines': list(p.TTS_ENGINES),\n"
    "           'missing_api': [n for n in %r if not hasattr(p, n)],\n"
    "           'banner': getattr(app, '_launch_banner', lambda: '')()}\n"
    "except RuntimeError as e:\n"
    "    out = {'error': str(e)}\n"
    "print(json.dumps(out))\n"
) % (_ROOT, tuple(_bs.REQUIRED_API))


def _sel(**over):
    """Import app in a fresh interpreter with these env vars, return its answer."""
    env = dict(_os.environ)
    # Drop the pipeline vars too: a host that happens to export them would
    # otherwise leak into the "unset" cases below and test the wrong thing.
    for _k in ("AUTODUB_PIPELINE", "AUTODUB_TTS_ENGINES",
               "AUTODUB_PIPELINE_VARIANTS"):
        env.pop(_k, None)
    env.update({k: str(v) for k, v in over.items()})
    proc = _sp.run([sys.executable, "-c", _SEL], capture_output=True,
                   text=True, env=env, cwd=_ROOT, timeout=180)
    if proc.returncode != 0:
        return None, proc.stderr[-1200:]
    return json.loads(proc.stdout.strip().splitlines()[-1]), ""


_sel_d, _err = _sel(AUTODUB_PIPELINE="pipeline2")
if _sel_d is None:
    check("app binds pipeline2 when AUTODUB_PIPELINE=pipeline2", False)
    print(_err)
else:
    check("app binds pipeline2 when AUTODUB_PIPELINE=pipeline2",
          _sel_d.get("module") == "pipeline2"
          and _sel_d.get("variant") == "2.0")
    check("…so the 2.0 build's UI offers Chatterbox only",
          _sel_d.get("engines") == ["chatterbox"])
    check("…and pipeline2 carries every name app.py needs",
          _sel_d.get("missing_api") == [])
    # The banner is the ONLY place the running process says which build it is:
    # both notebooks clone one folder, so a wrong launch is otherwise just a UI
    # that "looks like the other notebook's".
    _b2 = _sel_d.get("banner") or ""
    check("…and its launch banner reports the 2.0 build + a bare env",
          "build 2.0" in _b2 and "module pipeline2" in _b2
          and "AUTODUB_PIPELINE         = 'pipeline2'" in _b2)
    check("…and it blames the missing SWITCHER, not the wrong build",
          "build 2.0 with the switcher off" in _b2
          and "this is the 1.0 build" not in _b2)

_base_d, _err = _sel()
if _base_d is None:
    check("app defaults to pipeline.py when unset", False)
    print(_err)
else:
    check("app defaults to pipeline.py when AUTODUB_PIPELINE is unset",
          _base_d.get("module") == "pipeline"
          and _base_d.get("variant") == "1.0")
    check("…so Colab_Runner keeps all six engines",
          len(_base_d.get("engines") or []) == 6)
    check("…and pipeline.py carries every name app.py needs",
          _base_d.get("missing_api") == [])
    # The Colab_Runner banner must NAME the trap it is in: the studio's own
    # pipeline2.py is sitting next to app.py, unbound, so the fix is one env var.
    _b1 = _base_d.get("banner") or ""
    check("…and the unset banner warns this is the 1.0 build, with the fix",
          "build 1.0" in _b1 and "this is the 1.0 build" in _b1
          and "AUTODUB_PIPELINE=pipeline2" in _b1)

_bad_d, _err = _sel(AUTODUB_PIPELINE="../evil")
if _bad_d is None:
    check("a non-identifier AUTODUB_PIPELINE is rejected, not imported", False)
    print(_err)
else:
    check("a non-identifier AUTODUB_PIPELINE is rejected, not imported",
          "plain Python module name" in (_bad_d.get("error") or ""))


# ── cleanup self-test artifacts (leave a pristine tree) ────────────────────
for p in (srt_path, tsrt, pipeline.DIARIZATION_JSON, pipeline.EMOTION_GRID_JSON,
          pipeline.FINAL_SCRIPT_JSON, pipeline.STATE_JSON,
          pipeline.SPEAKER_SCRIPTS_JSON, pipeline.FINAL_MIX_WAV,
          pipeline.MUSIC_WAV, track, elog,
          pipeline.OUTPUTS_DIR / "Project_Speaker1.txt",
          pipeline.OUTPUTS_DIR / "Project_Speaker2.txt"):
    p.unlink(missing_ok=True)

# ── § 15 · Tab 3 diagnostics must name the engine that actually ran ──────────
# Step 7's report used to hard-code "cosyvoice_tts", so the Chatterbox-only
# 2.0 build blamed a cosyvoice run for a chatterbox failure — and the printed
# fix never mentioned the protobuf mismatch that actually caused it.
_src = {name: (Path(_ROOT) / f"{name}.py").read_text(encoding="utf-8")
        for name in ("pipeline", "pipeline2")}
# The exact text Colab produced in the failed Tab 3 run.
_PB_VERSION_ERROR = (
    "VersionError: Detected incompatible Protobuf Gencode/Runtime versions "
    "when loading onnx/onnx-ml.proto: gencode 6.31.1 runtime 5.29.6. Runtime "
    "version cannot be older than the linked gencode version. See Protobuf "
    "version guarantees at https://protobuf.dev/support/"
    "cross-version-runtime-guarantee.")
for _name, _text in _src.items():
    check(f"[{_name}] Step 7 diagnostic label is derived, not hard-coded",
          '"Step 7 · cosyvoice_tts"' not in _text
          and 'f"Step 7 · {get_tts_engine()}"' in _text)
    check(f"[{_name}] Step 7 SKIPPED line names the active engine too",
          "cosyvoice_tts SKIPPED" not in _text
          and "{get_tts_engine()} SKIPPED" in _text)
    check(f"[{_name}] carries the protobuf gencode/runtime repair",
          "def _repair_protobuf_runtime(" in _text
          and '"protobuf>=6.31.1"' in _text)
    check(f"[{_name}] repair is wired into the chatterbox loader, not dead code",
          "attempts.extend(_repair_protobuf_runtime(log))" in _text
          and "_CHATTERBOX_FIX" in _text
          and "protobuf>=6.31.1" in _text.split("_CHATTERBOX_FIX", 1)[1][:4000])
    # Regression guard for the shipped-but-never-ran repair: the first version
    # decided whether to repair by PROBEing `import onnx.onnx_ml_pb2`, and
    # because that module imports cleanly on its own the probe returned early
    # and the repair never executed — while the printed fix still told people
    # to run the very command the code would not. Detection must come from the
    # exception actually raised, and there must be an attempt after the repair.
    check(f"[{_name}] repair is driven by the caught exception, not a probe",
          "def _is_protobuf_gencode_error(" in _text
          and "_is_protobuf_gencode_error(e2)" in _text)
    # ...and the repair itself must go straight to pip instead of pre-flighting
    # with `python -c`. Scoped to the function body so the docstring's account
    # of why the probe was abandoned does not defeat the check.
    _repair = (_text.split("def _repair_protobuf_runtime(", 1)[1]
               .split("\ndef ", 1)[0])
    check(f"[{_name}] repair runs pip directly instead of probing first",
          '"-c"' not in _repair
          and '"protobuf>=6.31.1"' in _repair
          and "subprocess.run" in _repair)
    check(f"[{_name}] retries once more after repairing protobuf",
          "[import retry 2]" in _text)

# Behaviour, not just source shape — the previous bug was invisible to
# string-matching because every symbol was present and wired up.
_VErr = type("VersionError", (Exception,), {})
check("protobuf detector matches the exact Colab VersionError",
      pipeline._is_protobuf_gencode_error(_VErr(_PB_VERSION_ERROR)))
check("protobuf detector rejects a missing module",
      not pipeline._is_protobuf_gencode_error(
          ModuleNotFoundError("No module named 'chatterbox'")))
check("protobuf detector rejects an unrelated import failure",
      not pipeline._is_protobuf_gencode_error(
          ImportError("cannot import name 'Tokenizers' from 'transformers'")))
check("protobuf detector rejects a plain runtime error mentioning neither",
      not pipeline._is_protobuf_gencode_error(RuntimeError("CUDA out of memory")))

# ── § 16 · Tab 3's original-audio player must be wired, not just declared ────
# A Gradio generator that yields the wrong number of values fails only at click
# time inside the browser, so the arity is verified here against the outputs
# list each callback was actually wired with.
import ast as _ast                      # noqa: E402

_APP = (Path(_ROOT) / "app.py").read_text(encoding="utf-8")
_tree = _ast.parse(_APP)


def _click_outputs(event: str):
    for node in _ast.walk(_tree):
        if not (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "click"):
            continue
        if not (_ast.get_source_segment(_APP, node) or "").startswith(event + "."):
            continue
        for kw in node.keywords:
            if kw.arg == "outputs" and isinstance(kw.value, _ast.List):
                return len(kw.value.elts)
    return None


def _tuple_arities(fn_name: str):
    """(line, width) for every tuple the function yields or returns.

    Walks nested closures too — _run_full_auto only ever yields emit_msg(...)
    and emit_stage(...), so its arity lives in those helpers' returns.
    """
    fn = next((n for n in _ast.walk(_tree)
               if isinstance(n, _ast.FunctionDef) and n.name == fn_name), None)
    if fn is None:
        return []
    return [(n.lineno, len(n.value.elts))
            for n in _ast.walk(fn)
            if isinstance(n, (_ast.Yield, _ast.Return))
            and isinstance(n.value, _ast.Tuple)]


for _ev, _fn in (("render_btn", "_run_rendering"),
                 ("auto_btn", "_run_full_auto")):
    _n = _click_outputs(_ev)
    _found = _tuple_arities(_fn)
    check(f"{_ev} wired with {_n} outputs", _n is not None)
    check(f"{_fn}: all {len(_found)} tuples are {_n}-wide",
          _found and all(w == _n for _, w in _found))

check("original_audio component declared", "original_audio = gr.Audio(" in _APP)
_i_mix = _APP.find('label="🎧 Final master mix"')
_i_src = _APP.find("original_audio = gr.Audio(")
_i_vid = _APP.find('label="🎬 Final dubbed video')
check("original_audio sits below Final master mix, above the video player",
      -1 < _i_mix < _i_src < _i_vid)
_hlpr = _APP.split("def _original_audio():", 1)
check("_original_audio helper defined", len(_hlpr) == 2)
if len(_hlpr) == 2:
    _body = _hlpr[1].split("\ndef ", 1)[0]
    check("_original_audio reads Step 1's AUDIO_WAV",
          "pipeline.AUDIO_WAV" in _body)
    check("_original_audio does not serve the Demucs vocal stem",
          "VOCALS_WAV" not in _body)
    check("_original_audio yields None rather than a missing path",
          "exists()" in _body)
check("original_audio wired into the render callback",
      "render_status,\n                     original_audio]" in _APP)
check("original_audio wired into the auto-pilot callback",
      "match_status, analysis_status, original_audio]" in _APP)

# ── § 17 · the language menus must offer only what the engine can speak ──────
# target_lang travels straight to generate(language_id=...), where Chatterbox
# raises on anything outside its own SUPPORTED_LANGUAGES. The old static menu
# offered "yue", which that engine does not have.
_CB23 = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
         "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}

check("ENGINE_LANGUAGES declared in both pipelines",
      all("ENGINE_LANGUAGES" in t for t in _src.values()))
check("chatterbox entry is the vendor's full 23-language set",
      {c for c, _ in pipeline.ENGINE_LANGUAGES["chatterbox"]} == _CB23
      and len(pipeline.ENGINE_LANGUAGES["chatterbox"]) == 23)
check("chatterbox entry is verbatim in pipeline2 too",
      "ENGINE_LANGUAGES" in _src.get("pipeline2", "")
      and all(c in _src["pipeline2"] for c in _CB23))
check("language_choices() exists", hasattr(pipeline, "language_choices"))

_checking = pipeline.get_tts_engine()
for _eng in pipeline.TTS_ENGINES:
    pipeline._TTS_ENGINE = _eng
    _c = pipeline.language_choices()
    _ca = pipeline.language_choices(include_auto=True)
    check(f"[{_eng}] offers at least one language", bool(_c))
    check(f"[{_eng}] source menu opens with auto-detect",
          _ca and _ca[0][1] == "auto")
    check(f"[{_eng}] dub menu has no auto-detect (generate needs a code)",
          all(v != "auto" for _, v in _c))
    check(f"[{_eng}] no duplicate codes",
          len({v for _, v in _c}) == len(_c))
    if _eng == "chatterbox":
        check("[chatterbox] never offers Cantonese, which it cannot speak",
              "yue" not in {v for _, v in _c})
pipeline._TTS_ENGINE = _checking


# ── the (label, value) contract, proven against Gradio itself ────────────────
# Gradio unpacks every choice as `for _, value in choices` (see
# gradio/components/dropdown.py → Dropdown.preprocess), so the SECOND element is
# the value the browser posts back and the first is pure cosmetics. Shipping
# (code, label) made each pretty name a "legal" value while rejecting the code
# the UI actually sends:
#     Error · Value: auto is not in the list of choices: ['🌐 Auto-detect', …]
# — the red banner in Tab 1. The tuple-order checks below fail loudly if the
# order is ever swapped back; the round-trip reproduces the exact user error.
def _accepts(comp, value) -> bool:
    """True when Gradio's own preprocess() accepts `value` for `comp`."""
    try:
        return comp.preprocess(value) == value
    except Exception:                # gradio.exceptions.Error, or an older API
        return False


try:
    import gradio as _gr
except Exception:                    # compiled / headless build
    _gr = None

for _eng in pipeline.TTS_ENGINES:
    pipeline._TTS_ENGINE = _eng
    _c = pipeline.language_choices()
    _ca = pipeline.language_choices(include_auto=True)
    # Compared against the source table rather than a shape heuristic: the
    # projections are the exact thing Gradio splits, so if the tuple order
    # regresses the values become the names and this fails immediately.
    _tbl = pipeline.ENGINE_LANGUAGES.get(_eng) or pipeline.DUBBING_LANGUAGES
    _codes = {c for c, _ in _tbl if c != "auto"}
    _names = {l for c, l in _tbl if c != "auto"}
    check(f"[{_eng}] choice VALUES are the engine's language codes",
          {v for _, v in _c} == _codes)
    check(f"[{_eng}] choice LABELS are the human-readable names",
          {l for l, _ in _c} == _names)
    check(f"[{_eng}] the code is second, not first — Gradio reads it there",
          not ({l for l, _ in _c} & _codes))
    if _gr is None:
        skip(f"[{_eng}] Gradio accepts its own menu values", "no gradio here")
        continue
    if not (_c and _ca):
        check(f"[{_eng}] menus are non-empty for the Gradio round-trip", False)
        continue
    _src_pick = _gr.Dropdown(choices=_ca, value="auto",
                             allow_custom_value=False)
    _dub_pick = _gr.Dropdown(choices=_c, value=_c[0][1],
                             allow_custom_value=False)
    check(f"[{_eng}] preprocess() accepts the source picker's own 'auto'",
          _accepts(_src_pick, "auto"))
    check(f"[{_eng}] preprocess() accepts every dub-side code",
          all(_accepts(_dub_pick, v) for _, v in _c))
    check(f"[{_eng}] a label is never accepted in place of a code",
          not _accepts(_dub_pick, _c[0][0]) or _c[0][0] == _c[0][1])
pipeline._TTS_ENGINE = _checking

check("Tab 1's original-language menu derives from language_choices",
      "pipeline.language_choices(\n                                    include_auto=True)"
      in _APP)
check("Tab 1's target-language menu derives from language_choices",
      "choices=pipeline.language_choices()," in _APP)
check("switching engines re-derives both language menus",
      "outputs=[tts_engine_note, lang_orig_in,\n                                      lang_target_in]"
      in _APP)

# ── § 16 · Tab 1 pipeline switcher — flip modules WITHOUT a restart ──────────
# Opt-in by design: the control renders only when AUTODUB_PIPELINE_VARIANTS
# names MORE THAN ONE module that actually exists as a file. That gate is what
# leaves Colab_Runner.ipynb — which sets no such variable — with the exact Tab 1
# it had before, while AutoDub Studio 2.0 gets a switcher on top.
_SWITCH = (
    "import json, sys\n"
    "sys.path.insert(0, %r)\n"
    "import app\n"
    "cfg = app.build_ui().get_config_file()\n"
    "radios = [c for c in cfg['components']\n"
    "          if c['props'].get('label') == 'Pipeline module']\n"
    "wired = [x for x in cfg['dependencies']\n"
    "         if x.get('api_name') == '_bind_pipeline']\n"
    "out = {'variants': list(app._PIPELINE_VARIANTS),\n"
    "       'bound': app.pipeline.__name__,\n"
    "       'banner': app._launch_banner(),\n"
    "       'switcher': len(radios),\n"
    "       'choice_vals': [c[1] for c in radios[0]['props']['choices']] if radios else [],\n"
    "       'default': radios[0]['props'].get('value') if radios else None,\n"
    "       'wired': len(wired),\n"
    "       'outputs': len(wired[0]['outputs']) if wired else 0,\n"
    "       'self_trig': bool(radios and wired and\n"
    "                          (radios[0]['id'], 'change') in wired[0]['targets'])}\n"
    "r = app._bind_pipeline('new_pipeline', 'auto', 'en')\n"
    "out['after'] = app.pipeline.__name__\n"
    "out['radio'] = r[0].get('value') if isinstance(r[0], dict) else None\n"
    "out['pipe_chip'] = r[1] if isinstance(r[1], str) else ''\n"
    "out['eng_chip'] = r[2] if isinstance(r[2], str) else ''\n"
    "out['eng_vals'] = ([c[1] for c in r[3].get('choices')]\n"
    "                   if isinstance(r[3], dict) and r[3].get('choices') else None)\n"
    "out['pre_evil'] = app.pipeline.__name__\n"
    "r = app._bind_pipeline('../evil', 'auto', 'en')\n"
    "out['evil_bound'] = app.pipeline.__name__\n"
    "out['evil_radio'] = r[0].get('value') if isinstance(r[0], dict) else None\n"
    "out['evil_chip'] = r[1] if isinstance(r[1], str) else ''\n"
    "out['arity'] = len(r)\n"
    "print(json.dumps(out))\n"
) % _ROOT


def _switch(**over):
    """Probe build_ui() + the live switcher with these env vars applied."""
    env = dict(_os.environ)
    for _k in ("AUTODUB_PIPELINE", "AUTODUB_TTS_ENGINES",
               "AUTODUB_PIPELINE_VARIANTS"):
        env.pop(_k, None)
    env.update({k: str(v) for k, v in over.items()})
    proc = _sp.run([sys.executable, "-c", _SWITCH], capture_output=True,
                   text=True, env=env, cwd=_ROOT, timeout=300)
    if proc.returncode != 0:
        return None, proc.stderr[-1500:]
    return json.loads(proc.stdout.strip().splitlines()[-1]), ""


_on_d, _err = _switch(AUTODUB_PIPELINE="pipeline2",
                      AUTODUB_PIPELINE_VARIANTS="pipeline2,new_pipeline")
if _on_d is None:
    check("the switcher builds when two variants are offered", False)
    print(_err)
else:
    check("the switcher lists exactly the two variants asked for",
          _on_d.get("variants") == ["pipeline2", "new_pipeline"]
          and _on_d.get("choice_vals") == ["pipeline2", "new_pipeline"])
    check("…and opens on the LAUNCH module, so it can't disagree with what is bound",
          _on_d.get("bound") == "pipeline2"
          and _on_d.get("default") == "pipeline2")
    check("one handler, wired to the switcher's own change event",
          _on_d.get("wired") == 1 and _on_d.get("self_trig") is True)
    check("…and it returns a value for every output it claims (6)",
          _on_d.get("outputs") == 6 and _on_d.get("arity") == 6)
    check("flipping to new_pipeline rebinds the global app reads at call time",
          _on_d.get("after") == "new_pipeline"
          and _on_d.get("radio") == "new_pipeline")
    check("…the switcher's chip names the module NOW driving the app",
          "Pipeline: new_pipeline" in (_on_d.get("pipe_chip") or ""))
    check("…and the engine chip is re-derived instead of left stale",
          "TTS Engine:" in (_on_d.get("eng_chip") or ""))
    check("…and the engine radio is re-derived with it",
          _on_d.get("eng_vals") == ["chatterbox"])
    # Same process, switcher ON: the banner must agree with the radio rather
    # than leaving the launch line contradicting the visible UI.
    _b0 = _on_d.get("banner") or ""
    check("the launch banner lists the switcher it just built",
          "Tab 1 switcher  : ON" in _b0
          and "pipeline2 | new_pipeline" in _b0
          and "1.0 build" not in _b0)
    # Rejection must be a true NO-OP: the module active before the call stays
    # active, and the radio is pulled back to it rather than left on the name
    # that was clicked (a half-switch would look like it silently worked).
    check("a hostile module name leaves the ACTIVE module running, radio reverted",
          _on_d.get("pre_evil") == "new_pipeline"
          and _on_d.get("evil_bound") == _on_d.get("pre_evil")
          and _on_d.get("evil_radio") == _on_d.get("pre_evil"))
    check("…with an explanation in the switcher's own chip",
          "Unknown pipeline" in (_on_d.get("evil_chip") or ""))

# No variable at all ⇒ no switcher, no handler: this is the Colab_Runner case.
_off_d, _err = _switch()
if _off_d is None:
    check("the switcher stays hidden when no variants are requested", False)
    print(_err)
else:
    check("no AUTODUB_PIPELINE_VARIANTS → Tab 1 has NO switcher (Colab_Runner unchanged)",
          _off_d.get("switcher") == 0 and _off_d.get("wired") == 0
          and _off_d.get("outputs") == 0)
    check("…and app still defaults to the 1.0 build with six engines",
          _off_d.get("bound") == "pipeline" and _off_d.get("variants") == [])

# Only ONE real variant (the other missing) ⇒ nothing to switch between, so the
# control must not render either; and a non-identifier must never be offered.
_forge_d, _err = _switch(AUTODUB_PIPELINE="pipeline2",
                         AUTODUB_PIPELINE_VARIANTS="pipeline2,does_not_exist")
check("variants with no matching file are dropped before rendering",
      _forge_d is None or (_forge_d.get("variants") == ["pipeline2"]
                           and _forge_d.get("switcher") == 0
                           and _forge_d.get("wired") == 0))

_evil_v, _err = _switch(AUTODUB_PIPELINE="pipeline2",
                        AUTODUB_PIPELINE_VARIANTS="../evil,pipeline2")
check("a path-shaped variant name is never offered as a choice",
      _evil_v is None or ("../evil" not in (_evil_v.get("choice_vals") or [])
                          and _evil_v.get("switcher") == 0))


# ── § 18 · Steps 3 & 4 · the ungated stack, its CSV side-cars, its labels ─────
# new_pipeline.py replaces the GATED Pyannote 3.1 diarizer and SenseVoice-Small
# with Silero-VAD + FunASR CAM++ + sklearn clustering and emotion2vec_plus_large.
# Steps 5-7 must not notice the swap: diarization_map.json and emotion_grid.json
# keep their exact shape. So this section pins the engines themselves, the four
# CSV side-cars now emitted alongside the frozen JSON, the 9→7 emotion label
# table, and the two Tab 2 input-order bugs fixed in app.py.
try:
    _npl = _il.import_module("new_pipeline")
except Exception as _e:            # a broken twin must FAIL the suite, not pass
    _npl, _np_err = None, f"{type(_e).__name__}: {_e}"
else:
    _np_err = ""

check("new_pipeline imports cleanly", _npl is not None)
if _npl is None:
    print(_np_err)

if _npl is not None:
    # ── engine identity: who runs Steps 3 & 4 now ───────────────────────────
    check("Step 3/4 engine string names the ungated stack (silero + campplus)",
          "silero" in _npl.DIAR_ENGINE.lower()
          and "campplus" in _npl.DIAR_ENGINE.lower())
    check("emotion model is emotion2vec_plus_large (keeps all 9 classes)",
          "emotion2vec_plus_large" in _npl.EMOTION2VEC_MODEL)
    check("CAMPPlus repo is the public VoxCeleb checkpoint",
          "voxceleb" in _npl.CAMPPLUS_MODEL.lower())
    check("STEP34_TOOLS is published for the Tab 1 chip",
          bool(getattr(_npl, "STEP34_TOOLS", "")))
    check("NEEDS_HF_TOKEN is False — Steps 3/4 gate nothing",
          getattr(_npl, "NEEDS_HF_TOKEN", None) is False)
    check("legacy model ids survive as inert string aliases only",
          isinstance(getattr(_npl, "PYANNOTE_MODEL", None), str)
          and isinstance(getattr(_npl, "SENSEVOICE_MODEL", None), str))

    # ── pyannote / SenseVoice must be OUT of the active path ────────────────
    # Tested on the AST, NOT on raw source: both functions still carry
    # docstrings/comments saying "UNCHANGED from the Pyannote version". Those
    # document the swap, they do not perform it. Only an import or a string
    # literal can actually reach a model, so those are what we check.
    _s3_src = inspect.getsource(_npl.step3_diarization)
    _s4_src = inspect.getsource(_npl.step4_emotion_analysis)
    _s3, _s4 = _s3_src.lower(), _s4_src.lower()

    def _live(fn):
        """(imported module roots, string literals) inside `fn`, minus its
        docstring and its comments — i.e. only what the interpreter can act on."""
        _t = _ast.parse(inspect.getsource(fn))
        _d = _t.body[0]
        _body = list(_d.body) if isinstance(_d, _ast.FunctionDef) else list(_t.body)
        if (_body and isinstance(_body[0], _ast.Expr)
                and isinstance(_body[0].value, _ast.Constant)
                and isinstance(_body[0].value.value, str)):
            _body = _body[1:]
        _mods, _strs = set(), set()
        for _top in _body:
            for _nd in _ast.walk(_top):
                if isinstance(_nd, _ast.Import):
                    _mods.update(a.name.split(".")[0] for a in _nd.names)
                elif isinstance(_nd, _ast.ImportFrom) and _nd.module:
                    _mods.add(_nd.module.split(".")[0])
                elif isinstance(_nd, _ast.Constant) and isinstance(_nd.value, str):
                    _strs.add(_nd.value)
        return _mods, _strs

    _m3, _t3 = _live(_npl.step3_diarization)
    _m4, _t4 = _live(_npl.step4_emotion_analysis)
    check("step3 imports no pyannote module", "pyannote" not in _m3)
    check("step3 names no gated speaker-diarization repo",
          not any("speaker-diarization" in _s for _s in _t3))
    check("step3 runs silero-vad through the shared VAD region helper",
          "silero" in _s3 and "_vad_regions" in _s3)
    check("step3 still writes the frozen diarization_map.json",
          _npl.DIARIZATION_JSON.name == "diarization_map.json"
          and "DIARIZATION_JSON" in _s3_src)
    check("step4 loads no SenseVoice model",
          "sensevoice" not in _m4
          and not any("sensevoice" in _s.lower() for _s in _t4))
    check("step4 loads emotion2vec+ and reads utterance granularity",
          "emotion2vec" in _s4 and "utterance" in _s4)
    check("step4 still writes the frozen emotion_grid.json",
          _npl.EMOTION_GRID_JSON.name == "emotion_grid.json"
          and "EMOTION_GRID_JSON" in _s4_src)

    # ── the Step 3 primitives the rewrite is built from ─────────────────────
    for _fn in ("_load_silero_vad", "_vad_regions", "_campplus_embed",
                "_cluster_speakers", "_write_csv", "_normalise_emotion"):
        check(f"new_pipeline defines {_fn}()", callable(getattr(_npl, _fn, None)))

    check("_cluster_speakers signature is (embs, n_hint, log)",
          list(inspect.signature(_npl._cluster_speakers).parameters)
          == ["embs", "n_hint", "log"])
    check("legacy PYANNOTE/SENSEVOICE ids are never called as models",
          "PYANNOTE_MODEL" not in _s3 and "SENSEVOICE_MODEL" not in _s4)

    # ── regression: each function must import the names IT uses ──────────────
    # Both real runtime bugs in this step were `NameError: name 'X' is not
    # defined` inside a function that treated a name as if it were module-level.
    # `_vad_regions` lost Silero and silently fell back to energy VAD; a missing
    # AutoModel import would abort the stage outright. Checking the function's
    # own source catches the whole class, not just the two we happened to hit.
    for _fn, _needle in (
        ("_vad_regions", "import torch"),
        ("_load_silero_vad", "import torch"),
        ("_funasr_auto", "from funasr import AutoModel"),
        ("_funasr_auto", "redirect_stdout"),
    ):
        _src = inspect.getsource(getattr(_npl, _fn))
        check(f"{_fn}() imports/uses its own '{_needle}'", _needle in _src)

    # ── regression: the Silero CUDA/CPU clash ───────────────────────────────
    # `model.to("cuda")` on the TorchScript bundle relocates the weights but not
    # the LSTM state the script allocates internally, so the first forward pass
    # died with  RuntimeError: Expected all tensors to be on the same device, but
    # got weight is on cuda:0, different from other tensors on cpu  and Step 3
    # fell back to the energy VAD on every T4 session.
    _sil_src = inspect.getsource(_npl._load_silero_vad)
    check("Silero-VAD is never moved onto the step device (no .to(device))",
          ".to(device)" not in _sil_src)
    check("new_pipeline defines _module_device()",
          callable(getattr(_npl, "_module_device", None)))
    _vad_src = inspect.getsource(_npl._vad_regions)
    check("_vad_regions builds the waveform on the model's own device",
          "_module_device(vad_model)" in _vad_src and "audio.to(mdev)" in _vad_src)

    # _module_device must read a real device, and must not explode on an object
    # that exposes neither parameters() nor buffers() — e.g. the OnnxWrapper.
    class _FakeDev:
        def __init__(self, d):
            self.device = d

    class _FakeMod:
        def parameters(self):
            return iter([_FakeDev("cuda:0")])

    check("_module_device reads a module's real device",
          _npl._module_device(_FakeMod()) == "cuda:0")
    check("_module_device answers 'cpu' for an opaque object",
          _npl._module_device(object()) == "cpu")

    # FunASR reports a failed download as "<raw hub id> is not registered", so a
    # single AutoModel call can only ever fail confusingly. Step 3 must offer a
    # second hub; Step 4 must NOT, because ModelScope's export has 4 classes.
    check("CAM++ has an official Hugging Face fallback id",
          getattr(_npl, "CAMPPLUS_MODEL_HF", "") == "funasr/campplus")
    # NOTE `_s3`/`_s4` are .lower() copies — needles must be written lowercase.
    check("step3 loads CAM++ through the multi-route loader",
          "_load_campplus(" in _s3
          and "automodel(model=campplus_model" not in _s3)
    check("step4 loads emotion2vec through the loader, HF only",
          "_funasr_auto(" in _s4 and 'hub="hf"' in _s4)
    check("step4 offers no modelscope fallback (wrong label count)",
          _npl.EMOTION2VEC_MODEL.split("/")[0] == "iic"
          and _s4.count("modelscope ·") == 0)

    # Static proof that no name is USED without being IMPORTED. This is the
    # exact class of bug that produced both runtime failures above: a name that
    # looks module-level but never was. pyflakes is optional, so skip cleanly.
    try:
        from pyflakes.api import checkPath as _pf_check
        from pyflakes.reporter import Reporter as _pf_rep
    except ImportError:
        skip("no used-but-never-imported names", "pyflakes not installed")
    else:
        import io as _pfio
        _root = Path(__file__).resolve().parents[1]
        _srcs = [_root / _f for _f in ("new_pipeline.py", "app.py")]
        _srcs = [_p for _p in _srcs if _p.exists()]
        if not _srcs:
            skip("no used-but-never-imported names", "sources not found")
        else:
            _bad: list = []
            for _p in _srcs:
                _out, _err = _pfio.StringIO(), _pfio.StringIO()
                try:
                    _pf_check(str(_p), _pf_rep(_out, _err))
                except Exception:
                    continue
                _bad += [_l for _l in (_out.getvalue() + _err.getvalue()).splitlines()
                         if "undefined name" in _l]
            check("no used-but-never-imported names in new_pipeline/app",
                  not _bad)

    # ── _funasr_auto really falls back, and reports funasr's SWALLOWED error ──
    # FunASR downgrades a failed download to a bare print(), so without
    # capturing stdout the operator only ever sees "<hub id> is not registered"
    # and has no idea the actual fault was SSL/proxy/offline.
    import types as _t2
    _saved_fun = sys.modules.get("funasr")
    _fun_mod = _t2.ModuleType("funasr")
    _am_calls: list = []
    _am_logs: list = []

    def _mk_auto(fail_on):
        def _AutoModel(**kw):
            _am_calls.append(kw.get("model"))
            if len(_am_calls) - 1 in fail_on:
                print("Download: %s failed!: SSL certificate expired"
                      % kw.get("model"))
                raise AssertionError(f'{kw.get("model")} is not registered')
            return f"MODEL#{len(_am_calls)}"
        return _AutoModel

    _am_routes = [("modelscope · iic/x", {"model": "iic/x"}),
                  ("huggingface · funasr/y", {"model": "funasr/y", "hub": "hf"})]

    try:
        _fun_mod.AutoModel = _mk_auto({0})          # first hub fails, second works
        sys.modules["funasr"] = _fun_mod
        _got = _npl._funasr_auto(_am_routes, "cpu", _am_logs.append, "CAM++")
        check("_funasr_auto falls back to the second hub", _got == "MODEL#2")
        check("…trying hubs in the order given",
              _am_calls == ["iic/x", "funasr/y"])
        check("…logging the failure and then the success",
              any("modelscope" in _m and "failed" in _m for _m in _am_logs)
              and any("huggingface" in _m and "✓" in _m for _m in _am_logs))

        _am_calls.clear()
        _am_logs.clear()
        _fun_mod.AutoModel = _mk_auto({0, 1})       # every hub fails
        try:
            _npl._funasr_auto(_am_routes, "cpu", _am_logs.append, "CAM++")
            _boom = None
        except RuntimeError as _e:
            _boom = str(_e)
        check("a total failure raises RuntimeError, not a bare assert",
              _boom is not None)
        check("…naming every hub that was tried",
              bool(_boom) and "modelscope" in _boom and "huggingface" in _boom)
        check("…and surfacing funasr's swallowed download error",
              bool(_boom) and "SSL certificate expired" in _boom)
    finally:
        sys.modules.pop("funasr", None)
        if _saved_fun is not None:
            sys.modules["funasr"] = _saved_fun


    # ── the four CSV side-cars ──────────────────────────────────────────────
    _CSVS = (("SPEAKER_TURNS_CSV", "speaker_turns.csv"),
             ("DIAR_CUES_CSV", "diarization_cues.csv"),
             ("EMOTION_GRID_CSV", "emotion_grid.csv"),
             ("LINE_FIT_CSV", "line_fit_report.csv"))
    check("all four CSV side-car paths are published as Path objects",
          all(isinstance(getattr(_npl, _a, None), Path) for _a, _ in _CSVS))
    check("all four side-cars land in outputs/ under the expected filename",
          all(getattr(_npl, _a).parent.name == "outputs"
              and getattr(_npl, _a).name == _n for _a, _n in _CSVS))
    check("speaker_turns.csv header is [Speaker ID, Start_Time, End_Time]",
          '["Speaker ID", "Start_Time", "End_Time"]' in _s3_src)
    check("diarization_cues.csv adds the Text column",
          '["Speaker ID", "Start_Time", "End_Time", "Text"]' in _s3_src)
    check("emotion_grid.csv header adds the Emotion column",
          '["Speaker ID", "Start_Time", "End_Time", "Emotion"]' in _s4_src)

    # ── 9-class emotion2vec+ label → 7 canonical emotions ──────────────────
    _EMO_CASES = (("生气/angry", "angry"), ("厌恶/disgusted", "disgusted"),
                  ("恐惧/fearful", "fearful"), ("开心/happy", "happy"),
                  ("中立/neutral", "neutral"), ("其他/other", "neutral"),
                  ("难过/sad", "sad"), ("吃惊/surprised", "surprised"),
                  ("未知/unknown", "neutral"), ("", "neutral"),
                  ("angry", "angry"), (None, "neutral"))
    _wrong = [_c for _c in _EMO_CASES
              if _npl._normalise_emotion(_c[0]) != _c[1]]
    check(f"_normalise_emotion maps all {len(_EMO_CASES)} label forms correctly",
          not _wrong)
    if _wrong:
        print(f"   mismatches: "
              f"{[(i, _npl._normalise_emotion(i), o) for i, o in _wrong]}")
    check("unknown / GASP / blank labels degrade to 'neutral', never raise",
          _npl._normalise_emotion("GASP") == "neutral"
          and _npl._normalise_emotion("not-a-label") == "neutral")
    check("every emotion2vec+ label collapses into the 7 canonical emotions",
          len(_npl.EMOTIONS) == 7
          and all(v in _npl.EMOTIONS for v in _npl.EMOTION2VEC_MAP.values()))
    check("Chatterbox exaggeration table covers all 7 canonical emotions",
          all(k in _npl.EMOTIONS for k in getattr(_npl, "_CHATTER_EXAG", {})))

    # ── Step 7 · two-way per-line fit + its side-car ────────────────────────
    _s7_src = inspect.getsource(_npl.step7_synthesize)
    check("step7 exposes both fit caps as plain arguments",
          {"fit_max_speed", "fit_min_speed", "fit_stretch_short"}
          <= set(inspect.signature(_npl.step7_synthesize).parameters))
    check("step7 may SLOW a short line down, not only speed a long one up",
          "fit_stretch_short" in _s7_src and '"stretched"' in _s7_src)
    check("step7 keeps the hard-snap + 30 ms fade as the last resort",
          "last resort" in _s7_src and "linspace(1.0, 0.0" in _s7_src)
    check("step7 writes line_fit_report.csv with per-row Rate and Mode",
          "LINE_FIT_CSV" in _s7_src and '"Rate", "Mode"' in _s7_src)
    check("step7 progress log no longer calls a speed-up a 'stretch'",
          "overlap-protection stretch" not in _s7_src)

# ── § 19 · app.py · Tab 2 side-car tables + the two input-order bugs ──────────
_app_src = (Path(_ROOT) / "app.py").read_text(encoding="utf-8")
_app_tree = _ast.parse(_app_src)


def _wired(btn: str, arg: str):
    """Names inside `<btn>.click(..., <arg>=[...])`, in source order."""
    for _n in _ast.walk(_app_tree):
        if (isinstance(_n, _ast.Call) and isinstance(_n.func, _ast.Attribute)
                and _n.func.attr == "click" and isinstance(_n.func.value, _ast.Name)
                and _n.func.value.id == btn):
            for _kw in _n.keywords:
                if _kw.arg == arg and isinstance(_kw.value, _ast.List):
                    return [getattr(_e, "id", None) for _e in _kw.value.elts]
    return None


def _fn_params(name: str):
    for _n in _app_tree.body:
        if isinstance(_n, _ast.FunctionDef) and _n.name == name:
            return [a.arg for a in _n.args.args]
    return None


def _ret_widths(fn: str):
    """Widths of the literal tuples `fn` RETURNS (helper-shaped callables)."""
    for _n in _ast.walk(_app_tree):
        if isinstance(_n, _ast.FunctionDef) and _n.name == fn:
            return sorted({len(_r.value.elts) for _r in _ast.walk(_n)
                           if isinstance(_r, _ast.Return)
                           and isinstance(_r.value, _ast.Tuple)})
    return []


def _yield_widths(fn: str, _seen=None):
    """Every distinct tuple width `fn` can hand to Gradio.

    app.py uses both shapes: a literal `yield a, b, c`, and
    `yield helper(...)`, where the tuple is RETURNED by the helper. A bare
    `yield from other()` is followed too, which is how the auto-pilot's
    per-second pause ticks reach the same seven outputs.
    """
    _seen = set() if _seen is None else _seen
    if fn in _seen:
        return []
    _seen.add(fn)
    for _n in _ast.walk(_app_tree):
        if not (isinstance(_n, _ast.FunctionDef) and _n.name == fn):
            continue
        _w = set()
        for _y in _ast.walk(_n):
            if isinstance(_y, _ast.Yield):
                _v = _y.value
            elif isinstance(_y, _ast.YieldFrom):
                _v = _y.value
            else:
                continue
            if isinstance(_v, _ast.Tuple):
                _w.add(len(_v.elts))
            elif isinstance(_v, _ast.Call) and isinstance(_v.func, _ast.Name):
                _w.update(_ret_widths(_v.func.id))
                _w.update(_yield_widths(_v.func.id, _seen))
        return sorted(x for x in _w if x)
    return []


check("app.py reads the CSV side-cars rather than re-deriving them",
      "_sidecar_rows" in _app_src and "EMOTION_GRID_CSV" in _app_src
      and "DIAR_CUES_CSV" in _app_src)
check("app.py renders both new Tab 2 tables (diar cues + emotion grid)",
      "diar_tbl" in _app_src and "emo_tbl" in _app_src)
check("Tab 2 no longer promises a Pyannote HF token",
      "Pyannote diarization)" not in _app_src)

# BUG 1 · match_btn fed (…, diag_in, spk_hint_in) into _run_matching(
# token, srt_o, srt_t, num_speakers, diagnostic): a bool landed in the
# speaker-count slot, so int(num_speakers) threw before the step ran at all.
_mn_in = _wired("match_btn", "inputs")
_mn_fn = _fn_params("_run_matching")
check("match_btn input count matches _run_matching's parameter count",
      _mn_in is not None and _mn_fn is not None and len(_mn_in) == len(_mn_fn))
check("match_btn puts spk_hint_in in the num_speakers slot (index 3)",
      _mn_in is not None and len(_mn_in) > 3
      and _mn_in[3] == "spk_hint_in" and _mn_in[4] == "diag_in")

# BUG 2 · auto_btn fed (…, diag_in, spk_hint_in, translit_in) into
# _run_full_auto(media, srt_o, srt_t, token, lang_o, lang_t, num_speakers,
# translit, diagnostic) — three slots shifted, so the bool reached the
# speaker count and the string reached the transliterate flag.
_au_in = _wired("auto_btn", "inputs")
_au_fn = _fn_params("_run_full_auto")
check("auto_btn input count matches _run_full_auto's parameter count",
      _au_in is not None and _au_fn is not None and len(_au_in) == len(_au_fn))
check("auto_btn orders spk_hint → translit → diag like the signature does",
      _au_in is not None
      and _au_in.index("spk_hint_in") < _au_in.index("translit_in")
      and _au_in.index("translit_in") < _au_in.index("diag_in"))
check("auto_btn leaves diag_in (a bool) in the LAST, diagnostic slot",
      _au_in is not None and _au_in[-1] == "diag_in")

# BUG 3 · Tab 3's STOP button was RENDERED (`stop_btn = gr.Button("🛑 STOP")`)
# but never bound to any handler, so `_stop_auto` — and the `_stop_event` the
# auto-pilot polls between stages — could never be triggered from the UI. Its
# own caption promised "STOP halts between stages". With `demo.queue()` at the
# default concurrency there was therefore no way to abort a runaway render
# except killing the Colab cell, which the browser reports as
# "Connection to the server was lost".
_stop_click = next((_n for _n in _ast.walk(_app_tree)
                    if isinstance(_n, _ast.Call)
                    and isinstance(_n.func, _ast.Attribute)
                    and _n.func.attr == "click"
                    and isinstance(_n.func.value, _ast.Name)
                    and _n.func.value.id == "stop_btn"), None)
check("Tab 3 STOP button is bound to a click handler", _stop_click is not None)
check("…and that handler is _stop_auto, the _stop_event setter",
      _stop_click is not None
      and any(_k.arg == "fn" and getattr(_k.value, "id", None) == "_stop_auto"
              for _k in _stop_click.keywords))
check("_stop_auto reaches the same Event the auto-pilot polls",
      "_stop_event.set()" in _app_src
      and "_stop_event.clear()" in _app_src
      and "_stop_event.is_set()" in _app_src)

# The auto-pilot polls `_stop_event` after EVERY stage and between each of the
# 15 one-second review ticks, so the button takes effect promptly rather than
# only after the whole render.
_auto_src = _app_src.split("def _run_full_auto(", 1)[-1].split("\ndef ", 1)[0]
check("auto-pilot checks the stop flag after every stage",
      _auto_src.count("_stop_event.is_set()") >= 5)
check("auto-pilot polls the stop flag once per review tick",
      "if _stop_event.is_set():" in _auto_src
      and "time.sleep(1)" in _auto_src)

# Every generator yield must be as wide as the outputs list it feeds, or
# Gradio silently drops the tail of the stream.
for _fn, _btn in (("_run_matching", "match_btn"),
                  ("_run_full_auto", "auto_btn")):
    _out = _wired(_btn, "outputs")
    _ys = _yield_widths(_fn)
    check(f"{_fn}'s yields are uniform and match {_btn}'s output count",
          _out is not None and len(_ys) == 1 and _ys[0] == len(_out))

# ── § 19b · the switcher chip must describe the build it just bound ──────────
if _on_d is not None:
    _chip_txt = _on_d.get("pipe_chip") or ""
    check("switcher chip still names the bound module (compat substring kept)",
          "Pipeline: new_pipeline" in _chip_txt)
    check("switcher chip now surfaces that build's Step 3/4 toolchain",
          "steps 3–4:" in _chip_txt)
    check("…and names new_pipeline's real stack (no Pyannote/SenseVoice)",
          "Silero" in _chip_txt and "emotion2vec" in _chip_txt
          and "Pyannote" not in _chip_txt and "SenseVoice" not in _chip_txt)
# ── § 19c · "Auto" must never reach int() ────────────────────────────────────
# `spk_hint_in` is a gr.Dropdown with choices=["Auto","1"…"10"] and
# value="Auto", so `num_speakers` arrives as a STRING on every run. The old
# `int(num_speakers) if num_speakers else 0` only guarded the empty case and
# died with `invalid literal for int() with base 10: 'Auto'` — before Step 3
# ever started.
_lines = _app_src.splitlines()


def _fn_src(name: str) -> str:
    for _n in _app_tree.body:
        if isinstance(_n, _ast.FunctionDef) and _n.name == name:
            return "\n".join(_lines[_n.lineno - 1:_n.end_lineno])
    return ""


_sh_node = next((_n for _n in _app_tree.body
                 if isinstance(_n, _ast.FunctionDef)
                 and _n.name == "_speaker_hint"), None)
check("app defines a _speaker_hint() converter", _sh_node is not None)
if _sh_node is not None:
    # Exec just that one function — importing all of app.py would run Gradio.
    _ns: dict = {}
    exec(compile(_ast.Module(body=[_sh_node], type_ignores=[]),
                 "app.py", "exec"), _ns)
    _hint = _ns["_speaker_hint"]
    check("_speaker_hint maps the dropdown's 'Auto' to 0 (auto-detect)",
          _hint("Auto") == 0)
    check("_speaker_hint survives '', None and arbitrary text",
          _hint("") == 0 and _hint(None) == 0 and _hint("four") == 0)
    check("_speaker_hint still honours a real numeric hint",
          _hint("3") == 3 and _hint(3) == 3 and _hint(10) == 10)
    check("_speaker_hint clamps a negative hint to 0", _hint("-4") == 0)

for _fn in ("_run_matching", "_run_full_auto"):
    _fs = _fn_src(_fn)
    check(f"{_fn} converts num_speakers through _speaker_hint",
          "_speaker_hint(num_speakers)" in _fs
          and "int(num_speakers) if" not in _fs)

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)