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

# ── cleanup self-test artifacts (leave a pristine tree) ────────────────────
for p in (srt_path, tsrt, pipeline.DIARIZATION_JSON, pipeline.EMOTION_GRID_JSON,
          pipeline.FINAL_SCRIPT_JSON, pipeline.STATE_JSON,
          pipeline.SPEAKER_SCRIPTS_JSON, pipeline.FINAL_MIX_WAV,
          pipeline.MUSIC_WAV, track,
          pipeline.OUTPUTS_DIR / "Project_Speaker1.txt",
          pipeline.OUTPUTS_DIR / "Project_Speaker2.txt"):
    p.unlink(missing_ok=True)

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)