# AutoDub Studio 2.0 — Pipeline, Tools & Process Reference

> **Provenance.** Measured **2026-10-02** against HEAD `5a8b990` (`fix(tab1): language menus were (code,label); Gradio needs (label,value)`).
>
> Every figure marked **(measured)** was fetched live at that time from the HuggingFace model API, the PyPI JSON API, or an HTTP `HEAD` request. Figures marked **(stated)** are quoted from code comments or `README.md` and were not independently re-measured. Sizes drift — re-measure before quoting them elsewhere.
>
> Scope: the **2.0 build only** (`pipeline2.py` + `AutoDub Studio 2.0.ipynb`). The original 1.0 build (`pipeline.py` + `Colab_Runner.ipynb`) is referenced only where the two differ.

---

## 1. What 2.0 actually is

| Build | Launcher | Pipeline module | Engines published | Default engine |
|---|---|---|---|---|
| **AutoDub Studio 2.0** | `AutoDub Studio 2.0.ipynb` | **`pipeline2.py`** | **Chatterbox only** | **Chatterbox** |
| AutoDub Studio 2.0 · experimental twin | same, via the Tab 1 switcher | **`new_pipeline.py`** | Chatterbox only | Chatterbox |
| Original (1.0) | `Colab_Runner.ipynb` | `pipeline.py` | all six | CosyVoice 2.0 |

The binding happens at the top of `app.py`:

```python
_PIPELINE_NAME = (os.environ.get("AUTODUB_PIPELINE") or "pipeline").strip()
if not _is_module_name(_PIPELINE_NAME):
    raise RuntimeError(f"AUTODUB_PIPELINE must be a plain Python module name, …")
…
pipeline = _load_pipeline(_PIPELINE_NAME)     # importlib.import_module, cached
```

A non-identifier value is rejected before import, so the env var cannot be used to import arbitrary paths.

A second, **optional** variable — `AUTODUB_PIPELINE_VARIANTS`, comma-separated — turns on a **runtime switcher at the top of Tab 1**: a radio listing every name that both passes `_is_module_name()` *and* has a matching `.py` file next to `app.py`, wired to `_bind_pipeline()`.

`_bind_pipeline()` rebinds the module-global `pipeline`, and because every call site resolves `pipeline.` at **call** time, that one rebinding redirects Tab 1 analysis, Tab 2 matching, Tab 3 render, the auto-pilot and the pre-flight. It mirrors `_switch_tts_engine()` in re-deriving the engine radio and both engine-scoped language menus, so no widget keeps advertising options the new module can't serve.

- **Opt-in.** Leave the variable unset and no switcher renders — which is exactly why `Colab_Runner.ipynb` (which sets neither) keeps its original Tab 1 byte for byte.
- **Can't contradict reality.** The launch module is always offered first and is always the radio's default; anything else is appended. A name that is missing or malformed drops out instead of becoming a broken option.
- **Session state survives.** Both modules derive `BASE_DIR` from their own `__file__` in the same directory, so they share `outputs/state.json` and the rest of `outputs/`. A switch applies from the **next** stage run; a stage already iterating finishes on the module it started with.
- **Failure is contained.** `_load_pipeline()` caches per name, and loading is lazy, so a syntax error in the experimental twin is reported in the switcher's chip rather than preventing the app from booting. Rejection is a true no-op: the module active before the call stays active and the radio is pulled back to it.

`pipeline2.py` differs from `pipeline.py` by **exactly 31 diff lines** = 1 replaced header line + 18 added header lines + 2 identity constants (verified with `diff pipeline.py pipeline2.py | wc -l`):

```python
PIPELINE_VARIANT = "2.0"                    # 1.0 build: "1.0"
ENGINE_ALLOWLIST: tuple = ("chatterbox",)   # 1.0 build: ()
```

Because the allow-list narrows `TTS_ENGINES` in place (`pipeline2.py:2406-2415`), `DEFAULT_TTS_ENGINE = "cosyvoice2"` is no longer present in the registry, so `DEFAULT_TTS_ENGINE = next(iter(TTS_ENGINES))` resolves to **chatterbox**. One rule drives the Tab 1 radio, the Tab 2 audition dropdown, the default engine, and the fallback logic.

> ⚠️ **Consequence to keep in mind.** Several UI strings still read "CosyVoice". In 2.0 that code path can never execute — CosyVoice, Edge, Kokoro, Fish and CosyVoice 3 are all absent from the registry — but the wording is inaccurate. See §9.

---

## 2. Repository structure and file sizes

Measured from the working tree (`wc -l` + `ls -la`).

### Core code

| File | Bytes | Lines | Role in 2.0 |
|---|---:|---:|---|
| `pipeline2.py` | 210,870 | **4,568** | **The engine 2.0 runs.** 8 steps, engine registry, OOM defense, persistence |
| `new_pipeline.py` | 210,870 | 4,568 | Byte-identical twin of `pipeline2.py` at creation — the module Tab 1's switcher flips **to**, so it can be edited without touching `pipeline2.py` |
| `pipeline.py` | 209,516 | 4,550 | Original 1.0 engine — *not run by 2.0* (kept as the diff baseline) |
| `app.py` | 67,639 | 1,318 | Gradio glassmorphism UI, 3 tabs, auto-pilot, engine binding, Tab 1 pipeline switcher |
| `tools/selftest.py` | 85,369 | 1,766 | 330-check offline regression suite (no GPU required) |
| `tts_tester.py` | 40,969 | 888 | Standalone TTS probe harness |
| `fish_s2_probe.py` | 23,762 | 569 | Fish-S2 backend diagnostic |
| `build_secure.py` | 14,077 | 323 | Cython obfuscation builder |
| `tools/bridge.py` | 3,557 | 89 | Desktop tunnel bridge |
| `tools/_patch_ui.py` | 4,011 | 80 | One-off UI patcher — leftover, see §9 |

### Notebooks, docs, launchers

| File | Bytes | Notes |
|---|---:|---|
| `Colab_Runner.ipynb` | 37,648 | 8 cells · drives `pipeline.py` |
| `AutoDub Studio 2.0.ipynb` | 31,603 | **8 cells** · drives `pipeline2.py` |
| `TTS_Tester.ipynb` | 29,910 | 3 cells |
| `.kilo/plans/PROJECT_DOCUMENTATION.md` | 32,668 | Internal spec (a `REQUIREMENTS.md` referenced at `pipeline2.py:1663` **does not exist**) |
| `README.md` | 22,714 | 455 lines |
| `requirements.txt` | 3,177 | 48 lines |
| `tunnel.txt` | 1,076 | 10 lines |
| `Launcher.bat` / `Launcher.command` | 1,247 / 1,004 | Windows / macOS bridges |
| `assets/icons/*.svg` | 6 files, ~0.5 KB each | `analyze, brain, download, play, rocket, sparkles` |

### Generated at runtime (gitignored)

| Path | Size observed | Content |
|---|---:|---|
| `outputs/step1_audio.wav` | **44,729,934 B ≈ 42.7 MB** | 44.1 kHz / PCM-16 / stereo → **≈ 253 s ≈ 4 m 13 s** of source media |
| `uploads/` | empty | user media |
| `__pycache__/`, `tools/__pycache__/` | — | bytecode |

`.gitignore` excludes: `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.venv/`, `venv/`, **`uploads/`**, **`outputs/`**, `.gradio/`, `.DS_Store`, `Thumbs.db`, `.vscode/`, `.idea/`.


---

## 3. The 8 cells of `AutoDub Studio 2.0.ipynb`

| Cell | Purpose | What it does |
|---:|---|---|
| 0 | Title markdown | "🎬 AutoDub Studio 2.0 — Chatterbox-Only Colab Deployment" |
| 1 | 🩺 Diagnostics | Prints versions of 19 packages, GPU name + VRAM + CUDA, `/content` free disk, RAM, `ffmpeg` path, and probes the HF gate on both gated pyannote repos |
| 2 | Pinned ML-stack install | Checks 9 exact pins, installs only what is missing, **fail-soft** (prints the real pip error and keeps "Run all" going); then detects the missing `demucs.api` and reinstalls demucs **from GitHub source** |
| 3 | Clone / pull | Idempotent clone to `ai-video-dubber`, then `git pull --ff-only` → **re-running picks up a pushed fix without a new runtime** |
| 4 | gradio-client realign + audit | Fixes Colab's stale `gradio-client 1.3.0` (bool-schema `/api/info` crash) to gradio's own pin |
| 5 (4b) | **Cache check / restore** | HF-Hub → Google-Drive priority; explicit 401-vs-empty reporting |
| 6 | **Persistence + pre-download** | Drive/HF persistence, then pre-downloads SenseVoice, the 6 Chatterbox files, and both pyannote repos |
| 7 | 🚀 Launch | Sets `AUTODUB_PIPELINE=pipeline2` (which build runs) and `AUTODUB_PIPELINE_VARIANTS=pipeline2,new_pipeline` (turns on Tab 1's switcher), starts `app.py` → prints the `https://….gradio.live` link |

---

## 4. The 8 pipeline steps at a glance

| Step | Name | Tool / model | Device | Writes |
|---|---|---|---|---|
| 1 | `extract_audio` | FFmpeg → MoviePy fallback | CPU | `step1_audio.wav` |
| 2 | `separate_vocals` | **Demucs v4 `htdemucs`** | GPU → flush | `vocals.wav` + `music.wav` |
| 3 | `diarization` + mining | **Pyannote 3.1** + pYIN gender | GPU → flush | `diarization_map.json`, `speakerX_clone_prompt.wav`, `speaker_profiles.json` |
| 4 | `emotion_analysis` | **SenseVoice-Small** (FunASR) | GPU → flush | `emotion_log.txt`, `emotion_grid.json`, `clone_prompt_transcripts.json` |
| 5 | `assemble_script` | **pysrt** + optional indic-transliteration | CPU | `final_script.json` |
| 6 | `split_speaker_scripts` | custom sanitiser | CPU | `speaker_scripts.json`, `Project_<Speaker>.txt` |
| 7 | `synthesize` | **Chatterbox** (+ librosa stretch) | GPU → flush | `track_speakerN.wav`, `tts_report.json` |
| 8 | `mixdown` | numpy + soundfile + FFmpeg | CPU | `final_mix.wav`, `final_dubbed.mp4` |

**Steps 3 and 4 differ by build.** The table above is `pipeline2.py` (the default). `new_pipeline.py` — selectable from the Tab 1 switcher — swaps exactly those two rows for an ungated stack and leaves Steps 1, 2 and 5–8 identical:

| Step | `pipeline2.py` (default) | `new_pipeline.py` (experimental) |
|---|---|---|
| 3 | **Pyannote 3.1** + librosa pYIN gender | **Silero-VAD → FunASR CAM++ 192-d → scikit-learn agglomerative** + librosa pYIN gender |
| 4 | **SenseVoice-Small** | **`iic/emotion2vec_plus_large`** |
| gate | Pyannote repos are **gated** — an HF token is required | **none** — `NEEDS_HF_TOKEN = False` |

`STEP34_TOOLS`, `DIAR_ENGINE` and `NEEDS_HF_TOKEN` are published by each module, and app.py's Tab 1 chip reads them, so the UI always names the stack that will actually run. The JSON artifacts (`diarization_map.json`, `emotion_grid.json`) keep their exact shape across both builds — Steps 5–7 read them by key and cannot tell the swap happened.

**Design rule** (`pipeline2.py:32-35`): *exactly one heavy model lives on the GPU at any moment.* Every model step ends with `clear_gpu_cache()` (`pipeline2.py:213`) = drop reference → `gc.collect()` → `torch.cuda.empty_cache()`. All heavy imports are **lazy, inside step functions**, so importing the module is instant and the UI opens even without torch installed.

---

## 5. Step-by-step: process, tool, and links

### STEP 1 · Extract audio — FFmpeg / MoviePy · CPU

- Container sniff by extension (`VIDEO_EXTS` = mp4/mkv/mov/avi/webm/m4v/mpg/mpeg/ts/flv/wmv/3gp; `AUDIO_EXTS` = mp3/wav/flac/m4a/aac/ogg/opus/wma).
- **Primary:** `ffmpeg -i IN -vn -acodec pcm_s16le -ar 44100 -ac 2 step1_audio.wav` — lossless PCM-16 @ 44.1 kHz stereo, video never decoded.
- **Fallback** if no FFmpeg binary: MoviePy `VideoFileClip().audio` / `AudioFileClip()` (handles both moviepy 1.x and ≥2.0 import paths).
- Cached: any rerun short-circuits unless `force=True`.
- **Links:** [FFmpeg](https://ffmpeg.org/) · [github.com/FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) · [github.com/Zulko/moviepy](https://github.com/Zulko/moviepy) · [zulko.github.io/moviepy](https://zulko.github.io/moviepy/)
- **Package:** `moviepy==1.0.3` (notebook pin). Latest sdist 58.4 MB (measured PyPI); repo is MIT.

### STEP 2 · Vocal / music separation — Demucs v4 `htdemucs` · GPU

- **Path A:** `demucs.api.Separator(model="htdemucs", device=…, shifts=0)` → in-process.
- **Path B (fallback):** `demucs --two-stems=vocals -n htdemucs --shifts 0 -o outputs step1_audio.wav` → renames `vocals.wav`/`no_vocals.wav` to the pipeline's artifact names, deletes the temp tree.
- **Why path B exists:** PyPI `demucs==4.0.1` ships **without `api.py`** — it only exists in the GitHub source, which Cell 2 installs as a second step.
- Demucs writes 4 stems internally; only vocals + instrumental are kept.
- **Links:** [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs) · [dl.fbaipublicfiles.com/demucs](https://dl.fbaipublicfiles.com/demucs/) · MIT · ~10.4k★
- **Weights (measured):** `htdemucs` = **955717e8-8726e21a.th = 84.1 MB**, fetched to `~/.cache/torch/hub/checkpoints/`. (The `htdemucs_ft` variant is the other 84.1 MB file — not used here.)
- **Package:** `demucs==4.0.1`, 1.2 MB sdist.

### STEP 3 · Diarization + clone-prompt mining — Pyannote 3.1 · GPU

The richest step. Process:

1. **Compatibility shims before import:** `_torchaudio_compat()` (APIs removed in torchaudio ≥2.9), `_numpy2_compat()` (`np.NaN`/`np.float_` aliases for NumPy 2.x), `_hf_hub_compat()` (patches `hf_hub_download` to accept Pyannote's `use_auth_token`, then rebinds it into `requests`-style submodules), `_torch_load_compat()` (legacy `torch.load` default for trusted checkpoints).
2. **HF gate pre-flight** (`pipeline2.py:1224-1249`): before downloading ~1 GB it HTTP-probes `pyannote/speaker-diarization-3.1` **and** `pyannote/segmentation-3.0` with the token, so a denied gate fails instantly and specifically instead of as an opaque blob error after minutes.
3. `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=…)` with a `TypeError` → `use_auth_token` fallback for older `huggingface_hub`.
4. **Cross-reference with the user's ORIGINAL SRT:** for every cue, the speaker who owns the most milliseconds wins a max-overlap vote.
5. **Canonical identity tags:** raw `SPEAKER_00…` are ranked by total speech time → `Speaker1`, `Speaker2`… The "expected speakers" hint folds the smallest clusters into the temporally-nearest big cluster.
6. **Clone-prompt mining:** per speaker, the **top-3 cleanest 5–10 s windows** are located across the merged speech intervals and written as 16 kHz mono `outputs/speakerX_clone_prompt.wav`.
7. **Automatic gender detection** (`pipeline2.py:1380-1480`) — Pyannote emits **no gender field at all**, so without this every speaker got a female voice (the first pool key). Gender is recovered acoustically with **librosa pYIN** on each speaker's own clone prompt:
   - voiced/unvoiced flag first — `_MIN_VOICED_FRAC = 0.15`; below that → **UNDETERMINED**, never a guess.
   - median **F0** split: `_F0_MALE_MAX = 150 Hz`, `_F0_FEMALE_MIN = 175 Hz`; the band 150–175 Hz is ambiguous.
   - ambiguous band → long-term **spectral centroid** (`_CENTROID_MALE_MAX = 1500 Hz`, `_CENTROID_FEMALE_MIN = 2400 Hz`, wide dead zone) — literature-calibrated, **not validated against real speech** (stated in-code).
   - `_MAX_F0_IQR = 70 Hz` → a window mixing two speakers is rejected.
   - **Why pYIN, not YIN:** measured on synthetic input, plain YIN called **pure silence "female, 400 Hz"** and **white noise "male, 73 Hz"**, both at full confidence. pYIN rejected all four non-voice inputs. Both regressions are locked in `tools/selftest.py`.
   - A Tab 2 manual override always wins over the estimate.
8. Result → `diarization_map.json` = `{model, speakers:{…total_speech_s, cue_count, clone_prompts, gender/f0…}, cues:[…]}`.

- **Links:** [github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) · [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) · [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) · [huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) · MIT
- **Weights (measured):** `speaker-diarization-3.1` repo = **10.9 MB** (24 files, mostly benchmark `.rttm` — it is a config repo) · `segmentation-3.0` `pytorch_model.bin` = **5.9 MB** · `wespeaker-voxceleb-resnet34-LM` = **26.6 MB** (cc-by-4.0). Combined ≈ **32 MB** + code.
- **Gotcha documented in code:** the real gate download is ~1 GB because the pipeline pulls the segmentation + embedding stack. `pyannote.audio==3.1.1` is pinned exactly because 4.x exists upstream.

### STEP 4 · Emotion scan — SenseVoice-Small (FunASR) · GPU

- `AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time":30_000}, device=…, trust_remote_code=True, disable_update=True, disable_pbar=True)`.
- Runs **exclusively over the diarized cue windows** (not the whole file), on `vocals.wav` resampled to 16 kHz mono.
- Parses rich-transcription tags `<|HAPPY|>` etc. through `SENSEVOICE_EMO_MAP` into the **7 canonical emotions**: `happy, sad, angry, surprised, neutral, fearful, disgusted` (`EXCITED→happy`, `FEARSONE→fearful`, `EMO_UNKNOWN→neutral`).
- Produces:
  - `emotion_log.txt` — spec format `[00:50.000] Speaker1 [calm]` (`neutral` reads as "calm" for humans)
  - `emotion_grid.json` — the machine grid consumed by Step 5
  - and it transcribes each speaker's **cleanest clone prompt** for Step 7 conditioning (`clone_prompt_transcripts.json`)
- Writes detailed guidance to `outputs/engine_error.log` instead of a bare `No module named …` when funasr/modelscope are missing.
- **Links:** [github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) · [github.com/modelscope/FunASR](https://github.com/modelscope/FunASR) · [huggingface.co/FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) · [modelscope.cn/models/iic/SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall) · [huggingface.co/funasr/fsmn-vad](https://huggingface.co/funasr/fsmn-vad) · SenseVoice MIT / FunASR MIT / repo license "other"
- **Weights (measured):** `FunAudioLLM/SenseVoiceSmall` repo = **944 MB**, of which **`model.pt` = 936.3 MB**. `fsmn-vad` = **4 MB**.
- **Pins:** `funasr==1.2.6`, `modelscope==1.27.1` (notebook); `requirements.txt` allows `funasr>=1.1.4,<2`.

### STEP 5 · Script assembly — `pysrt` + transliteration · CPU

- Loads the **TRANSLATED** SRT, then aligns each translated cue to an analysed cue: **index-aligned first**, and if `|Δstart| > 1.5 s` it searches a **±10-cue window for max temporal overlap**.
- Merges speaker + emotion + clone-prompt + ASR language into one row:
  `{index, start, end, speaker, emotion, instruct, language, target_language, clone_prompt, original_text, translated_text}`.
- `instruct` comes from `EMOTION_INSTRUCT` (e.g. `"in a calm, neutral tone"`).
- Optional **Roman → Devanagari** transliteration (`indic-transliteration`) gated by the Tab 1 checkbox.
- Output `final_script.json` — this is what the Tab 2 table displays.
- **Links:** [github.com/byroot/pysrt](https://github.com/byroot/pysrt) · [github.com/indic-transliteration/indic_transliteration_py](https://github.com/indic-transliteration/indic_transliteration_py) · pysrt GPL-3.0, transliteration MIT
- **Packages (measured):** `pysrt==1.1.2` (0.1 MB), `indic-transliteration` (0.2 MB wheel).

### STEP 6 · Speaker text splitting — custom algorithmic engine · CPU

- For every row: `sanitize_for_tts()` strips speaker labels, timecodes and metadata. **The TTS engine sees only `text` — never a label it could read aloud.**
- Groups into `per_speaker`, sorts chronologically inside each channel, and writes **`Project_<Speaker>.txt` = pure speakable lines, no labels, no timecodes**.
- Rows that empty out after sanitising are logged and skipped (not silently dropped).
- Output `speaker_scripts.json`.
- **Tool:** none — pure Python. This is the "custom algorithmic engine" row in the README table.

### STEP 7 · Emotional zero-shot TTS — **Chatterbox** (+ librosa) · GPU

In 2.0 this is Chatterbox-only. Mechanism:

1. **Engine bring-up** via `EngineRuntime`; `resolve_engine()` reads the Tab 1 radio.
2. **Silent-master timeline:** one `np.zeros(duration × sr)` float32 array **per speaker**, exactly matching the original media clock. No clock drift is possible because nothing is concatenated.
3. **Clone prompt** = the speaker's mined `speakerX_clone_prompt.wav`; a cloning engine refuses to skip silently if one is missing (counted as a failed row).
4. **Per-line synthesis** — `ChatterboxMultilingualTTS.generate(text, language_id=…, audio_prompt_path=…, exaggeration=…, cfg_weight=0.0)`:
   - `language_id` must be inside Chatterbox's own `SUPPORTED_LANGUAGES` — offering "yue" (Cantonese) to it would raise `ValueError` one line in. Hence `ENGINE_LANGUAGES["chatterbox"]` (all 23, verbatim) and `language_choices()` deriving the UI menu from the *active engine*, so the UI can never advertise a language the engine will reject.
   - **`cfg_weight=0.0` is deliberate:** clone prompts are recorded in the *source* language but we dub into Hindi, and Resemble advise 0 to stop the reference clip's accent bleeding through. Older builds without `cfg_weight` get a `TypeError` → the kwarg is dropped.
   - **Emotion → `exaggeration` scalar (0–2)** via `_CHATTER_EXAG`: happy 0.70, sad 0.35, angry 0.80, surprised 0.75, neutral 0.40, fearful 0.70, disgusted 0.65.
   - Every generated file carries a **Resemble PerTh neural watermark** (`import perth` is a hard top-level dependency of the wheel).
5. **Overpressure fix:** if a clip overruns its slot (next line's start − this start, floor 0.5 s) it is time-stretched with `librosa.effects.time_stretch` (non-pitch-shifting) up to **1.15×**; still too long → hard snap + 30 ms fade. 5 ms de-click fades on both edges of every segment.
6. **VRAM teardown**, then a **hard guard**: if `report["rows"]` is empty it raises `TerminalEngineError` rather than letting Step 8 mix an all-silent master.
7. Writes `track_<speaker>.wav` (FLOAT subtype, at master SR) + `tts_report.json` (per-row start/available/generated/speed/trimmed, engine id, elapsed). Changing the engine invalidates the cache and re-renders.
8. **Bootstrap:** `chatterbox-tts` is installed **lazily on demand with `--no-deps`** plus only its true runtime deps, because a plain install would pin `numpy<2`, `torch==2.6.0`, `torchaudio==2.6.0`, `transformers==5.2.0`, `gradio==6.8.0` and downgrade the whole stack. It is **TERMINAL**: no fallback, ever — a silent voice swap is the bug you would want to see.

- **Links:** [huggingface.co/ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) · [huggingface.co/ResembleAI/Chatterbox-Multilingual-hi](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-hi) · [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) · [github.com/resemble-ai/PerTh](https://github.com/resemble-ai/PerTh) · MIT
- **Weights (measured)** — the 6 files the pipeline actually snapshots:
  `ve.pt` **5.7 MB** · `t3_mtl23ls_v2.safetensors` **2,144.0 MB** · `s3gen.pt` **1,057.2 MB** · `Cangjie5_TC.json` **1.9 MB** · `grapheme_mtl_merged_expanded_v1.json` **0.1 MB** · `conds.pt` **0.1 MB** → **≈ 3,209 MB (3.2 GB)**.
  The full repo is 23,651 MB across 18 files (it keeps v2/v3/23-lang T3s and `.safetensors`+`.pt` duplicates) — `allow_patterns` avoids paying for that.
  Opt-in Hindi finetune overlay `Chatterbox-Multilingual-hi`: `t3_hi.safetensors` **2,144.0 MB** (+ `s3gen_v3` 1,056.9/1,056.4 MB) — **off by default** (`t3_model: ""`), because Hindi is already one of the base 23. When enabled, `_chatterbox_overlay_dir` builds a *symlink overlay* (falling back to hard link → copy) so the finetune sits under the wheel's hardcoded `t3_mtl23ls_v2.safetensors` name **without** corrupting the shared HF cache.
- **Gate:** public, no token needed.
- **librosa** (Step 7 stretch): [github.com/librosa/librosa](https://github.com/librosa/librosa) · ISC · pin `0.11.0`, 0.4 MB.

### STEP 8 · Mixdown, ducking & remux — numpy / soundfile / FFmpeg · CPU

1. Load `music.wav` stereo at master SR.
2. Sum every `track_speaker*.wav` into one mono voice bus (resampling any track that disagrees via torchaudio, with a numpy-linear fallback).
3. **Voice-activity envelope** at 10 ms frames: RMS per frame, threshold `max(1e-4, 0.06 × max)`.
4. **Target gain** = `10^(−6/20) = 0.5012` where active, `1.0` where silent.
5. **Slew-rate-limited curve** for smooth attack (90 ms) / release (300 ms).
6. **Lookahead 120 ms** pre-duck: the curve is shifted earlier so the dip lands *before* each voice onset.
7. `final = music × gain + voices`; peak-normalise to 0.99 if it clips.
8. `final_mix.wav` (PCM-16).
9. **Remux:** if the source was video →
   `ffmpeg -i SRC -i final_mix.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest final_dubbed.mp4`
   → **stream copy, zero video re-encode**. A remux failure is a warning; the audio master is still kept.
- **Links:** [numpy.org](https://numpy.org/) · [python-soundfile.readthedocs.io](https://python-soundfile.readthedocs.io/) · [ffmpeg.org](https://ffmpeg.org/) · numpy BSD-3, soundfile BSD-3


---

## 6. Model / weight inventory for 2.0 — measured sizes

| # | Model | Repo | Size (measured) | License | Gate |
|---|---|---|---|---:|---|
| 1 | Demucs v4 `htdemucs` | `dl.fbaipublicfiles.com/demucs` | **84.1 MB** (`955717e8-8726e21a.th`) | MIT | none |
| 2 | Pyannote diarization | `pyannote/speaker-diarization-3.1` | **10.9 MB** (config repo) | MIT | **gated (auto)** |
| 3 | Pyannote segmentation | `pyannote/segmentation-3.0` | **5.9 MB** | MIT | **gated (auto)** |
| 4 | Speaker embedding | `pyannote/wespeaker-voxceleb-resnet34-LM` | **26.6 MB** | cc-by-4.0 | none |
| 5 | VAD | `funasr/fsmn-vad` | **4 MB** | Apache-2.0 | none |
| 6 | SenseVoice-Small | `FunAudioLLM/SenseVoiceSmall` (= `iic/SenseVoiceSmall` on ModelScope) | **936.3 MB** (`model.pt`) | other (MIT repo) | none |
| 7 | **Chatterbox v2 T3** | `ResembleAI/chatterbox` | **2,144.0 MB** | MIT | none |
| 8 | Chatterbox s3gen | `ResembleAI/chatterbox` | **1,057.2 MB** | MIT | none |
| 9 | Chatterbox ve / conds / grapheme / Cangjie | `ResembleAI/chatterbox` | **7.8 MB** combined | MIT | none |
| 10 | *(opt-in)* Hindi T3 finetune | `ResembleAI/Chatterbox-Multilingual-hi` | 2,144.0 MB | MIT | none |

**Rows 2–6 are the *default* build.** `new_pipeline.py` (Tab 1 switcher) substitutes Steps 3 and 4 only, with everything ungated:

| # | Model | Repo | Size (measured) | License | Gate |
|---|---|---|---|---|---:|
| 3a | Silero-VAD | PyPI `silero_vad` 6.2.3 wheel (bundles JIT + ONNX + states) | **11.3 MB** (11,317,527 B) | MIT | none |
| 3b | CAM++ speaker embedding | `iic/speech_campplus_sv_en_voxceleb_16k` (ModelScope) | **30.4 MB** (30,431,202 B) | Apache-2.0 | none |
| 4a | emotion2vec+ large | HF `emotion2vec/emotion2vec_plus_large` → `model.pt` | **1,945.8 MB** (1,945,790,254 B) | FunASR model-license | none |

⚠️ **The saving here is gating, not weight.** Measured against the default build:

- **Step 3 is size-neutral:** 43.4 MB gated (10.9 + 5.9 + 26.6) → 41.7 MB ungated (11.3 + 30.4). What disappears is the HF token and the two "Agree and access repository" checkboxes.
- **Step 4 gets BIGGER, not smaller:** 936.3 MB → 1,945.8 MB = **+1,009.5 MB**, in exchange for emotion2vec+'s **9** classes where the ModelScope path would return 4.

Net model cache for `new_pipeline.py` is therefore roughly **+1.0 GB** versus `pipeline2.py` (~4.27 GB → ~5.3 GB), while Steps 3–4 become completely gate-free. The previously quoted "−1 GB pyannote, −944 MB SenseVoice" is **wrong on both counts** and should not be reused: Pyannote's weights are ~43 MB, not 1 GB (the ~1 GB figure in §5 refers to the whole `pyannote.audio` dependency install, not the checkpoints), and emotion2vec+ costs more than SenseVoice saves.

Sizes above come from the HF API (`usedStorage` / per-file `size`) and the ModelScope API (`StorageSize`), not from an estimate.

**Steady-state 2.0 model cache ≈ 4,183 MB (4.08 GiB)** + 84 MB Demucs ⇒ **≈ 4.27 GB (~4.2 GiB)**. Two independent confirmations:

- `README.md:405` states "Model caches (one-time) **~5 GB**" — close, and slightly conservative.
- The pre-download cell in `AutoDub Studio 2.0.ipynb` downloads exactly these three groups, so that cell's own printout is the authoritative per-session number.

**Not downloaded in 2.0 but cached if you switch builds:** CosyVoice2-0.5B **4,857 MB** (`llm.pt` 2,023.3 MB), Kokoro-82M **~363 MB** working set (`kokoro-v1_0.pth` 327.2 MB + 54 voice tensors; the repo reports 1,235 MB used storage including LFS history), Fish `s2-pro` **11,012 MB** (two shards 4,986.9 + 4,136.9 MB + `codec.pth` 1,871.1 MB).

Per-job artifacts: `README.md:406` says **~0.3 GB per 5-min job — local VM only, never uploaded**. The observed 4 m 13 s `step1_audio.wav` is 42.7 MB, consistent with that once stems + tracks + masters are counted.

---

## 7. Tool / package inventory & sizes

### Pinned by `AutoDub Studio 2.0.ipynb` Cell 2 (installed only if missing)

| Package | Pin | Latest on PyPI (measured) | Notes |
|---|---|---|---|
| `demucs` | **4.0.1** | 4.1.0 · 1.2 MB | `api.py` missing → GitHub-source second install |
| `pyannote.audio` | **3.1.1** | 4.0.7 · 14.0 MB | pinned exact, 4.x exists |
| `funasr` | **1.2.6** | 1.4.16 · 1.0 MB | |
| `modelscope` | **1.27.1** | 1.40.1 · 6.1 MB | model hub for SenseVoice |
| `pysrt` | **1.1.2** | 1.1.2 · 0.1 MB | |
| `pydub` | **0.25.1** | 0.25.1 · ~0 MB | |
| `moviepy` | **1.0.3** | 2.2.1 · 58.4 MB | Step 1 fallback |
| `ffmpeg-python` | **0.2.0** | 0.2.0 · ~0 MB | |
| `librosa` | **0.11.0** | 1.0.0 · 0.4 MB | pYIN + time-stretch |

`torch` / `torchaudio` / `numpy` / `soundfile` are deliberately **excluded** — Colab's own CUDA builds must survive (the torch wheel alone is 554.6 MB on PyPI; Colab's CUDA build is far larger installed).

### `requirements.txt` extras

`gradio>=4.44.0` (latest 6.29.0 · 45.1 MB) · `huggingface_hub>=0.25` · `modelscope>=1.17,<2` · `indic-transliteration` · `soundfile` · `pandas` · `numpy` · `tqdm` · `scipy` (via librosa) · `setuptools>=68`.

### Lazy, on-demand, `--no-deps`

| Package | Why lazy | Wheel size (measured) |
|---|---|---:|
| `chatterbox-tts==0.1.7` | pins numpy<2 / torch 2.6.0 / torchaudio 2.6.0 / transformers 5.2.0 / gradio 6.8.0 | 0.1 MB |
| `resemble-perth` | **mandatory** — `import perth` is top-level in the wheel | 34.4 MB |
| `s3tokenizer`, `conformer`, `safetensors`, `omegaconf`, `pyloudnorm`, `pykakasi`, `spacy-pkuseg`, `diffusers`, `librosa>=0.10` | chatterbox's true runtime deps | 0.2 / ~0 / — / 3.3 / ~0 / 21.8 / 5.2 / 5.9 MB |
| `kokoro` | needs the **espeak-ng system package** for Hindi G2P — pip cannot express it | 26.2 MB |
| `edge-tts` | cloud, zero weights | ~0 MB |

Notebook note: pip cannot install the system `espeak-ng`, and both chatterbox/kokoro are **terminal engines** — a failure produces a detailed report in `outputs/engine_error.log`, never a silent substitute.

### Runtime environment

- **Target:** Google Colab free **T4 — 16 GB VRAM** (stated `pipeline2.py:3`). CPU-only works but steps 2–8 are slow.
- **UI:** `gradio` (latest 6.29.0), glassmorphism theme + 6 inline SVG icons.
- **Also present:** `tts_tester.py`, `TTS_Tester.ipynb` (engine audition), `fish_s2_probe.py`, `build_secure.py` (Cython obfuscator — `README.md:364` reports it translates `pipeline.py` to 7.4 MB of C), `tools/bridge.py`, `Launcher.bat` / `Launcher.command`.


---

## 8. Storage, persistence and caching

| Artifact | Size | Where |
|---|---|---|
| Model cache | **≈ 4.2 GB** (measured) | cloud if persistence ON, else ephemeral VM |
| Per 5-min job | ~0.3 GB | **local VM only — never uploaded** |
| Uploaded media | as uploaded | local VM only |

Three modes (notebook Cell 6 / Tab 1 ▸ Advanced), default **OFF**:

- **none** — ephemeral; after a disconnect models re-download (~15 min) and steps re-run.
- **drive** — `/content/drive/MyDrive/AutoDub_Studio/model_cache`; `outputs/` + `uploads/` become **symlinks** to Drive so a disconnect does not lose progress.
- **hf** — private dataset repo `<user>/autodub-model-cache`; pulled at session start, pushed after setup. `HF_HOME` / `MODELSCOPE_CACHE` redirect the caches.

Auth subtleties documented in `README.md:433-455`: the repo is **private**, so restore must authenticate; an unauthorised pull returns **401, which looks exactly like "empty storage"** — hence the distinct messages `TOKEN REJECTED` / `token has NO ACCESS (401)` / `accessible but EMPTY (0 files)` / `no such private dataset`. Use one **write** token everywhere.

---

## 9. Findings — stale or inconsistent items (verified, not guessed)

1. **`tools/_patch_ui.py` (80 lines, 4,011 B) is dead weight.** A one-shot UI patcher left in the tree; it is not imported by `app.py` or the selftest.
2. **`pipeline2.py` `run_rendering()` docstring still says "split → CosyVoice TTS → ducked mixdown"** — stale in a Chatterbox-only build.
3. **`app.py` Tab 3 copy says "clones every line with CosyVoice 3.0 zero-shot"** — 2.0 can never run CosyVoice 3.
4. **`requirements.txt` comments describe six engines** and name `pipeline._load_chatterbox / _load_kokoro`; accurate for 1.0, only half-true for 2.0.
5. **Two contradictory "cloud storage" sections in `README.md`** — the table at `:403-413` says model caches go to cloud *"if persistence ON"*, while the section at `:429-431` says cloud holds model caches categorically. They should be merged to "only when persistence is enabled".
6. **README HF-setup bullet says `segmentation-3.1`** while the bullet's own link and the runtime pre-flight (`pipeline2.py:1229`) both use **`segmentation-3.0`**. One-word fix: `3.1` → `3.0`.
7. **`README.md` references `REQUIREMENTS.md`** (line 495) which **does not exist** in the repo; the spec now lives at `.kilo/plans/PROJECT_DOCUMENTATION.md`.
8. **`app.py`'s Tab 3 description quotes `1.15×` and "−6 dB" thresholds** — correct values, but hardcoded prose that will silently rot if the parameters change.
9. **Repo/README framing:** the project name says "ELEVENLABS Quality" but every Step 7 engine is open-source (Chatterbox MIT here). The name describes the *target quality bar*, not a dependency — worth stating explicitly so nobody hunts for a missing `ELEVENLABS_API_KEY`.

---

## 10. Appendix — cross-reference map

| UI surface | Step(s) | Key functions (`pipeline2.py`) | Artifacts | Cache file(s) |
|---|---|---|---|---|
| Tab 1 · File Import & Analysis | 1–5 | `extract_audio`, `separate_vocals`, `diarization`, `emotion_analysis`, `assemble_script` | `step1_audio.wav`, `vocals.wav`, `music.wav`, `diarization_map.json`, `speakerX_clone_prompt.wav`, `emotion_log.txt`, `emotion_grid.json`, `final_script.json` | Demucs `.th`, Pyannote 3 repos, SenseVoice `model.pt`, `fsmn-vad` |
| Tab 2 · Script Matching & Assembly | 5–6 | `assemble_script`, `split_speaker_scripts`, `preview_voice`, `_estimate_gender` | `final_script.json`, `speaker_scripts.json`, `Project_<Speaker>.txt`, `speaker_profiles.json` | — (CPU only) |
| Tab 3 · Rendering Engine | 6–8 | `split_speaker_scripts`, `synthesize`, `mixdown` | `speaker_scripts.json`, `track_speakerN.wav`, `tts_report.json`, `final_mix.wav`, `final_dubbed.mp4` | Chatterbox 6 files (+ optional `t3_hi`) |

**Engine resolution chain (2.0):**

```
Tab 1 radio  →  resolve_engine()  →  TTS_ENGINES  →  EngineRuntime
                                          │
                        ENGINE_ALLOWLIST = ("chatterbox",)
                                          │
                              only "chatterbox" survives
                                          │
                        TERMINAL — failure raises TerminalEngineError,
                        writes outputs/engine_error.log, Step 8 skipped
```

**Verified invariants (asserted by `tools/selftest.py`, 330 checks):**

- `diff pipeline.py pipeline2.py` = 31 lines (1 replaced + 18 added + 2 constants)
- `PIPELINE_VARIANT` / `ENGINE_ALLOWLIST` are the only identity differences
- `TTS_ENGINES` in 2.0 = `("chatterbox",)`; default engine = `chatterbox`
- `language_choices()` returns `(label, value)` — Gradio reads element `[1]`
- Chatterbox's 23-language set matches the wheel's `SUPPORTED_LANGUAGES` exactly
- No `SwaraNeural` hardcoded default remains in the casting path
- Tab 1's switcher renders **only** when `AUTODUB_PIPELINE_VARIANTS` names >1 existing module; with no such variable, Tab 1 has neither the control nor its handler
- Flipping it rebinds the module-global `pipeline`, rewrites both language menus, and refreshes the engine radio + both status chips; a rejected name leaves the active module running and pulls the radio back to it

