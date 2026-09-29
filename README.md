# 🎬 AutoDub Studio — Automatic Dubbing Engine (ElevenLabs-Quality, Open-Source)

A **free, open-source automatic dubbing pipeline** designed to run on a **Google Colab
free T4 GPU**, with an **iPhone-15 frosted-glass (glassmorphism) Gradio UI**.
It isolates voices, maps every speaker & emotion, mines voice-clone prompts,
and assembles an emotion-tagged multilingual dubbing script — ready for
zero-shot emotional TTS.

---

## 🧩 The 8-Step Open-Source Stack

| Step | What happens | Tool | Status |
|------|--------------|------|--------|
| 1 | Extract audio from video | **FFmpeg / MoviePy** | ✅ Phase 2 |
| 2 | Separate vocals & music | **Demucs v4** (Meta) | ✅ Phase 2 |
| 3 | Speaker diarization | **Pyannote 3.1** | ✅ Phase 3 |
| 4 | Emotion detection (7 emotions) | **SenseVoice-Small** (FunASR) | ✅ Phase 3 |
| 5 | SRT mapping & merging | **pysrt** | ✅ Phase 3 |
| 6 | Speaker text splitting (TTS-safe) | custom algorithmic engine | ✅ Phase 4 |
| 7 | Multilingual emotional TTS | **CosyVoice 2.0 / 3.0 · Chatterbox · Fish S2-Pro · Edge-TTS** (5-way selector) | ✅ Phase 4 |
| 8 | Master mixdown + ducking + remux | **numpy / FFmpeg** | ✅ Phase 5 |

---

## 🚀 Quickstart

### Google Colab (recommended — T4 GPU)

```python
# 1 ▸ Upload the project folder (or clone your repo) into Colab
# 2 ▸ In a notebook cell:
!pip install -q -r requirements.txt
!python app.py
```

The app auto-detects Colab and publishes a **public `*.gradio.live` link**.
Pick the **T4 GPU** runtime: *Runtime ▸ Change runtime type ▸ T4 GPU*.

### Local machine

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121   # or CPU wheel
pip install -r requirements.txt
python app.py
```

> ℹ️ FFmpeg is auto-detected; if missing, MoviePy extraction is used as a fallback.
> CPU-only machines work but model steps are slow — Colab T4 is the target.

### 🧪 Testing locally without a GPU?

Steps 2–8 need the ML stack. For a local CPU test run:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install demucs pyannote.audio funasr modelscope librosa
```

Each step now prints **exact install guidance** if something is missing —
the console will never just say `No module named 'torch'`. Full local
(CPU) runs work but are slow; Colab's T4 remains the intended runtime.
Step 3 additionally requires the Hugging Face token (see below).


---

## 🔑 One-time Hugging Face setup (required for Step 3)

Pyannote 3.1 is gated. With the account that owns your token:

1. Accept conditions at **huggingface.co/pyannote/speaker-diarization-3.1**
2. Accept conditions at **huggingface.co/pyannote/segmentation-3.1**
3. Create a token at **huggingface.co/settings/tokens** (read access)
4. Paste it in the app: **Tab 1 ▸ 🔑 Advanced ▸ HF access token**

---

## 🖥 Using the app

| Tab | What it does | Pipeline steps |
|-----|--------------|----------------|
| 📂 **File Import & Analysis** | Upload the audio/video master + Original SRT + Translated SRT, then hit the 🔍 button | 1–2 (extract → Demucs split) |
| 📝 **Script Matching & Assembly** | Hit the 🧬 button — diarizes speakers, mines clone prompts, scans emotions, merges the translated SRT | 3–5 |
| 🚀 **Rendering Engine** | Hit ▶ to **render the full dub** — or ✨ for the artifact pre-flight | 6–8 |

### 🚀 The render stage (Steps 6–8)

- **Step 6 · Splitting engine** — clones the consolidated array into per-speaker
  channels (`Project_SpeakerN.txt`) with every speaker label & timecode stripped,
  so the TTS voice can never read metadata aloud
- **Step 7 · Pluggable TTS** (engine selected in Tab 1) — builds a **silent
  zero-signal master array per speaker**, exactly matching the original media
  duration; each line is cloned with the speaker's voice profile + emotion tags
  (English · Hindi · Spanish). **Overpressure fix:** if a clip would bleed into
  the next line, it is time-stretched with `librosa.effects.time_stretch`
  (non-pitch-shifting, up to **1.15×**) so it snaps inside its slot without
  moving the clock
- **Step 8 · Mixdown & ducking** — layers every speaker master over the Demucs
  instrumental with a **lookahead −6 dB ducking curve** (smooth linear
  attack/release), exports `final_mix.wav`, and — when the source was video —
  stream-copies the master audio back behind the original video →
  `final_dubbed.mp4`

> ℹ️ **CosyVoice bootstrap (once, in Colab):**
> `git clone https://github.com/FunAudioLLM/CosyVoice && cd CosyVoice && git submodule update --init --recursive && pip install -r requirements.txt && pip install -e .`

Everything streams live into a glass **pipeline console**; vocals/music previews
appear right under Tab 1, and the consolidated script table + emotion log +
clone prompts appear in Tab 2.

---

## 🎙 Pluggable TTS engines (5-way selector · Tab 1)

Pick the synthesizer at the very top of **Tab 1**. Every engine shares the same
per-speaker clone prompts and the same silent-master timeline, so swapping an
engine only changes *who* speaks — never *when*.

| Engine | Params | VRAM | Zero-shot clone | Fallback |
|---|---|---|---|---|
| **CosyVoice 2.0** *(default)* | 0.5 B | ~2 GB | ✅ | Edge-TTS |
| **CosyVoice 3.0** | 0.5 B | ~2 GB | ✅ | Edge-TTS |
| **Chatterbox** (Resemble AI) | 0.5 B | ~2 GB | ✅ | Edge-TTS |
| **Fish Audio S2-Pro** | — | 4–12 GB | ✅ | Edge-TTS |
| **Edge-TTS** *(lightweight)* | cloud | 0 | ❌ | — terminal — |

**Automatic fallback.** Every non-Edge engine degrades to **Edge-TTS** when it
fails to load (missing package, OOM, absent weights) *or* after **3 consecutive
line failures** mid-render. Edge-TTS is the **terminal node** — if Edge itself
fails, the error is surfaced instead of silently substituting a different voice.
Changing the engine in Tab 1 invalidates the Step 7 cache and re-renders
automatically.

**Distinct voice per speaker.** Edge-TTS cannot clone, so voices are cast per
speaker instead — deterministically, in name order, matched to the diarized
gender. Microsoft ships only **two** true Hindi neural voices, so the pool is
tiered by phonetic fidelity (`hi-IN` → `mr-IN`/`ne-NP`, which are written in
Devanagari and read Hindi natively → other Indic locales, flagged in the console
because they *will* mispronounce Devanagari). That gives **15 female + 15 male**
verified voices, of which 3 per gender are Devanagari-faithful.

**Cross-lingual guard.** Clone prompts are recorded in the *source* language but
we dub *into Hindi*, so Chatterbox is driven with `cfg_weight=0.0` to stop the
reference clip's accent bleeding into the Hindi output.

**Emotion mapping.** SenseVoice's 7 emotions drive each engine natively:
CosyVoice takes a separate `instruct` string, Chatterbox maps to its
`exaggeration` scalar (0–2), Fish embeds `[tag]` inline, and Edge has no
emotion control.

> ⚠️ **Licence & watermark notice** — Chatterbox stamps a Resemble **PerTh**
> neural watermark into every clip it generates. Fish Audio S2-Pro output falls
> under the **Fish Audio Research License (non-commercial)**. Edge-TTS audio is
> synthesised remotely by Microsoft.

Validate the heaviest engine before committing to it:

```bash
python fish_s2_probe.py --all
```


---

## 🧠 Memory defense (Colab T4 OOM protection)

`pipeline.py` enforces a **sequential execution model** — exactly one heavy
model on the GPU at any moment. After every GPU stage, `clear_gpu_cache()`
drops the model, sweeps the Python garbage collector, and calls
`torch.cuda.empty_cache()` before the next stage boots. Completed stages are
cached on disk (`outputs/state.json`) and **auto-skip** on re-runs; uploading
a *new* source file invalidates the chain automatically.

```
Step 1 extract ──▶ Step 2 Demucs ──🧹──▶ Step 3 Pyannote ──🧹──▶ Step 4 SenseVoice ──🧹──▶ Step 5 assemble (CPU)
```

---

## 📁 Project layout

```
├── app.py                 # Phase 1 · glassmorphism Gradio UI (3 tabs)
├── pipeline.py            # Phases 2–3 · steps 1–5 + clear_gpu_cache OOM defense
├── requirements.txt
├── assets/icons/          # premium gradient SVG button icons
├── uploads/               # runtime · user uploads
└── outputs/               # runtime artifacts
    ├── step1_audio.wav          # extracted master audio
    ├── vocals.wav / music.wav   # Demucs stems
    ├── diarization_map.json     # Speaker1/2/… per SRT cue
    ├── speakerX_clone_prompt.wav# top-3 cleanest 5–10 s refs per speaker
    ├── emotion_log.txt          # "[00:50.000] Speaker1 [calm]"
    ├── emotion_grid.json        # machine emotion grid
    ├── final_script.json        # consolidated dubbing script ★
    └── state.json               # pipeline stage cache
```

**`final_script.json` row shape (consumed by the render phase):**

```json
{
  "index": 3, "start": 12.34, "end": 15.2,
  "speaker": "Speaker1", "emotion": "happy",
  "instruct": "in a happy, upbeat tone", "language": "en",
  "clone_prompt": "outputs/speaker1_clone_prompt.wav",
  "original_text": "...", "translated_text": "..."
}
```

---

## 🔐 Ship securely (Phase 6 · public Colab sessions)

Compile the workflow into an unreadable binary and auto-scrub the plaintext:

```bash
pip install cython setuptools          # gcc is pre-installed on Colab
python build_secure.py                 # → pipeline.*.so + pipeline.py deleted
```

`app.py` keeps working unchanged — the compiled binary shadows the source
module. (Use `--keep-source` in development.)

## 🖥 Desktop bridge tunnels (Phase 7)

1. Paste your Colab `*.gradio.live` URL into **`tunnel.txt`**
2. Double-click **`Launcher.bat`** (Windows) or **`Launcher.command`** (macOS —
   right-click ▸ Open the first time)

A hidden background routine polls the tunnel asynchronously and drops the
default browser straight into the hosted UI — no console windows, no clutter.

## 🗺 Roadmap

- ✅ **Phase 1** — iPhone-15 glassmorphism UI (3 tabs, SVG-icon buttons)
- ✅ **Phase 2** — pipeline core, `clear_gpu_cache()`, steps 1–2
- ✅ **Phase 3** — diarization + emotion analysis + SRT assembly (steps 3–5)
- ✅ **Phase 4** — speaker splitting (TTS-safe) + CosyVoice 3.0 zero-shot TTS
  with silence padding & 1.15× overlap protection (steps 6–7)
- ✅ **Phase 5** — master mixdown, lookahead ducking, video remux (step 8)
- ✅ **Phase 6** — Cython obfuscation builder (`build_secure.py`)
- ✅ **Phase 7** — Windows/macOS desktop bridge tunnels
- 🔜 **Phase 8** — ready-made Colab `.ipynb` one-click launcher

## ⚠️ Notes

- SenseVoice tags `neutral` ≈ "calm" in the human-readable emotion log.
- Demucs writes 4 stems internally but only `vocals.wav` + `music.wav` are kept.
- On long videos, expect Step 2 (Demucs) to be the slowest GPU stage.

## Cloud persistence & storage

| Item | Size | Where |
|---|---|---|
| Model caches (one-time) | ~5 GB | cloud (if persistence ON) / VM (OFF) |
| Per 5-min audio job | ~0.3 GB | **local VM only - never uploaded** |
| Uploaded media | media size | local VM only |

**Three modes** (choose in Cell 0 - default is **OFF**):

- **None (default)** - everything runs on the ephemeral Colab VM. Full functionality; after a disconnect models re-download (~15 min) and pipeline steps re-run.
- **Google Drive (drive)** - model caches live in MyDrive/AutoDub_Studio/model_cache; reconnects restore instantly. Colab's own permission popup handles auth.
- **HuggingFace Hub (hf)** - model caches sync to a **private** dataset repo YourName/autodub-model-cache using your HF token (paste in Cell 0). Pulled automatically at session start, pushed after setup.




### Model persistence (UI-driven)

- A **fast Cache Check / Restore cell** runs just before the CosyVoice cell: paste
  an HF token at the top to restore models from your private HF dataset repo,
  otherwise it tries Google Drive, otherwise it downloads normally.
- A **Model persistence panel** lives in **Tab 1 > Advanced**: enable it, pick
  **Hugging Face Hub** or **Google Drive**, paste your HF token, and click
  **Start upload**. Uploads run with a live progress bar while you keep dubbing.
- Cloud storage holds **models only** (auto-cleanup) - job artifacts never leave
  the VM.

**Auto-cleanup policy:** cloud storage holds **only model caches** - job artifacts
(audio/outputs) are never uploaded and stay on the VM. The **Clear cloud storage**
button (Tab 1 > Advanced) wipes cached models (~5 GB re-downloads next session).

### Using the same token twice

The cache repo is created **private**, so **restore must authenticate too**.

- Use **one WRITE token everywhere** - a `write` token can read *and* write.
  A `read` token is enough to *download* but will break **Start upload**.
- Fine-grained tokens: grant **Read + Write** on that dataset repo (or "all
  repos"), otherwise the upload 401s.
- Paste the **same** token in Cell 4B and in the Tab 1 panel, or simply leave
  Cell 4B blank once persistence is enabled: the repo id is remembered in
  `/content/autodub_persist.json`.

> ⚠️ **If restore reports "empty storage" while your dashboard shows GBs, that
> is an auth failure, not an empty repo.** An unauthenticated download of a
> private repo returns **401**, which is easily mistaken for "nothing stored".
> The restore cell now says which it is:
>
> | Message | Meaning | Fix |
> |---|---|---|
> | `TOKEN REJECTED` | the token itself is invalid | regenerate at huggingface.co/settings/tokens |
> | `token has NO ACCESS (401)` | valid token, wrong repo or no grant | check the repo id / token scopes |
> | `accessible but EMPTY (0 files)` | genuine empty repo | upload from Tab 1 first |
> | `no such private dataset` | repo id mismatch | set `HF_REPO_RESTORE` in Cell 4B |
