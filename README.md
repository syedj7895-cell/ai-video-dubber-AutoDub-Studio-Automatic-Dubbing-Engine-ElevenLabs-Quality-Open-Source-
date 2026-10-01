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
| 7 | Multilingual emotional TTS | **CosyVoice 2.0 / 3.0 · Chatterbox · Fish S2-Pro · Kokoro-82M · Edge-TTS** (6-way selector) | ✅ Phase 4 |
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

## 🎙 Pluggable TTS engines (6-way selector · Tab 1)

Pick the synthesizer at the very top of **Tab 1**. Every engine shares the same
per-speaker clone prompts and the same silent-master timeline, so swapping an
engine only changes *who* speaks — never *when*.

| Engine | Params | VRAM | Zero-shot clone | Fallback |
|---|---|---|---|---|
| **CosyVoice 2.0** *(default)* | 0.5 B | ~2 GB | ✅ | Edge-TTS |
| **CosyVoice 3.0** | 0.5 B | ~2 GB | ✅ | — terminal — |
| **Chatterbox** (Resemble AI) | 0.5 B | ~2 GB | ✅ | — terminal — |
| **Fish Audio S2-Pro** | — | 4–12 GB | ✅ | — terminal — |
| **Kokoro-82M** *(lightest)* | 82 M | ~1 GB | ❌ | — terminal — |
| **Edge-TTS** *(cloud)* | cloud | 0 | ❌ | — terminal — |

**One automatic fallback.** **CosyVoice 2.0** degrades to **Edge-TTS** when it
fails to load (missing package, OOM, absent weights) *or* after **3 consecutive
line failures** mid-render. Every other engine is **terminal**: it must either
deliver the real dubbed mix or abort with a detailed report. Nothing is ever
silently substituted, because a quiet voice swap hides the bug you actually
need to fix. Changing the engine in Tab 1 invalidates the Step 7 cache and
re-renders automatically.

**Two builds, two pipelines.** The table above belongs to `pipeline.py`.
`pipeline2.py` is AutoDub Studio 2.0's **dedicated copy** of it — byte-for-byte
identical apart from its header and two identity constants, so
`diff pipeline.py pipeline2.py` shows exactly what makes 2.0 2.0 and nothing
else:

```python
PIPELINE_VARIANT = "2.0"             # pipeline.py has "1.0"
ENGINE_ALLOWLIST = ("chatterbox",)   # pipeline.py has ()
```

`app.py` binds whichever module `AUTODUB_PIPELINE` names, so neither pipeline
file references the other and the two can evolve independently. A non-identifier
value is rejected at import rather than imported:

| Launcher | Pipeline module | Engines published |
|---|---|---|
| `Colab_Runner.ipynb` | `pipeline.py` | all six |
| `AutoDub Studio 2.0.ipynb` | `pipeline2.py` | **Chatterbox only** — baked into `ENGINE_ALLOWLIST` |

**Narrowing further.** `AUTODUB_TTS_ENGINES` (comma-separated ids) still narrows
whichever build you launched, and takes precedence over `ENGINE_ALLOWLIST` when
set. Both mechanisms filter the registry itself, so the Tab 1 radio, the Tab 2
audition dropdown, the default engine and the fallback rules all follow from one
rule instead of each re-implementing it.

Requesting an engine that was filtered out degrades to the active one rather
than raising, and a single-engine build's Tab 1 helper text and licence notice
are *computed* from what is actually published — so neither build ever advertises
an engine it does not ship.

**Terminal failures are loud.** When a terminal engine dies, Step 7 raises with:

- one line per cause, in chronological order (`[import]`, `[from_local]`,
  `[row 12 · SPEAKER_00]`, …)
- a fingerprint of what is actually installed (`python`, `torch`, `numpy`,
  `gradio`, `librosa`, `transformers`, CUDA + free VRAM)
- a copy-pasteable fix, where one is known
- the **full traceback** in `outputs/engine_error.log`

Step 7 also refuses to write all-zero master tracks: if no row produced audio,
the render aborts *before* Step 8, so you can never end up with a "successful"
final mix that contains no speech at all.

**Distinct voice per speaker.** Engines that cannot clone (Edge-TTS, Kokoro)
have voices cast per speaker instead — deterministically, matched to the
**auto-detected** gender (see below). Two pools are used:

- **Edge** — Microsoft ships only **two** true Hindi neural voices, so the pool
  is tiered by phonetic fidelity (`hi-IN` → `mr-IN`/`ne-NP`, written in
  Devanagari and read Hindi natively → other Indic locales, flagged in the
  console because they *will* mispronounce Devanagari). That gives **15 female
  + 15 male** verified voices, of which 3 per gender are Devanagari-faithful.
- **Kokoro** — the **4** shipped Hindi voices (`hf_alpha`, `hf_beta`, `hm_omega`,
  `hm_psi`). A cast larger than two per gender necessarily shares voices, and
  that is reported rather than hidden.

**Automatic gender detection (no more "everyone sounds female").** Pyannote
reports *how many* voices there are, never *which is which* — it emits no gender
field at all. Step 3 therefore profiles each speaker from their own mined clone
prompt: median **F0** via pYIN, with the well-established 150 Hz / 175 Hz
male-female split. A speaker in the ambiguous 150–175 Hz band needs a decisive
long-term spectral centroid, otherwise the gender is left **undetermined** and
their voice **rotates** through a gender-interleaved pool instead of defaulting
to female. Your Tab 2 override always wins over the estimate.

> Why pYIN and not plain YIN: plain YIN has no way to say "there is no pitch
> here". Measured on synthetic input it labelled **pure silence** as *female,
> 400 Hz* and **white noise** as *male, 73 Hz* — both at full confidence. pYIN's
> voiced/unvoiced flag rejects every non-voice input, and both regressions are
> now locked down in `tools/selftest.py`.

**Kokoro voice blending.** Kokoro has no per-voice dictionary —
`KPipeline.load_voice(name)` is the accessor and it returns a tensor, so a blend
is a weighted average of two voice tensors (the mechanism Kokoro's own docs
showcase via `voice=<tensor>`). The weight is clamped to **0.30–0.70**: outside
that range the mix is just a worse copy of one voice. A partner of the *other*
gender is refused, so a blend can never de-gender a speaker. Set it with
`set_kokoro_blend(partner, weight)`.

**Phase D · auto-blend above capacity.** Kokoro ships only **4** Hindi voices, so
a balanced cast of four maps one-to-one and everyone is already distinct. Past
that a voice is *recycled* and two speakers become audibly identical — the same
failure the Edge pool has, just smaller. Step 7 therefore gives every **shared**
speaker a weighted mix of the two same-gender voices at a canonical weight no
other speaker uses (0.70 → 0.30 → 0.50 …, the extremes first, so consecutive
speakers sit as far apart as the clamp allows). The canonical weight is computed
on the gender's *first* pool voice and then translated onto whatever voice the
allocator actually assigned — otherwise two recycled speakers could request the
same ladder value and end up with the identical tensor.

Precedence is explicit: a Tab 2 **blend partner** you set by hand is an
instruction, so it wins and auto-blend switches off — with a console warning if
voices are still being shared, because one identical blend cannot separate two
speakers. Clear the partner to hand control back to Phase D.

**Edge prosody.** Edge cannot clone, so `pitch` (Hz), `rate` and `volume`
(percent) are its only expressive controls — persisted via
`set_engine_prosody()` and applied to every line. Pitch support is probed at
runtime, since it only exists in edge-tts ≥ 6.1, and is skipped rather than
crashing on older builds.

**Cross-lingual guard.** Clone prompts are recorded in the *source* language but
we dub *into Hindi*, so Chatterbox is driven with `cfg_weight=0.0` to stop the
reference clip's accent bleeding into the Hindi output.

**Emotion mapping.** SenseVoice's 7 emotions drive each engine natively:
CosyVoice takes a separate `instruct` string, Chatterbox maps to its
`exaggeration` scalar (0–2), Fish embeds `[tag]` inline, and Edge/Kokoro have no
emotion control.

**Chatterbox's installed API — worth knowing before debugging it.** The released
`chatterbox-tts` 0.1.7 wheel exposes
`ChatterboxMultilingualTTS.from_local(ckpt_dir, device)` and
`from_pretrained(device)`. **Neither accepts `repo_id` or `t3_model`**, because
`REPO_ID` *and* the T3 checkpoint filename are hardcoded inside the package. The
loader is therefore invoked through `_call_with_supported_kwargs()`, which
passes only the keywords the installed build actually declares — that keeps
0.1.7 working while picking up the unreleased master's `t3_model` argument
automatically, instead of dying on an unknown keyword. The checkpoint snapshot
requests exactly the six names `from_local()` opens, and Hindi is already one of
its 23 languages, so no extra weights are needed.

> ⚠️ **Licence & watermark notice** — Chatterbox stamps a Resemble **PerTh**
> neural watermark into every clip it generates. Fish Audio S2-Pro output falls
> under the **Fish Audio Research License (non-commercial)**. Edge-TTS audio is
> synthesised remotely by Microsoft. Kokoro-82M weights are Apache-2.0.

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
├── pipeline2.py           # AutoDub Studio 2.0's DEDICATED copy (Chatterbox only)
├── build_secure.py        # Phase 6 · Cython obfuscation builder (compile + scrub)
├── Colab_Runner.ipynb     # Phase 8 · one-click Colab launcher (pipeline.py · 6 engines)
├── AutoDub Studio 2.0.ipynb  # launcher → pipeline2.py (AUTODUB_PIPELINE=pipeline2)
├── requirements.txt
├── assets/icons/          # premium gradient SVG button icons
├── tools/
│   ├── selftest.py        # 177 pure-Python checks · python tools/selftest.py
│   └── bridge.py          # Phase 7 · desktop tunnel poller
├── uploads/               # runtime · user uploads
└── outputs/               # runtime artifacts
    ├── step1_audio.wav          # extracted master audio
    ├── vocals.wav / music.wav   # Demucs stems
    ├── diarization_map.json     # Speaker1/2/… per SRT cue
    ├── speakerX_clone_prompt.wav# top-3 cleanest 5–10 s refs per speaker
    ├── emotion_log.txt          # "[00:50.000] Speaker1 [calm]"
    ├── emotion_grid.json        # machine emotion grid
    ├── final_script.json        # consolidated dubbing script ★
    ├── voice_preview.wav        # Tab 2 ▸ 🔊 Preview audition clip
    ├── track_speaker*.wav       # one silent-grid track per speaker (step 7)
    ├── tts_report.json          # step 7 verdict (engine, rows, cache key)
    ├── engine_error.log         # full traceback for any terminal-engine abort
    ├── final_mix.wav            # master mix (step 8)
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

**The scrub is guarded, not blind.** `pipeline.py` is deleted *only after* a
child process has imported the compiled binary **with the source hidden** and
confirmed the public API (13 names, incl. `step7_synthesize`, `resolve_engine`,
`auto_kokoro_blends`) is present. If that import fails — wrong architecture,
missing symbol, broken build — the source is kept and the run exits non-zero.
The child restores the source in a `finally` and the parent restores it again,
so even a killed verifier leaves a working tree. `--no-verify` can never scrub:
you cannot opt out of the one check that makes deletion safe.

| Flag | Effect |
|---|---|
| *(none)* | compile → verify → scrub |
| `--keep-source` | compile → verify, keep `pipeline.py` (development) |
| `--dry-run` | translate to C only — no compiler, nothing written or deleted |
| `--force` | rebuild even if an extension already exists |
| `--no-verify` | skip the import check; **scrubbing is refused** |
| `--target NAME` | compile a different module (default `pipeline`; use `--target pipeline2` for the 2.0 build) |

Verified here: Cython translates `pipeline.py` cleanly (7.4 MB of C, no
warnings). The actual C compile needs a compiler — Colab ships `gcc`; a bare
Windows box does not, and `build_secure.py` detects that and exits **before**
touching anything rather than failing halfway.

> After obfuscation, `tools/selftest.py`'s two *source-scanning* checks are
> reported as **skipped** instead of passing vacuously: compiled code has no
> retrievable source for `inspect.getsource`.


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
- ✅ **Phase 8** — ready-made Colab `.ipynb` one-click launcher

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
