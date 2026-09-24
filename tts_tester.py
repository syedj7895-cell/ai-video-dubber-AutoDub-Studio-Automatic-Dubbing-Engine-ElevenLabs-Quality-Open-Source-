"""
tts_tester.py — standalone multi-backend TTS registry for evaluation.

Used by TTS_Tester.ipynb to compare TTS engines (Hindi/multilingual) in a
lightweight Gradio UI. Each backend:
  * declares its own metadata  (info)
  * declares its own settings  (settings_schema)
  * lazily loads its model     (load) — called only when its tab is opened
  * synthesizes                (synthesize) -> (float32 mono np.ndarray, sr)

Everything is FAIL-SOFT: load/synthesize raise RuntimeError with a readable
message that the UI prints to the tab's console. No backend is import-heavy
until its load() runs.

Design goals: no API keys, no payment, models-only, Colab-friendly.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Log = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


def _require(mod: str, pip_hint: str = "") -> object:
    """Import a module or raise a helpful RuntimeError."""
    try:
        return importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        hint = pip_hint or f"pip install {mod.split('.')[0]}"
        raise RuntimeError(f"'{mod}' is not installed — run: {hint}") from e


def _pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                   capture_output=True, text=True)


# ─────────────────────────────────────────────────────────────────────
#  Metadata + base class
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TTSInfo:
    id: str
    name: str
    creator: str
    repo: str
    size: str
    license: str
    languages: str
    clone: bool
    notes: str = ""


class TTSBackend:
    """Base class — subclasses override info + load + synthesize."""
    info: TTSInfo = TTSInfo("base", "Base", "", "", "", "", "", False)

    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        self._sr = 22050

    # UI uses this to render per-model settings controls
    def settings_schema(self) -> List[dict]:
        return []

    # voices/languages the UI can populate dropdowns with
    def voices(self) -> List[str]:
        return []

    def languages(self) -> List[str]:
        return ["hi", "en"]

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, log: Log = _noop) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None
        self._loaded = False

    def synthesize(self, text: str, *, voice: Optional[str] = None,
                   language: Optional[str] = None,
                   ref_audio: Optional[str] = None,
                   settings: Optional[dict] = None,
                   log: Log = _noop) -> Tuple[np.ndarray, int]:
        raise NotImplementedError

    # helper: write floats to a temp wav and return the path
    @staticmethod
    def _wav(audio: np.ndarray, sr: int) -> str:
        import soundfile as sf
        p = Path(tempfile.mkstemp(suffix=".wav")[1])
        sf.write(str(p), np.asarray(audio, dtype=np.float32), sr)
        return str(p)


# ─────────────────────────────────────────────────────────────────────
#  1 · Edge-TTS  (network, free, no key)
# ─────────────────────────────────────────────────────────────────────

class EdgeTTS(TTSBackend):
    info = TTSInfo(
        "edge", "Edge-TTS", "Microsoft", "https://github.com/rany2/edge-tts",
        "0 MB (network)", "MIT", "Multilingual (100+) incl. Hindi", False,
        "Free Microsoft neural voices via the public Edge read-aloud endpoint. "
        "No key. Great quality, many Hindi voices.")
    _VOICES = ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural",
               "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]

    def voices(self) -> List[str]:
        return self._VOICES

    def settings_schema(self) -> List[dict]:
        return [
            {"name": "rate", "label": "Rate (%)", "type": "slider",
             "min": -50, "max": 50, "value": 0},
            {"name": "volume", "label": "Volume (%)", "type": "slider",
             "min": -50, "max": 50, "value": 0},
        ]

    def load(self, log: Log = _noop) -> None:
        _require("edge_tts", "pip install edge-tts")
        self._loaded = True
        log("Edge-TTS ready (network, no model download).")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        edge_tts = _require("edge_tts")
        import asyncio
        settings = settings or {}
        v = voice or self._VOICES[0]
        rate = f"{int(settings.get('rate', 0)):+d}%"
        vol = f"{int(settings.get('volume', 0)):+d}%"

        async def _run() -> bytes:
            comm = edge_tts.Communicate(text, v, rate=rate, volume=vol)
            buf = b""
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf += chunk["data"]
            return buf

        data = asyncio.run(_run())
        p = Path(tempfile.mkstemp(suffix=".mp3")[1])
        p.write_bytes(data)
        import librosa
        y, sr = librosa.load(str(p), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


# ─────────────────────────────────────────────────────────────────────
#  2 · gTTS  (network, free, no key)
# ─────────────────────────────────────────────────────────────────────

class GTTS(TTSBackend):
    info = TTSInfo(
        "gtts", "gTTS", "Google", "https://github.com/pndurette/gTTS",
        "0 MB (network)", "MIT", "Multilingual incl. Hindi", False,
        "Free Google Translate TTS endpoint. Zero install, very reliable, "
        "single fixed voice per language.")

    def languages(self) -> List[str]:
        return ["hi", "en", "ur", "bn", "ta", "te"]

    def load(self, log: Log = _noop) -> None:
        _require("gtts", "pip install gTTS")
        self._loaded = True
        log("gTTS ready (network).")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        gtts = _require("gtts")
        lang = (language or "hi").split("-")[0]
        p = Path(tempfile.mkstemp(suffix=".mp3")[1])
        gtts.gTTS(text=text, lang=lang, slow=False).save(str(p))
        import librosa
        y, sr = librosa.load(str(p), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


# ─────────────────────────────────────────────────────────────────────
#  3 · Meta MMS-TTS (local, tiny)
# ─────────────────────────────────────────────────────────────────────

class MMSTTS(TTSBackend):
    info = TTSInfo(
        "mms", "Meta MMS-TTS", "Meta AI", "https://github.com/facebookresearch/fairseq/tree/main/examples/mms",
        "~150 MB", "CC-BY-NC 4.0", "Hindi (native) + 1100 langs", False,
        "Tiny VITS model, fast on CPU, always-correct Hindi. One fixed voice.")
    _REPO = {"hi": "facebook/mms-tts-hin", "en": "facebook/mms-tts-eng"}

    def languages(self) -> List[str]:
        return ["hi", "en"]

    def load(self, log: Log = _noop) -> None:
        _require("transformers", "pip install transformers torch")
        self._loaded = True
        log("MMS-TTS ready (loads per-language on demand).")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        import torch
        from transformers import VitsModel, AutoTokenizer
        lang = (language or "hi").split("-")[0]
        repo = self._REPO.get(lang, self._REPO["hi"])
        log(f"MMS: loading {repo} …")
        tok = AutoTokenizer.from_pretrained(repo)
        model = VitsModel.from_pretrained(repo)
        model.eval()
        with torch.no_grad():
            inputs = tok(text, return_tensors="pt")
            out = model(**inputs).waveform
        y = out.squeeze().cpu().numpy().astype(np.float32)
        return y, int(model.config.sampling_rate)


# ─────────────────────────────────────────────────────────────────────
#  4 · Piper TTS (local, ONNX, tiny)
# ─────────────────────────────────────────────────────────────────────

class PiperTTS(TTSBackend):
    info = TTSInfo(
        "piper", "Piper TTS (Hindi)", "Rhasspy", "https://github.com/rhasspy/piper",
        "~60 MB / voice", "MIT", "Hindi + many", False,
        "Very fast ONNX neural TTS. Voices are downloaded per language.")

    def voices(self) -> List[str]:
        return ["hi_IN-pratham-medium", "hi_IN-priyamvada-medium"]

    def languages(self) -> List[str]:
        return ["hi", "en"]

    def load(self, log: Log = _noop) -> None:
        _require("piper", "pip install piper-tts")
        self._loaded = True
        log("Piper ready.")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        # piper exposes a python API (piper.voice) in recent builds
        try:
            from piper.voice import PiperVoice
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Piper python API unavailable; use the piper CLI or "
                "pip install piper-tts==1.2.0") from e
        vname = voice or self.voices()[0]
        # resolve a model file (users may pre-download into /content/piper)
        model_path = self._find_model(vname)
        v = PiperVoice.load(model_path)
        import wave
        p = Path(tempfile.mkstemp(suffix=".wav")[1])
        with wave.open(str(p), "wb") as wf:
            v.synthesize(text, wf)
        import soundfile as sf
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        return np.asarray(y, dtype=np.float32), int(sr)

    @staticmethod
    def _find_model(name: str) -> str:
        for base in ("/content/piper", str(Path.home() / "piper"),
                     str(Path.cwd() / "piper_voices")):
            b = Path(base)
            if b.exists():
                for f in b.rglob(f"{name}*.onnx"):
                    return str(f)
        raise RuntimeError(
            f"Piper voice '{name}' not found. Download the .onnx + .json from "
            "huggingface.co/rhasspy/piper-voices into /content/piper")


# ─────────────────────────────────────────────────────────────────────
#  5 · Kokoro TTS (local, small)
# ─────────────────────────────────────────────────────────────────────

class KokoroTTS(TTSBackend):
    info = TTSInfo(
        "kokoro", "Kokoro-82M", "hexgrad", "https://huggingface.co/hexgrad/Kokoro-82M",
        "~330 MB", "Apache-2.0", "English + (zh/ja); Hindi limited", False,
        "Very small, high-quality English TTS. Hindi support is limited.")

    def languages(self) -> List[str]:
        return ["en", "hi"]

    def load(self, log: Log = _noop) -> None:
        _require("kokoro", "pip install kokoro soundfile")
        self._loaded = True
        log("Kokoro ready.")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code=(language or "en")[:1])
        chunks = []
        sr = 24000
        for _gs, _ps, audio in pipeline(text, voice=voice or "af_heart"):
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")
        return np.concatenate(chunks), sr


# ─────────────────────────────────────────────────────────────────────
#  6 · AI4Bharat IndicF5 (local, Hindi-native, clone)
# ─────────────────────────────────────────────────────────────────────

class IndicF5TTS(TTSBackend):
    info = TTSInfo(
        "indicf5", "AI4Bharat IndicF5", "AI4Bharat", "https://huggingface.co/ai4bharat/IndicF5",
        "~1.5 GB", "MIT", "Indic (Hindi etc.)", True,
        "F5-TTS fine-tuned for Indian languages. Needs a reference clip + its "
        "transcript for cloning (zero-shot).")

    def load(self, log: Log = _noop) -> None:
        _require("torch", "pip install torch")
        _require("transformers", "pip install transformers")
        self._loaded = True
        log("IndicF5 loader ready (weights load at synthesize).")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        if not ref_audio:
            raise RuntimeError("IndicF5 needs a reference audio clip (upload one).")
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained("ai4bharat/IndicF5",
                                              trust_remote_code=True)
            ref_text = (settings or {}).get("ref_text", "")
            audio = model(text, ref_audio_path=ref_audio, ref_text=ref_text)
            import soundfile as sf
            p = Path(tempfile.mkstemp(suffix=".wav")[1])
            sf.write(str(p), audio, 24000)
            y, sr = sf.read(str(p), dtype="float32")
            return np.asarray(y, dtype=np.float32), int(sr)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"IndicF5 synthesis failed: {str(e)[:200]}") from e

    def settings_schema(self) -> List[dict]:
        return [{"name": "ref_text", "label": "Reference transcript",
                 "type": "textbox", "value": ""}]


# ─────────────────────────────────────────────────────────────────────
#  7 · F5-TTS (local, clone)
# ─────────────────────────────────────────────────────────────────────

class F5TTS(TTSBackend):
    info = TTSInfo(
        "f5", "F5-TTS", "SWivid", "https://github.com/SWivid/F5-TTS",
        "~1.3 GB", "MIT", "Multilingual incl. Hindi", True,
        "Flow-matching TTS with strong zero-shot cloning. Needs a reference "
        "clip (+ optional transcript).")

    def load(self, log: Log = _noop) -> None:
        try:
            importlib.import_module("f5_tts")
        except Exception:  # noqa: BLE001
            log("Installing f5-tts …")
            _pip("f5-tts")
        self._loaded = True
        log("F5-TTS ready.")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        if not ref_audio:
            raise RuntimeError("F5-TTS needs a reference audio clip (upload one).")
        try:
            from f5_tts.api import F5TTS as _F5
            f5 = _F5()
            ref_text = (settings or {}).get("ref_text", "")
            wav, sr, _ = f5.infer(ref_file=ref_audio, ref_text=ref_text,
                                  gen_text=text)
            return np.asarray(wav, dtype=np.float32), int(sr)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"F5-TTS synthesis failed: {str(e)[:200]}") from e

    def settings_schema(self) -> List[dict]:
        return [{"name": "ref_text", "label": "Reference transcript",
                 "type": "textbox", "value": ""}]


# ─────────────────────────────────────────────────────────────────────
#  8 · Chatterbox (local, clone, multilingual)
# ─────────────────────────────────────────────────────────────────────

class ChatterboxTTS(TTSBackend):
    info = TTSInfo(
        "chatterbox", "Chatterbox TTS", "Resemble AI", "https://github.com/resemble-ai/chatterbox",
        "~2 GB", "MIT", "Multilingual incl. Hindi", True,
        "Resemble AI's open multilingual TTS with voice cloning + emotion "
        "control. Reference clip optional (uses a default voice otherwise).")

    def load(self, log: Log = _noop) -> None:
        try:
            importlib.import_module("chatterbox")
        except Exception:  # noqa: BLE001
            log("Installing chatterbox-tts …")
            _pip("chatterbox-tts")
        self._loaded = True
        log("Chatterbox ready.")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        try:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device="cuda"
                                                              if torch.cuda.is_available() else "cpu")
            kwargs = {"language_id": (language or "hi")}
            if ref_audio:
                kwargs["audio_prompt_path"] = ref_audio
            wav = model.generate(text, **kwargs)
            y = wav.squeeze().cpu().numpy().astype(np.float32)
            return y, int(model.sr)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Chatterbox failed: {str(e)[:200]}") from e

    def settings_schema(self) -> List[dict]:
        return [{"name": "exaggeration", "label": "Exaggeration",
                 "type": "slider", "min": 0, "max": 2, "value": 0},
                {"name": "cfg", "label": "CFG weight (stability)",
                 "type": "slider", "min": 0, "max": 2, "value": 0}]


# ─────────────────────────────────────────────────────────────────────
#  9 · Coqui XTTS v2 (local, clone)
# ─────────────────────────────────────────────────────────────────────

class XTTS(TTSBackend):
    info = TTSInfo(
        "xtts", "Coqui XTTS v2", "Coqui", "https://huggingface.co/coqui/XTTS-v2",
        "~1.8 GB", "Coqui CPML", "Multilingual incl. Hindi", True,
        "Classic multilingual cloning TTS. Needs a reference clip.")

    def languages(self) -> List[str]:
        return ["hi", "en", "es", "fr", "de", "zh-cn", "ja", "ko"]

    def load(self, log: Log = _noop) -> None:
        try:
            importlib.import_module("TTS")
        except Exception:  # noqa: BLE001
            log("Installing Coqui TTS …")
            _pip("TTS")
        self._loaded = True
        log("XTTS ready.")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        if not ref_audio:
            raise RuntimeError("XTTS needs a reference audio clip (upload one).")
        try:
            import torch
            from TTS.api import TTS as _TTS
            tts = _TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
                "cuda" if torch.cuda.is_available() else "cpu")
            out = tts.tts(text=text, speaker_wav=ref_audio,
                          language=(language or "hi"))
            y = np.asarray(out, dtype=np.float32)
            return y, 24000
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"XTTS failed: {str(e)[:200]}") from e


# ─────────────────────────────────────────────────────────────────────
#  10/11 · CosyVoice 2 & 3 (reuse pipeline loader)
# ─────────────────────────────────────────────────────────────────────

class _CosyVoiceBase(TTSBackend):
    _pairs: tuple = ()
    _label = "CosyVoice"

    def load(self, log: Log = _noop) -> None:
        # reuse the robust loader already proven in pipeline.py
        try:
            import pipeline as _pl
            self._pl = _pl
            self._pl._load_cosyvoice(log)
            self._loaded = True
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{self._label} loader unavailable: {str(e)[:200]}") from e

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        raise RuntimeError(f"{self._label}: use the main app for full synthesis; "
                           "standalone synth is disabled in the tester.")


class CosyVoice2(_CosyVoiceBase):
    info = TTSInfo("cosyvoice2", "CosyVoice 2", "FunAudioLLM",
                   "https://github.com/FunAudioLLM/CosyVoice",
                   "~3.5 GB", "Apache-2.0", "Chinese + English", True,
                   "Alibaba zero-shot TTS. No native Hindi (use CV3).")
    _label = "CosyVoice 2"


class CosyVoice3(_CosyVoiceBase):
    info = TTSInfo("cosyvoice3", "CosyVoice 3", "FunAudioLLM",
                   "https://github.com/FunAudioLLM/CosyVoice",
                   "~3.5 GB", "Apache-2.0", "Multilingual incl. Hindi", True,
                   "Alibaba's latest — native Hindi + cloning. Preferred engine.")
    _label = "CosyVoice 3"


# ─────────────────────────────────────────────────────────────────────
#  12/13 · Microsoft VibeVoice
# ─────────────────────────────────────────────────────────────────────

class VibeVoice(TTSBackend):
    info = TTSInfo(
        "vibevoice", "VibeVoice 1.5B", "Microsoft", "https://github.com/microsoft/VibeVoice",
        "~3 GB", "MIT", "Multilingual", True,
        "Microsoft's long-form, multi-speaker conversational TTS.")

    def load(self, log: Log = _noop) -> None:
        log("Installing VibeVoice (git) …")
        _pip("git+https://github.com/microsoft/VibeVoice.git")
        self._loaded = True

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        raise RuntimeError("VibeVoice standalone synth wrapper is experimental — "
                           "see repo inference example.")


class VibeVoiceHindi(VibeVoice):
    info = TTSInfo(
        "vibevoice_hi", "VibeVoice-Hindi-7B", "tarun7r (community)",
        "https://huggingface.co/tarun7r/vibevoice-hindi-7b",
        "~7 GB", "Apache-2.0", "Hindi", True,
        "Community Hindi fine-tune of VibeVoice (7B). Community weights.")

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        raise RuntimeError("VibeVoice-Hindi standalone synth wrapper is "
                           "experimental — see model card.")


# ─────────────────────────────────────────────────────────────────────
#  14 · Veena (Maya Research)
# ─────────────────────────────────────────────────────────────────────

class Veena(TTSBackend):
    info = TTSInfo(
        "veena", "Veena", "Maya Research", "https://huggingface.co/maya-research/Veena",
        "~? GB", "see model card", "Hindi + Indic", True,
        "Maya Research Indic TTS. Check the model card for the exact loader.")

    def load(self, log: Log = _noop) -> None:
        _require("transformers", "pip install transformers")
        log("Veena: load per the model card (custom inference code).")
        self._loaded = True

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        raise RuntimeError("Veena uses custom inference — see "
                           "huggingface.co/maya-research/Veena.")


# ─────────────────────────────────────────────────────────────────────
#  15 · VEXYL-TTS
# ─────────────────────────────────────────────────────────────────────

class VexylTTS(TTSBackend):
    info = TTSInfo(
        "vexyl", "VEXYL-TTS", "VEXYL AI", "https://github.com/vexyl-ai/vexyl-tts",
        "~? GB", "see repo", "Hindi + multilingual", True,
        "VEXYL-TTS (community). Install from git; see repo for usage.")

    def load(self, log: Log = _noop) -> None:
        log("Installing VEXYL-TTS (git) …")
        _pip("git+https://github.com/vexyl-ai/vexyl-tts.git")
        self._loaded = True

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        raise RuntimeError("VEXYL-TTS standalone wrapper is experimental — "
                           "see the repo README.")


# ─────────────────────────────────────────────────────────────────────
#  16 · Qwen3-TTS
# ─────────────────────────────────────────────────────────────────────

class Qwen3TTS(TTSBackend):
    info = TTSInfo(
        "qwen3tts", "Qwen3-TTS", "Alibaba Qwen", "https://github.com/QwenLM/Qwen3-TTS",
        "~? GB", "Apache-2.0", "Multilingual incl. Hindi", True,
        "Alibaba Qwen3 TTS. Install from git; exact HF weights per repo README.")

    def load(self, log: Log = _noop) -> None:
        _require("torch", "pip install torch")
        log("Qwen3-TTS: install per github.com/QwenLM/Qwen3-TTS.")
        self._loaded = True

    def synthesize(self, text, *, voice=None, language=None, ref_audio=None,
                   settings=None, log=_noop):
        raise RuntimeError("Qwen3-TTS uses custom inference — see the repo.")


# ─────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────

BACKENDS: Dict[str, type] = {
    c.info.id: c for c in (
        EdgeTTS, GTTS, MMSTTS, PiperTTS, KokoroTTS, IndicF5TTS, F5TTS,
        ChatterboxTTS, XTTS, CosyVoice2, CosyVoice3, VibeVoice,
        VibeVoiceHindi, Veena, VexylTTS, Qwen3TTS,
    )
}

_instances: Dict[str, TTSBackend] = {}


def get_backend(bid: str) -> TTSBackend:
    """Return a singleton backend instance (created on first use)."""
    if bid not in _instances:
        _instances[bid] = BACKENDS[bid]()
    return _instances[bid]


def list_backends() -> List[TTSInfo]:
    return [get_backend(bid).info for bid in BACKENDS]