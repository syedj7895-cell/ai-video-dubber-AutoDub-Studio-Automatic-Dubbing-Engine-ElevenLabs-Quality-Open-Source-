#!/usr/bin/env python3
"""
fish_s2_probe.py — Gate 1 feasibility probe for self-hosted Fish Audio S2-Pro.

Run this on Google Colab (T4) BEFORE wiring S2 into pipeline.py. It answers the
four questions that decide whether self-hosted S2-Pro is usable on a 16 GB T4:

  1. ENV     — GPU, compute capability, VRAM, CUDA/driver, RAM, disk
  2. BNB     — can bitsandbytes NF4 4-bit run on THIS GPU? (T4 = Turing sm_75)
  3. VULKAN  — is a Vulkan ICD present? (s2.cpp's only GPU backend)
  4. AB      — real VRAM peak + RTF on a Hindi line, vs CosyVoice 2.0

Every stage is independent. Nothing here modifies the project — it only
downloads model weights into the standard HuggingFace cache.

Usage:
    python fish_s2_probe.py              # env only (instant, zero risk)
    python fish_s2_probe.py --env
    python fish_s2_probe.py --bnb        # try NF4-quantized S2-Pro
    python fish_s2_probe.py --vulkan     # Vulkan ICD + s2.cpp build check
    python fish_s2_probe.py --ab         # Hindi A/B vs CosyVoice 2.0
    python fish_s2_probe.py --all

Exit codes: 0 = probe ran, 1 = crashed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252 and choke on ═ · → — force UTF-8 so the
# probe prints correctly everywhere (Colab is UTF-8 already).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RULES = "═" * 70

# A representative Hindi dub line (~110 chars) — long enough that S2's known
# "short-utterance artefacts" do not skew the measurement.
HINDI_LINE = ("नमस्ते, आपका स्वागत है। यह एक परीक्षण वाक्य है जो हिंदी "
              "उच्चारण की गुणवत्ता और स्वराघात की जाँच करता है।")

S2_REPO = "fishaudio/s2-pro"
BNB_REPO = "groxaxo/s2-pro"          # NF4 community quant (12 GB target)


# ─────────────────────────────────────────────────────────────────────────────
#  small helpers
# ─────────────────────────────────────────────────────────────────────────────

def hr(title: str = "") -> None:
    print("\n" + RULES)
    if title:
        print(f"  {title}")
        print(RULES)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def info(msg: str) -> None:
    print(f"  ..     {msg}")


def sh(cmd: str, timeout: int = 120) -> tuple:
    """Run a shell command, return (returncode, combined output)."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "(timed out)"
    except Exception as e:                                   # pragma: no cover
        return 1, str(e)


def gpu_report() -> dict:
    """Collect GPU facts via torch (primary) and nvidia-smi (fallback)."""
    facts: dict = {}
    try:
        import torch
        facts["torch"] = torch.__version__
        facts["cuda_built"] = torch.version.cuda
        facts["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            facts["gpu"] = p.name
            facts["cc"] = f"{p.major}.{p.minor}"
            facts["vram_gb"] = round(p.total_memory / 2**30, 2)
            try:
                facts["bf16"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                facts["bf16"] = None
            try:
                free, _tot = torch.cuda.mem_get_info()
                facts["vram_free_gb"] = round(free / 2**30, 2)
            except Exception:
                pass
    except Exception as e:
        facts["torch_error"] = str(e)

    rc, out = sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
                 "--format=csv,noheader")
    if rc == 0 and out.strip():
        facts["nvidia_smi"] = out.strip().splitlines()[0]
    return facts


# ─────────────────────────────────────────────────────────────────────────────
#  stage 1 · ENV
# ─────────────────────────────────────────────────────────────────────────────

def stage_env() -> dict:
    hr("1 · ENVIRONMENT")
    facts = gpu_report()

    info(f"python          {sys.version.split()[0]}")
    info(f"platform        {sys.platform}")

    gpu = facts.get("gpu")
    if gpu:
        ok(f"gpu             {gpu} · {facts.get('vram_gb', '?')} GB · "
           f"compute {facts.get('cc', '?')} · CUDA {facts.get('cuda_built')}")
        free = facts.get("vram_free_gb")
        if free is not None:
            info(f"vram free       {free} GB of {facts.get('vram_gb')} GB")
    else:
        bad("no CUDA GPU visible — S2 self-hosting is not testable here")
        if facts.get("nvidia_smi"):
            info(f"nvidia-smi      {facts['nvidia_smi']}")

    # ── the two hard capability gates ─────────────────────────────────────
    cc = str(facts.get("cc", ""))
    ccv = 0.0
    if cc:
        major, _, minor = cc.partition(".")
        try:
            ccv = float(f"{major}.{minor}")
        except ValueError:
            ccv = 0.0

        if ccv >= 7.5:
            ok(f"bitsandbytes 4-bit needs CC>=7.5 — this GPU is {cc} "
               f"(eligible)")
        else:
            bad(f"bitsandbytes 4-bit needs CC>=7.5 — this GPU is {cc} "
                f"(NOT eligible)")

        if ccv >= 8.9:
            ok(f"FP8 quant available (CC {cc} >= 8.9)")
        else:
            warn(f"FP8 quant NOT available (needs CC>=8.9; this is {cc}) — "
                 f"drbaph/s2-pro-fp8 would be unusable")

        if ccv >= 8.0:
            ok(f"BF16 native (CC {cc} >= 8.0)")
        else:
            warn(f"no native BF16 (this is {cc}) — S2 needs its --half path")

    # ── system resources ─────────────────────────────────────────────────
    try:
        import psutil
        vm = psutil.virtual_memory()
        info(f"ram             {vm.total / 2**30:.1f} GB "
             f"({vm.available / 2**30:.1f} GB free)")
    except Exception:
        pass
    try:
        du = shutil.disk_usage("/")
        info(f"disk            {du.total / 2**30:.1f} GB · "
             f"{du.free / 2**30:.1f} GB free")
        if du.free / 2**30 < 25:
            warn("under 25 GB free — S2 weights are 9-12 GB plus cache; "
                 "free some space first")
    except Exception:
        pass
    for mod in ("bitsandbytes", "huggingface_hub", "soundfile", "librosa"):
        try:
            __import__(mod)
            ok(f"module          {mod} present")
        except ImportError:
            warn(f"module          {mod} MISSING  →  pip install {mod}")

    facts["verdict_env"] = "GPU present" if gpu else "NO GPU"
    return facts


# ─────────────────────────────────────────────────────────────────────────────
#  stage 2 · BITSANDBYTES NF4  (decides the ~12 GB self-host path)
# ─────────────────────────────────────────────────────────────────────────────

BNB_TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # tiny stand-in, same quant path


def _ensure(mod: str, pip_name: str = "") -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        pkg = pip_name or mod
        info(f"installing {pkg} ...")
        rc, out = sh(f"{sys.executable} -m pip install -q {pkg}", timeout=900)
        if rc != 0:
            bad(f"pip install {pkg} failed: {out.strip()[-200:]}")
            return False
        try:
            __import__(mod)
            return True
        except ImportError as e:
            bad(f"still cannot import {mod}: {e}")
            return False


def stage_bnb() -> dict:
    hr("2 · BITSANDBYTES NF4  (the ~12 GB S2-Pro path)")
    res: dict = {"stage": "bnb"}

    if not _ensure("bitsandbytes"):
        res["verdict"] = "bitsandbytes unavailable"
        return res
    import bitsandbytes as bnb
    ok(f"bitsandbytes    {getattr(bnb, '__version__', '?')}")

    try:
        import torch
        if not torch.cuda.is_available():
            bad("no CUDA GPU — an NF4 test needs one")
            res["verdict"] = "no GPU"
            return res
        cc = torch.cuda.get_device_capability(0)
        info(f"compute         sm_{cc[0]}{cc[1]}")
    except Exception as e:
        warn(f"could not read compute capability: {e}")
        return res

    if not _ensure("transformers"):
        res["verdict"] = "transformers unavailable"
        return res

    # ── the decisive test: quantize a small model to NF4 and actually run it
    try:
        import gc

        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)

        info(f"loading {BNB_TEST_MODEL} in NF4 4-bit ...")
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,     # T4-safe (no bf16)
            bnb_4bit_use_double_quant=True,
        )
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(BNB_TEST_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BNB_TEST_MODEL, quantization_config=cfg, device_map={"": 0})
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        ok(f"NF4 load OK in {dt:.1f}s · peak VRAM {peak:.2f} GB")
        res.update(nf4_load_ok=True, nf4_load_s=round(dt, 1),
                   nf4_peak_gb=round(peak, 2))

        tok.pad_token = tok.eos_token
        ids = tok("Say hi in one word.", return_tensors="pt").to(0)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=8, do_sample=False)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        ok(f"NF4 inference OK → {txt.strip()[:60]!r}")
        res["nf4_infer_ok"] = True

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        ok("NF4 executes on this GPU → S2-Pro NF4 (~12 GB) is PLAUSIBLE")
        res["verdict"] = "NF4 viable"
    except Exception as e:
        bad(f"NF4 test FAILED: {str(e)[:300]}")
        res.update(nf4_error=str(e)[:500],
                   verdict="NF4 NOT viable — prefer the GGUF/s2.cpp path")
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  stage 3 · VULKAN  (s2.cpp's only GPU backend — the real Colab unknown)
# ─────────────────────────────────────────────────────────────────────────────

def stage_vulkan() -> dict:
    hr("3 · VULKAN  (required by s2.cpp — the lowest-VRAM path)")
    res: dict = {"stage": "vulkan", "icds": [], "usable": False}

    # ICD manifests are what actually make Vulkan work in a headless container
    for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d",
              "/usr/local/share/vulkan/icd.d"):
        p = Path(d)
        if p.is_dir():
            found = sorted(f.name for f in p.glob("*.json"))
            if found:
                ok(f"ICD manifests · {d}: {', '.join(found)}")
                res["icds"].extend(found)
            else:
                info(f"{d} exists but contains no *.json ICD")

    if not res["icds"]:
        bad("no Vulkan ICD manifest found — Vulkan is NOT usable here")
        warn("s2.cpp would first need:  apt-get install -y vulkan-tools "
             "libvulkan1   (plus the NVIDIA Vulkan ICD)")

    rc, _ = sh("ldconfig -p | grep -m1 libvulkan")
    if rc == 0:
        ok("loader          libvulkan present")
    else:
        warn("libvulkan loader not found via ldconfig")

    rc, out = sh("vulkaninfo --summary")
    if rc == 0:
        ok("vulkaninfo runs")
        for l in [x.strip() for x in out.splitlines()
                  if "deviceName" in x][:3]:
            info(f"vulkaninfo     {l}")
        res["usable"] = bool(res["icds"])
    else:
        warn("`vulkaninfo` unavailable/failed "
             "(apt-get install -y vulkan-tools)")

    rc, out = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if rc == 0 and out.strip():
        info(f"nvidia driver   {out.strip().splitlines()[0]}")

    if res["usable"]:
        res["verdict"] = "Vulkan usable — s2.cpp GGUF path is open"
        ok("Vulkan usable → s2.cpp q4_k_m (~4 GB) is worth trying")
    else:
        res["verdict"] = "Vulkan NOT usable — s2.cpp path blocked"
        warn("If Vulkan cannot be enabled, the GGUF path is off the table on "
             "this runtime — NF4 (stage 2) becomes the only self-host route")
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  stage 4 · VRAM PLANNER  (what actually fits this GPU)
# ─────────────────────────────────────────────────────────────────────────────

S2_PROFILES = [
    ("official bf16/f16, single GPU", 21.0, "not viable on 16 GB"),
    ("official fp16 --half, 2 GPUs", 21.0, "needs TWO 16 GB cards"),
    ("drbaph fp8", 6.2, "needs CC>=8.9 (Ada/Hopper)"),
    ("groxaxo BnB NF4", 12.0, "pure PyTorch; needs CC>=7.5"),
    ("baicai1145 GPTQ w4a16", 5.5, "Turing support unverified"),
    ("s2.cpp gguf f16 + f16 codec", 10.6, "reference quality; needs Vulkan"),
    ("s2.cpp gguf q8_0 + q8_0 codec", 6.4, "near-lossless; needs Vulkan"),
    ("s2.cpp gguf q4_k_m + q4_k_m", 3.8, "smallest; needs Vulkan"),
]


def stage_plan(facts: dict) -> dict:
    hr("4 · VRAM PLANNER — what fits this GPU")
    vram = facts.get("vram_gb")
    if vram:
        info(f"available VRAM  {vram} GB  (compute {facts.get('cc', '?')})")
    else:
        warn("no GPU detected — showing the general table only")

    res: dict = {"stage": "plan", "profiles": []}
    for name, gb, note in S2_PROFILES:
        fits = bool(vram) and gb < (vram - 1.5)
        print(f"  {'OK ' if fits else 'NO '} {name:32s} ~{gb:5.1f} GB  "
              f"({note})")
        res["profiles"].append({"name": name, "gb": gb,
                                "note": note, "fits": fits})
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  stage 5 · HINDI BASELINE  (CosyVoice 2.0, for A/B comparison)
# ─────────────────────────────────────────────────────────────────────────────

def stage_ab() -> dict:
    hr("5 · HINDI BASELINE — CosyVoice 2.0 (for A/B against S2)")
    res: dict = {"stage": "ab"}
    info(f"test line ({len(HINDI_LINE)} chars): {HINDI_LINE[:44]}…")

    try:
        import pipeline as _p
    except Exception as e:
        warn(f"project pipeline not importable here ({e})")
        info("Run this from the repo root AFTER the CosyVoice bootstrap to "
             "capture a baseline.")
        res["verdict"] = "manual — pipeline not importable"
        return res

    if not hasattr(_p, "_load_cosyvoice") or \
            not hasattr(_p, "_cosyvoice_speak"):
        warn("pipeline lacks the CosyVoice hooks — cannot measure a baseline")
        res["verdict"] = "manual — hooks missing"
        return res

    # ── find any usable reference clip for zero-shot cloning ─────────────
    ref = ""
    try:
        cands = sorted(_p.OUTPUTS_DIR.glob("speaker*_clone_prompt.wav"))
        if cands:
            ref = str(cands[0])
        else:
            voc = getattr(_p, "VOCALS_WAV", None)
            if voc and Path(voc).exists():
                import soundfile as sf
                y, sr = sf.read(str(voc), dtype="float32")
                if getattr(y, "ndim", 1) > 1:
                    y = y.mean(axis=1)
                tmp = Path("/content/probe_ref.wav")
                sf.write(str(tmp), y[: int(6 * sr)], sr)
                ref = str(tmp)
    except Exception as e:
        warn(f"could not prepare a reference clip: {e}")

    if not ref:
        warn("no reference audio found — run Tab 1 / Tab 2 first, then re-run")
        res["verdict"] = "skipped — no reference clip"
        return res
    info(f"reference clip  {ref}")

    try:
        import soundfile as sf
        import torch

        log = _p.Log()
        t0 = time.time()
        model = _p._load_cosyvoice(log)
        load_s = time.time() - t0
        ok(f"CosyVoice loaded in {load_s:.1f}s")
        res["cosyvoice_load_s"] = round(load_s, 1)

        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        y, sr = _p._cosyvoice_speak(model, HINDI_LINE, "Speak calmly.", ref, "")
        gen_s = time.time() - t1
        audio_s = len(y) / float(sr)
        peak = torch.cuda.max_memory_allocated() / 2**30
        rtf = gen_s / max(audio_s, 1e-6)

        ok(f"CosyVoice · {audio_s:.2f}s audio in {gen_s:.2f}s → "
           f"RTF {rtf:.2f} · peak {peak:.2f} GB")
        out = _p.OUTPUTS_DIR / "probe_cosyvoice.wav"
        sf.write(str(out), y, int(sr))
        ok(f"baseline written → {out}")
        res.update(cosyvoice_rtf=round(rtf, 2),
                   cosyvoice_peak_gb=round(peak, 2),
                   cosyvoice_audio_s=round(audio_s, 2),
                   baseline_wav=str(out), verdict="baseline captured")

        del model
        _p.clear_gpu_cache()
    except Exception as e:
        bad(f"CosyVoice baseline failed: {str(e)[:220]}")
        res["verdict"] = f"error: {str(e)[:220]}"
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  verdict + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _verdict(results: dict) -> None:
    hr("VERDICT")
    env = results.get("env", {})
    bnb = results.get("bnb", {})
    vk = results.get("vulkan", {})
    ab = results.get("ab", {})

    if env.get("vram_gb"):
        cc = str(env.get("cc", "?")).replace(".", "")
        print(f"  GPU              {env.get('gpu')} · {env['vram_gb']} GB · "
              f"sm_{cc}")
    if bnb:
        print(f"  NF4  (~12 GB)    {bnb.get('verdict', 'not tested')}")
    if vk:
        print(f"  Vulkan (~4 GB)   {vk.get('verdict', 'not tested')}")
    if ab.get("cosyvoice_rtf"):
        print(f"  CosyVoice RTF    {ab['cosyvoice_rtf']}  (baseline)")
    print()

    if bnb.get("nf4_infer_ok"):
        ok("RECOMMENDATION: NF4 works here — try self-hosted S2-Pro at ~12 GB.")
        warn("Then measure RTF: if it exceeds ~3, a 1-hour dub takes hours.")
    elif vk.get("usable"):
        ok("RECOMMENDATION: try the s2.cpp GGUF route (q4_k_m ≈ 4 GB).")
    elif bnb or vk:
        warn("Neither self-host path verified. Keep CosyVoice 2.0 as the "
             "default; leave S2 as a disabled option in the selector.")
    else:
        info("Run with --bnb and --vulkan to complete the verdict.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fish Audio S2-Pro self-hosting feasibility probe "
                    "(Gate 1 for AutoDub Studio)")
    ap.add_argument("--env", action="store_true", help="environment report")
    ap.add_argument("--bnb", action="store_true", help="bitsandbytes NF4 test")
    ap.add_argument("--vulkan", action="store_true", help="Vulkan availability")
    ap.add_argument("--ab", action="store_true",
                    help="Hindi baseline via CosyVoice 2.0")
    ap.add_argument("--all", action="store_true", help="run every stage")
    ap.add_argument("--json", metavar="PATH", default="",
                    help="also write results to a JSON file")
    args = ap.parse_args()

    if not any([args.env, args.bnb, args.vulkan, args.ab, args.all]):
        args.env = True                   # default: harmless env report

    results: dict = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "script": "fish_s2_probe.py"}
    facts = stage_env()
    results["env"] = facts
    results["plan"] = stage_plan(facts)

    if args.bnb or args.all:
        results["bnb"] = stage_bnb()
    if args.vulkan or args.all:
        results["vulkan"] = stage_vulkan()
    if args.ab or args.all:
        results["ab"] = stage_ab()

    _verdict(results)

    if args.json:
        try:
            Path(args.json).write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8")
            info(f"results written → {args.json}")
        except Exception as e:
            warn(f"could not write {args.json}: {e}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)





