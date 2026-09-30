#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#   build_secure.py — Phase 6 · Cython obfuscation builder
#
#   Compiles `pipeline.py` into an unreadable native extension and scrubs the
#   plaintext source, so a public Colab session cannot read the workflow.
#
#       python build_secure.py                 # compile → verify → scrub
#       python build_secure.py --keep-source   # compile → verify, keep source
#       python build_secure.py --dry-run       # translate to C only, touch nothing
#
#   SAFETY CONTRACT — why the verify step is not optional:
#     * the source is deleted ONLY after a child process has IMPORTED the
#       compiled binary with `pipeline.py` hidden and confirmed the public API
#       is present. A binary that does not import is worthless, and deleting
#       the source then would be unrecoverable.
#     * the child restores the source in a `finally`; the parent restores it
#       again afterwards, so even a killed child leaves a working tree.
#     * `--no-verify` can never scrub — you cannot opt out of the only check
#       that makes scrubbing safe.
#
#   The extension shadows the source anyway: Python resolves extension modules
#   (`.pyd` / `.so`) BEFORE `.py` sources on `sys.path`, so `app.py` keeps
#   working unchanged either way.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import List, Optional

PROJECT = Path(__file__).resolve().parent
DEFAULT_TARGET = "pipeline"

# Names that MUST exist on the compiled module. Chosen from the stable public
# surface (not internals), so the check still holds across refactors.
REQUIRED_API = [
    "BASE_DIR", "OUTPUTS_DIR", "TTS_ENGINES", "TerminalEngineError",
    "engine_error_report", "resolve_engine", "engine_fallback",
    "allocate_speaker_voices", "auto_kokoro_blends", "preview_voice",
    "speaker_overview", "run_script_matching", "step7_synthesize",
]

HIDDEN_SUFFIX = ".__hidden__"
MARKER = "<<<BUILD_SECURE_VERIFY>>>"


def note(msg: str) -> None:
    print(f"[build_secure] {msg}")


def fail(msg: str, code: int = 1) -> "None":
    print(f"[build_secure] ❌ {msg}", file=sys.stderr)
    sys.exit(code)


def _safe_streams() -> None:
    """Make stdout/stderr UTF-8 and never fatal on unencodable characters.

    Without this the script dies on a Windows cp1252 console: `→` and the emoji
    in our own log lines are not in that code page, so `print` raises
    UnicodeEncodeError in the middle of the build. Colab is already UTF-8, but
    this makes the tool work unchanged everywhere.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):          # closed / not seekable
                pass



def _has_compiler() -> Optional[str]:
    """Path of a usable C compiler, or None. (Colab ships gcc; Windows may not.)"""
    for name in ("gcc", "clang", "cl", "cc"):
        found = shutil.which(name)
        if found:
            return found
    cc = sysconfig.get_config_var("CC")
    if cc:
        found = shutil.which(str(cc).split()[0])
        if found:
            return found
    return None


def _binary_of(project: Path, mod: str) -> Optional[Path]:
    """The extension for `mod`, if one exists."""
    hits = sorted(project.glob(f"{mod}*.pyd")) + sorted(project.glob(f"{mod}*.so"))
    hits = [h for h in hits if h.is_file()]
    return hits[-1] if hits else None


def _ensure_restored(project: Path, mod: str) -> bool:
    """Put `mod.py` back if a verify run left it hidden. True if restored."""
    src = project / f"{mod}.py"
    bak = project / f"{mod}.py{HIDDEN_SUFFIX}"
    if not src.exists() and bak.exists():
        bak.replace(src)
        note(f"restored {src.name} from the verify backup")
        return True
    return False



# ── step 1 · preflight ───────────────────────────────────────────────────────
def preflight(project: Path, mod: str, dry_run: bool) -> Path:
    src = project / f"{mod}.py"
    if not src.exists():
        fail(f"{src} not found — nothing to build.")
    try:
        import cython  # noqa: F401
    except ImportError:
        fail("Cython is not installed → pip install cython setuptools")
    try:
        import setuptools  # noqa: F401
    except ImportError:
        fail("setuptools is not installed → pip install setuptools")
    if not dry_run and not _has_compiler():
        fail(
            "no C compiler found (looked for gcc, clang, cl, cc).\n"
            "        Colab ships gcc — a missing compiler is a local limitation.\n"
            "        Run `python build_secure.py --dry-run` to translate to C\n"
            "        without compiling; that modifies NOTHING.",
            code=2)
    return src


# ── step 2 · translate Python → C ────────────────────────────────────────────
def cythonize(project: Path, mod: str, work: Path) -> Path:
    c_file = work / f"{mod}.c"
    note(f"cythonizing {mod}.py → {c_file.name} (a few seconds)")
    r = subprocess.run(
        [sys.executable, "-m", "cython", "-3", "--output-file", str(c_file),
         str(project / f"{mod}.py")],
        cwd=str(project), capture_output=True, text=True)
    if r.returncode or not c_file.exists():
        fail("Cython translation failed:\n"
             + (r.stderr or r.stdout or "").strip()[-2000:])
    note(f"translated OK — {c_file.stat().st_size / 1e6:.1f} MB of C")
    return c_file


# ── step 3 · compile the C → native extension, in place ──────────────────────
def compile_ext(project: Path, mod: str, c_file: Path) -> Path:
    build_py = Path(tempfile.mkdtemp(prefix="autodub-build-")) / "build_ext.py"
    build_py.write_text(
        "from setuptools import Extension, setup\n"
        "setup(name='autodub_obfuscated',\n"
        f"      ext_modules=[Extension({mod!r}, [{str(c_file)!r}])],\n"
        "      script_args=['build_ext', '--inplace'])\n",
        encoding="utf-8")
    note(f"compiling {c_file.name} → native extension ({_has_compiler()})")
    r = subprocess.run([sys.executable, str(build_py)], cwd=str(project),
                       capture_output=True, text=True)
    if r.returncode:
        fail("C compilation failed (source left untouched):\n"
             + (r.stderr or r.stdout or "").strip()[-2000:])
    binary = _binary_of(project, mod)
    if binary is None:
        fail("build reported success but produced no extension file — "
             "source left untouched.")
    note(f"built {binary.name} ({binary.stat().st_size / 1e6:.1f} MB)")
    return binary


# ── step 4 · prove the binary imports WITHOUT the source ─────────────────────
_VERIFY_SRC = f'''
import json, sys, traceback, pathlib
proj, mod = sys.argv[1], sys.argv[2]
required = json.loads(sys.argv[3])
src = pathlib.Path(proj) / (mod + ".py")
bak = pathlib.Path(proj) / (mod + ".py{HIDDEN_SUFFIX}")
sys.path.insert(0, proj)
moved, res = False, {{}}
try:
    if src.exists():
        src.replace(bak); moved = True
    sys.modules.pop(mod, None)
    m = __import__(mod)
    f = getattr(m, "__file__", "") or ""
    res["file"] = f
    res["binary"] = f.endswith((".pyd", ".so", ".dll"))
    res["missing"] = [n for n in required if not hasattr(m, n)]
    res["base"] = str(getattr(m, "BASE_DIR", ""))
    res["ok"] = bool(res["binary"] and not res["missing"])
except Exception:
    res["ok"] = False
    res["err"] = traceback.format_exc()
finally:
    if moved and not src.exists() and bak.exists():
        bak.replace(src)
    res["restored"] = src.exists()
print({MARKER!r} + json.dumps(res))
'''


def verify(project: Path, mod: str) -> dict:
    """Import the compiled module with the source hidden → verdict dict."""
    note("verifying: importing the binary with the source hidden ...")
    r = subprocess.run(
        [sys.executable, "-c", _VERIFY_SRC, str(project), mod,
         json.dumps(REQUIRED_API)],
        capture_output=True, text=True, cwd=str(project))
    _ensure_restored(project, mod)                 # belt and braces

    line = next((ln for ln in (r.stdout or "").splitlines()
                 if ln.startswith(MARKER)), "")
    if not line:
        return {"ok": False, "err": (r.stderr or r.stdout or
                                     "verifier produced no verdict").strip()[-2000:]}
    try:
        return json.loads(line[len(MARKER):])
    except json.JSONDecodeError as e:
        return {"ok": False, "err": f"unparseable verdict: {e}"}


def can_scrub(keep_source: bool, no_verify: bool) -> bool:
    """True ONLY when scrubbing the source is allowed.

    `--no-verify` can never scrub: skipping the import check is fine for a
    throwaway build, but it is exactly the check that makes deletion safe.
    """
    return not keep_source and not no_verify


# ── main ─────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="build_secure.py",
        description="Compile pipeline.py into a native extension and scrub the "
                    "plaintext (Phase 6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The source is only removed after the compiled binary has been\n"
               "imported successfully with the source hidden.")
    ap.add_argument("--target", default=DEFAULT_TARGET, metavar="MODULE",
                    help=f"module to compile (default: {DEFAULT_TARGET})")
    ap.add_argument("--keep-source", action="store_true",
                    help="compile and verify but leave the .py in place")
    ap.add_argument("--dry-run", action="store_true",
                    help="translate to C only; build nothing, delete nothing")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a binary already exists")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the import check (implies --keep-source; "
                         "scrubbing is never allowed unverified)")
    args = ap.parse_args(argv)
    _safe_streams()                 # only when run, never on import

    project, mod = PROJECT, args.target
    src = project / f"{mod}.py"
    existing = _binary_of(project, mod)

    if existing and not args.force and not args.dry_run:
        note(f"{existing.name} already exists — nothing to do "
             f"(use --force to rebuild)")
        note("source already scrubbed." if not src.exists() else
             "source is still present (built with --keep-source).")
        return 0

    preflight(project, mod, args.dry_run)

    with tempfile.TemporaryDirectory(prefix="autodub-cython-") as td:
        c_file = cythonize(project, mod, Path(td))

        if args.dry_run:
            note("dry-run complete: Cython translation succeeded.")
            note(f"• {mod}.py untouched · no compiler invoked · nothing deleted.")
            note("run without --dry-run to produce the binary and scrub source.")
            return 0

        binary = compile_ext(project, mod, c_file)

    # Scrubbing needs BOTH flags off, so --no-verify can never scrub.
    if not can_scrub(args.keep_source, args.no_verify):
        verdict = {"ok": True, "file": "(verification skipped)"} \
            if args.no_verify else verify(project, mod)
        if not verdict.get("ok"):
            note("❌ verification FAILED — source kept.\n"
                 + str(verdict.get("err", ""))[:1550])
            _ensure_restored(project, mod)
            return 1
        note("source kept"
             + (" (--keep-source)." if args.keep_source else
                " — --no-verify never scrubs."))
        return 0

    verdict = verify(project, mod)
    if not verdict.get("ok"):
        note("❌ verification FAILED — source kept, binary is not usable.\n"
             + str(verdict.get("err", ""))[:1550])
        _ensure_restored(project, mod)
        return 1

    note(f"verified: {verdict.get('file', '')} · BASE_DIR={verdict.get('base', '?')}"
         + f" · {len(REQUIRED_API)} API names present")

    _ensure_restored(project, mod)                 # the child already did; be sure
    if not src.exists():
        fail("source went missing before scrub — refusing to continue.")
    if binary is None or not binary.exists():
        fail("no binary present — refusing to delete the source.")

    src.unlink()
    note(f"✅ scrubbed {mod}.py — {binary.name} now serves the module.")
    note("   app.py keeps working unchanged "
         "(extensions shadow .py sources on sys.path).")
    return 0


if __name__ == "__main__":
    sys.exit(main())


