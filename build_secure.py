#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  BUILD SECURE — Cython obfuscation compiler (Phase 6)
#
#  Shields the core dubbing workflow from theft / reverse-engineering inside
#  public Google Colab sessions:
#    1 ▸ Cython translates pipeline.py into heavily obfuscated C/C++ source
#    2 ▸ a C compiler builds it into an unreadable binary module
#         · Linux / Colab  →  pipeline.cpython-*.so   (target asset)
#         · Windows        →  pipeline.*.pyd          (dev convenience)
#    3 ▸ once the binary is verified importable, an automated scrub deletes
#       the plain-text pipeline.py (and intermediate .c) so ONLY the secure
#       binary ships — app.py keeps importing `pipeline` transparently,
#       because extension modules take priority over .py files.
#
#  Usage:
#     python build_secure.py                 # compile → verify → scrub source
#     python build_secure.py --keep-source   # compile, but keep pipeline.py
#
#  Requirements:  pip install cython setuptools      (+ a C toolchain —
#  gcc/build-essential is pre-installed on Google Colab)
# ═══════════════════════════════════════════════════════════════════════════

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "pipeline.py"


def find_binary() -> Path | None:
    for pattern in ("pipeline*.so", "pipeline*.pyd"):
        hits = sorted(ROOT.glob(pattern))
        if hits:
            return hits[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile pipeline.py into an unreadable binary module "
                    "and scrub the plaintext source.")
    parser.add_argument("--keep-source", action="store_true",
                        help="do NOT delete pipeline.py after a successful build")
    args = parser.parse_args()

    try:
        from Cython.Build import cythonize
        from setuptools import Extension, setup
    except ImportError:
        print("❌ missing build tools →  pip install cython setuptools")
        return 1

    if not SRC.exists():
        if find_binary():
            print("✔ pipeline is already compiled (binary present) — nothing to do.")
            return 0
        print("❌ pipeline.py not found.")
        return 1

    os.chdir(ROOT)  # keep compiler paths relative & predictable

    print("⚙  Cython · translating pipeline.py → obfuscated C/C++ …")
    ext = Extension("pipeline", ["pipeline.py"])
    try:
        exts = cythonize(
            [ext],
            compiler_directives={
                "language_level": "3",
                "binding": False,          # leaner, harder to introspect
                "annotation_typing": False,  # CRITICAL: the binary must behave
                                             # EXACTLY like the Python module —
                                             # never enforce `str`-style type
                                             # annotations (Gradio passes
                                             # NamedString, a str subclass)
                "boundscheck": False,
                "wraparound": False,
            },
            nthreads=2,
        )
    except Exception as e:
        print(f"❌ Cython translation failed — pipeline.py kept intact. ({e})")
        return 1

    print("🔨 compiling binary extension (this can take a minute) …")
    try:
        setup(name="pipeline-secure",
              ext_modules=exts,
              script_args=["build_ext", "--inplace"])
    except Exception as e:
        print(f"❌ compilation failed — pipeline.py kept intact. ({e})")
        return 1

    binary = find_binary()
    if binary is None:
        print("❌ no binary produced — pipeline.py kept intact.")
        return 1

    # verify the binary actually loads BEFORE destroying the source —
    # extension modules shadow the .py, so this import tests the .so itself
    probe = subprocess.run(
        [sys.executable, "-c",
         "import pipeline; assert hasattr(pipeline, 'run_rendering'); print('BINARY_OK')"],
        cwd=ROOT, capture_output=True, text=True)
    if "BINARY_OK" not in probe.stdout:
        print("❌ binary failed the import check — pipeline.py kept intact.")
        print(probe.stderr[-600:])
        return 1

    if args.keep_source:
        print("🧷 --keep-source set → pipeline.py left in place (dev mode).")
    else:
        # 🧨 automated scrub — the plaintext workflow must not ship
        SRC.unlink()
        for junk in ROOT.glob("pipeline.c"):
            junk.unlink(missing_ok=True)
        shutil.rmtree(ROOT / "build", ignore_errors=True)
        print("🧨 pipeline.py scrubbed — only the secure binary remains.")

    print(f"🔐 done → {binary.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())