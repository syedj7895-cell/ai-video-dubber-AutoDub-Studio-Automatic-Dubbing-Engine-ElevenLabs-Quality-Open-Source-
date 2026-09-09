#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  AUTO-DUB STUDIO · DESKTOP BRIDGE (Phase 7)
#
#  A silent, headless worker: it polls the remote Colab Gradio proxy tunnel
#  in an asynchronous background routine and, the moment the tunnel answers,
#  hands the user's default browser straight into the hosted frontend.
#  No console windows, no deployment noise — launchers start this with
#  `pythonw` (Windows) / `nohup` (macOS) so it never surfaces on screen.
#
#  The tunnel address lives in `tunnel.txt` beside the launchers (first
#  http(s) line wins). Optional direct override:
#      pythonw tools/bridge.py https://xxxxxxxx.gradio.live
# ═══════════════════════════════════════════════════════════════════════════

import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNNEL_FILE = ROOT / "tunnel.txt"
POLL_SEC = 3.0          # probe cadence
TIMEOUT_SEC = 5.0       # per-probe socket timeout
GIVE_UP_SEC = 15 * 60   # poll politely for 15 min, then exit silently


def read_url() -> str:
    """URL from argv, else first http(s) line inside tunnel.txt."""
    if len(sys.argv) > 1 and sys.argv[1].lower().startswith("http"):
        return sys.argv[1].strip()
    if TUNNEL_FILE.exists():
        for line in TUNNEL_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.lower().startswith(("http://", "https://")) \
                    and "REPLACE-ME" not in line:
                return line
    return ""


def alive(url: str, timeout: float = TIMEOUT_SEC) -> bool:
    """True when the tunnel answers like a live Gradio frontend."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return int(getattr(r, "status", 200)) < 400
    except Exception:
        return False


def watch(url: str, found: threading.Event) -> None:
    """Asynchronous watcher — flags `found` the moment the tunnel is up."""
    while not found.is_set():
        if alive(url):
            found.set()
            return
        found.wait(POLL_SEC)


def main() -> int:
    url = read_url()
    if not url:
        # nothing configured — surface the template so the user pastes theirs
        try:
            if sys.platform.startswith("win"):
                os_startfile = getattr(__import__("os"), "startfile")
                os_startfile(str(TUNNEL_FILE))
            else:
                webbrowser.open(TUNNEL_FILE.as_uri())
        except Exception:
            pass
        return 1

    print(f"🛰  AutoDub bridge · waiting for tunnel: {url}")  # hidden process

    found = threading.Event()
    threading.Thread(target=watch, args=(url, found), daemon=True).start()
    found.wait(GIVE_UP_SEC)

    if found.is_set():
        webbrowser.open(url)          # clean redirect into the hosted UI
        time.sleep(2)                 # linger briefly in case of a cold start
        return 0

    return 1                          # tunnel never woke — exit silently


if __name__ == "__main__":
    raise SystemExit(main())