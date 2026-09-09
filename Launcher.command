#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  AUTO-DUB STUDIO - macOS desktop bridge (Phase 7)
#  Double-click: a Terminal flash may appear for an instant, then
#  everything runs detached — the silent bridge polls the remote
#  Colab Gradio tunnel in the background and opens your default
#  browser straight into the hosted UI when it answers.
#
#  First run only: if Gatekeeper complains, right-click this file
#  ▸ Open ▸ Open once. After that it launches normally.
# ════════════════════════════════════════════════════════════════
cd "$(dirname "$0")" || exit 1

# fully detached background routine — nothing lingers on screen
nohup python3 tools/bridge.py >/dev/null 2>&1 &

exit 0