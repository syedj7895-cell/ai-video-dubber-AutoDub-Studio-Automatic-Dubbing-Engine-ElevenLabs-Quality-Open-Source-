# ═══════════════════════════════════════════════════════════════════════════
#  ONE-SHOT surgical patcher for app.py (run once, then delete):
#    1 ▸ remove the animated hero beam (div + CSS + reduced-motion ref)
#    2 ▸ add Original-language / Target-language dropdowns to Tab 1
#    3 ▸ wire the choices into the analysis callback & pipeline
# ═══════════════════════════════════════════════════════════════════════════

import pathlib
import re
import sys

p = pathlib.Path(__file__).resolve().parents[1] / "app.py"
s = p.read_text(encoding="utf-8")


def sub(old: str, new: str, tag: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        print(f"❌ [{tag}] found {n}× (expected exactly 1) — aborting.")
        sys.exit(1)
    s = s.replace(old, new)
    print(f"✅ [{tag}]")


# 1 · css — drop the beam rule + keyframes (tolerant regex)
pat = re.compile(
    r"\.hero \.beam \{[^\n]*\n"
    r"  background:[^\n]*\n"
    r"  background-size:[^\n]*\n"
    r"@keyframes beamSlide \{[^\n]*\}\n"
)
if not pat.search(s):
    print("❌ [css beam rule] pattern not found — aborting.")
    sys.exit(1)
s = pat.sub("", s, count=1)
print("✅ [css beam rule removed]")

# 3 · reduced-motion list — drop the beam reference
sub("  .gradio-container::before, .hero .beam, .chip .dot, .tabitem "
    "{ animation: none !important; }",
    "  .gradio-container::before, .chip .dot, .tabitem "
    "{ animation: none !important; }", "reduced-motion line")

# 4 · Tab 1 — insert the two language dropdowns before the HF accordion
anchor = '                        with gr.Accordion("🔑 Advanced — Hugging Face token "'
dropdowns = (
    "                        with gr.Row():\n"
    "                            lang_orig_in = gr.Dropdown(\n"
    "                                choices=[(lbl, code) for code, lbl\n"
    "                                         in pipeline.DUBBING_LANGUAGES],\n"
    "                                value=\"auto\", label=\"🎙 Original language\",\n"
    "                                info=\"Language of the source audio — used by \"\n"
    "                                     \"the emotion/ASR scan\")\n"
    "                            lang_target_in = gr.Dropdown(\n"
    "                                choices=[(lbl, code) for code, lbl\n"
    "                                         in pipeline.DUBBING_LANGUAGES],\n"
    "                                value=\"en\", label=\"🌍 Target / dub language\",\n"
    "                                info=\"Language of your Translated SRT — the \"\n"
    "                                     \"cloned voices speak this\")\n")
sub(anchor, dropdowns + anchor, "language dropdowns")

# 5 · callback — accept the two choices and pass them to the pipeline
sub("def _run_analysis(media, srt_o, srt_t, token):",
    "def _run_analysis(media, srt_o, srt_t, token, lang_o, lang_t):",
    "callback signature")
sub("        for line in pipeline.run_import_and_analysis(media_path, force=False):",
    "        for line in pipeline.run_import_and_analysis(\n"
    "                media_path, force=False,\n"
    "                source_lang=lang_o or \"auto\",\n"
    "                target_lang=lang_t or \"en\"):",
    "pipeline call")

# 6 · wiring — feed the dropdowns as inputs
sub("            inputs=[media_in, srt_orig_in, srt_trans_in, hf_token_in],",
    "            inputs=[media_in, srt_orig_in, srt_trans_in, hf_token_in,\n"
    "                    lang_orig_in, lang_target_in],",
    "wiring inputs")

p.write_text(s, encoding="utf-8")
print("🎉 app.py patched successfully")