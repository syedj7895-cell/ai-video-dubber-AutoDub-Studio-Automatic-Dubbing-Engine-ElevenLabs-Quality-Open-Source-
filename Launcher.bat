@echo off
rem ════════════════════════════════════════════════════════════════
rem  AUTO-DUB STUDIO - Windows desktop bridge (Phase 7)
rem  Double-click: no console window survives. A hidden WScript host
rem  launches the silent pythonw bridge, which polls the remote
rem  Colab Gradio tunnel in the background and drops your default
rem  browser straight into the hosted UI when it answers.
rem ════════════════════════════════════════════════════════════════
setlocal
cd /d "%~dp0"

rem prefer the windowless interpreter; fall back to python if missing
set "PYW=pythonw"
where pythonw >nul 2>nul || set "PYW=python"

rem hand the entire job to a hidden WScript host (window style 0)
> "%TEMP%\autodub_bridge.vbs" echo Set sh = CreateObject("WScript.Shell")
>> "%TEMP%\autodub_bridge.vbs" echo sh.CurrentDirectory = "%~dp0"
>> "%TEMP%\autodub_bridge.vbs" echo sh.Run """%PYW%"" ""%~dp0tools\bridge.py""", 0, False
wscript.exe "%TEMP%\autodub_bridge.vbs"

endlocal
exit