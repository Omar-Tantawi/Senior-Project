@echo off
REM ============================================================================
REM  One-time setup: isolated Marker venv with CUDA torch (does NOT touch your
REM  main env / fight-detection torch). The big download is the ~2.5 GB CUDA torch.
REM ============================================================================
set "PY=C:\Users\Hussin\AppData\Local\Programs\Python\Python312\python.exe"
set "VENV=D:\Senior\question-generator\.marker_venv"

echo [1/4] Creating isolated venv at %VENV% ...
"%PY%" -m venv "%VENV%"

echo [2/4] Upgrading pip ...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip

echo [3/4] Installing CUDA torch 2.12 (~2.5 GB, one-time) ...
"%VENV%\Scripts\python.exe" -m pip install torch==2.12.0 torchvision --index-url https://download.pytorch.org/whl/cu126

echo [4/4] Installing Marker + helpers (mostly from cache) ...
"%VENV%\Scripts\python.exe" -m pip install marker-pdf rank-bm25 pymupdf

echo.
echo === Verifying CUDA ===
"%VENV%\Scripts\python.exe" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo.
echo DONE.  Next, scan the math/physics books on GPU with:
echo     tools\scan_math_physics.bat
pause
