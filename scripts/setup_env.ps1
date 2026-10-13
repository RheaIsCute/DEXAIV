# Create .venv and install the pinned environment (Windows + NVIDIA, cu128 wheels).
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 [-Cpu]
# -Cpu installs CPU-only torch for development on a machine without CUDA.
param([switch]$Cpu)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { python -m venv (Join-Path $root ".venv") }
& $py -m pip install --upgrade pip
if ($Cpu) {
    & $py -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    Get-Content (Join-Path $root "requirements\runtime.txt") |
        Where-Object { $_ -notmatch '^(torch|torchaudio)==' -and $_ -notmatch '^#' -and $_.Trim() } |
        Set-Content -Encoding utf8 (Join-Path $env:TEMP "vox-runtime-cpu.txt")
    & $py -m pip install -r (Join-Path $env:TEMP "vox-runtime-cpu.txt")
} else {
    & $py -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    & $py -m pip install -r (Join-Path $root "requirements\runtime.txt") --extra-index-url https://download.pytorch.org/whl/cu128
}
& $py -m pip install -r (Join-Path $root "requirements\dev.txt")
& $py -m pip install -e $root --no-deps
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Write-Host "Next: .venv\Scripts\python scripts\fetch_assets.py --all"
