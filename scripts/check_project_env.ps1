$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) {
    Write-Error "Missing project virtual environment: $VenvPython"
}

$env:PYTHONPATH = $Root

Write-Host "[CHECK] Project root: $Root"
Write-Host "[CHECK] PYTHONPATH:   $env:PYTHONPATH"
Write-Host "[CHECK] Venv python:  $VenvPython"
Write-Host ""

& $VenvPython -c @"
import importlib
import os
import pathlib
import sys

root = pathlib.Path(r'$Root')
print('[PY] sys.executable =', sys.executable)
print('[PY] sys.version    =', sys.version.splitlines()[0])
print('[PY] cwd            =', pathlib.Path.cwd())
print('[PY] PYTHONPATH     =', os.environ.get('PYTHONPATH'))

if pathlib.Path(sys.executable).resolve() != (root / '.venv' / 'Scripts' / 'python.exe').resolve():
    raise SystemExit('Interpreter is not the project .venv python.')

if not sys.version.startswith('3.11'):
    raise SystemExit('Python 3.11 is required for this project.')

mods = ['torch', 'torchvision', 'transformers', 'timm', 'fastapi', 'uvicorn', 'chainlit', 'httpx', 'PIL']
missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f'{name}: {exc}')

if missing:
    print('[PY] Missing or broken modules:')
    for item in missing:
        print('  -', item)
    raise SystemExit(1)

import torch
print('[PY] torch           =', torch.__version__)
print('[PY] torch.cuda      =', torch.version.cuda)
print('[PY] cuda available  =', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[PY] gpu             =', torch.cuda.get_device_name(0))

print('[PY] Environment check passed.')
"@
