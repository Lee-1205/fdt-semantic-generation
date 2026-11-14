$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$freeBytes = (Get-PSDrive C).Free
if ($freeBytes -lt 32GB) { throw '설치 임시 용량과 최소20GiB 여유를 위해32GiB 이상이 필요합니다.' }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { py -3.10 -m venv .venv }
& ./.venv/Scripts/python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed' }
& ./.venv/Scripts/python.exe -m pip install numpy==2.2.6 PyYAML==6.0.3 filelock==4.1.0 typing_extensions sympy networkx jinja2 fsspec pytest==8.4.2
if ($LASTEXITCODE -ne 0) { throw 'dependency install failed' }
& ./.venv/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA torch install failed' }
& ./.venv/Scripts/python.exe -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'package install failed' }
& ./.venv/Scripts/python.exe -B -X utf8 -m fdt_local preflight --config configs/local_baseline.yaml
if ($LASTEXITCODE -ne 0) { throw 'preflight failed' }
