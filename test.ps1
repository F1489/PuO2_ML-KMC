$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m pytest -q
