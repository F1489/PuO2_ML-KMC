$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m src.main_validate_ml `
  --xyz input/PuO2_324.xyz `
  --model-dir results/01_model_best `
  --out-dir results/99_archive/validation_smoke `
  --n-events 100 `
  --seed 9101
