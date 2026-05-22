$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m src.main_validate_ml `
  --xyz input/PuO2_324.xyz `
  --model-dir results/best_model `
  --out-dir results/validation_smoke `
  --n-events 100 `
  --seed 9101
