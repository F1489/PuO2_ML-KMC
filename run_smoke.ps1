$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m src.main_run_kmc `
  --xyz input/PuO2_324.xyz `
  --model-dir results/01_model_best `
  --steps 20 `
  --out-dir results/99_archive/smoke_manual `
  --n-candidates-per-step 32 `
  --exact-shortlist-size 8 `
  --uncertainty-shortlist-size 4 `
  --exact-check-interval 5 `
  --reject-exact-delta-above 0 `
  --pre-relaxation-steps 100 `
  --save-xyz-interval 10 `
  --seed 9100
