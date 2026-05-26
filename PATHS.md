# Simple Path Map

Old output paths -> clean package paths:

- data/output/models -> results/01_model_best
- data/output/ml_training_mixed_dataset.csv -> results/training_events.csv
- data/output/validation_mixed_models_threshold -> results/02_model_validation
- data/output/final_5000_optimized_ml_kmc_run -> results/04_production_5000_steps
- data/output/seeded_production_run -> results/99_archive/crystallization_seeded
- data/output/production_like_ml_kmc_run -> results/99_archive/production_old
- data/output/final_baseline_run -> results/03_baseline_kmc
- data/output/final_ml_kmc_run -> results/99_archive/short_ml_run
- data/output/new_models_kmc_smoke_relaxed -> results/99_archive/smoke_best_model
- data/output/models_active_candidate -> results/99_archive/candidate_model
- data/output/active_candidate_kmc_smoke_relaxed -> results/99_archive/smoke_candidate_model
- data/output/validation_active_candidate_mixed -> results/99_archive/validation_candidate_model

Code module names in src/ are intentionally unchanged because Python imports depend on them.
