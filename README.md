# PuO2 ML-kMC Clean Package

Это расширенная, но более понятно разложенная версия проекта PuO2 ML-kMC. Главный результат, лучшая модель, валидация и дополнительные расчеты лежат в папке `results/` с простыми именами.

## Быстрый старт

```powershell
python -m pip install -r requirements.txt
.\test.ps1
.\run_smoke.ps1
```

Главный финальный расчет:

```text
results/final_5000_steps
```

Лучшая текущая модель:

```text
results/best_model
```

Исходная структура:

```text
input/PuO2_324.xyz
```

## Простая структура

```text
PuO2_ML_KMC_clean/
  README.md
  NOTES.md
  PATHS.md
  requirements.txt
  test.ps1
  validate.ps1
  run_smoke.ps1
  input/
    PuO2_324.xyz
  src/
    Python-код проекта
  tests/
    pytest-тесты
  results/
    best_model/
    validation/
    final_5000_steps/
    crystallization_seeded/
    baseline/
    production_old/
    short_ml_run/
    smoke_best_model/
    candidate_model/
    training_events.csv
```

Файлы внутри `src/` оставлены с техническими именами, потому что Python imports зависят от этих имен. Пользовательские результаты и скрипты переименованы проще.

## Главные папки результатов

| Путь | Что это |
|---|---|
| `results/best_model` | финальная хорошая модель LightGBM ensemble + classifier |
| `results/training_events.csv` | mixed dataset для обучения |
| `results/validation` | независимая валидация выбранной модели |
| `results/final_5000_steps` | главный финальный ML-kMC расчет на 5000 шагов |
| `results/crystallization_seeded` | seeded crystallization / local ordering workflow |
| `results/baseline` | baseline-сравнение |
| `results/production_old` | старый production-like ML-kMC результат |
| `results/short_ml_run` | короткий ML-kMC comparison run |
| `results/smoke_best_model` | smoke run для текущей модели |
| `results/candidate_model` | active-learning candidate model, оставлена для сравнения |
| `results/validation_candidate_model` | validation candidate model |

## Красивые графики для отчета

Для отчета и презентации лучше использовать новые publication-style фигуры:

```text
results/final_5000_steps/publication_figures/main_result_summary.png
results/final_5000_steps/publication_figures/model_diagnostics.png
results/crystallization_seeded/publication_figures/seeded_crystallization_summary.png
```

Они сделаны в более строгом научном стиле: белый фон, спокойная палитра, тонкая сетка, крупные панели, аннотации итоговых метрик и без перегруженных подписей.

Перегенерировать их можно так:

```powershell
python -m src.publication_plots results/final_5000_steps
python -m src.publication_plots results/crystallization_seeded --seeded
```

## Финальный результат 5000 шагов

Итоги из `results/final_5000_steps/summary.json`:

| Metric | Value |
|---|---:|
| Steps | 5000 |
| Speed | 1.47 steps/s |
| Applied events | 4911 |
| Acceptance ratio | 0.9822 |
| Initial E/PuO2 | -46.6442 eV |
| After repair E/PuO2 | -46.7132 eV |
| Final E/PuO2 | -47.3166 eV |
| Total Delta E/PuO2 | -0.6724 eV |
| kMC Delta E/PuO2 | -0.6034 eV |
| fluorite_order_score | 0.5238 -> 0.6214 |
| bulk_fluorite_order_score | 0.6426 -> 0.7914 |
| mean_abs_coordination_error | 1.5333 -> 1.3000 |
| Final close-contact thresholds satisfied | true |

Основная формулировка результата: модель и ML-kMC pipeline дают энергетическую релаксацию, отжиг дефектов и частичное fluorite-like локальное упорядочение. Это не нужно называть полной рекристаллизацией.

## Кристаллизация

Кристаллизация в проекте поддержана отдельным workflow:

```text
results/crystallization_seeded
```

В коде есть:

- order-biased score: `S = Delta E - lambda * Delta Q`;
- crystallization-oriented events: `snap_to_fluorite_site`, `growth_front`, `local_cluster_affine`;
- seeded two-stage workflow: `src/main_run_seeded_crystallization.py`;
- метрики `bulk_fluorite_order_score`, `soft_coordination_order_score`, `crystalline_core_size`, `growth_front_size`.

Для защиты лучше говорить: seeded workflow показывает локальное seeded ordering и рост bulk-order метрик, но из-за конечного кластера и поверхности это supplementary evidence, а не доказательство полной кристаллизации PuO2.

## Модель

Финальные модели:

```text
results/best_model/regressor.joblib
results/best_model/classifier.joblib
results/best_model/feature_columns.joblib
results/best_model/classifier_threshold.json
```

Качество независимой validation:

```text
sign_accuracy: 0.839
recall_energy_lowering: 0.862
R2: 0.7446
top_5_precision: 0.96
```

## Команды

Тесты:

```powershell
.\test.ps1
```

Короткий smoke:

```powershell
.\run_smoke.ps1
```

Валидация smoke:

```powershell
.\validate.ps1
```

Повторить финальный 5000-step расчет:

```powershell
python -m src.main_run_kmc --xyz input/PuO2_324.xyz --model-dir results/best_model --steps 5000 --out-dir results/final_5000_steps_new --n-candidates-per-step 64 --exact-shortlist-size 8 --uncertainty-shortlist-size 4 --exact-check-interval 25 --reject-exact-delta-above 0 --pre-relaxation-steps 300 --save-xyz-interval 1000 --seed 20260527
```

## Оптимизация скорости

В коде ускорены горячие участки:

- `src/potentials.py`: локальная энергия и сила для trial-позиции считаются без полной пересборки структуры.
- `src/features.py`: energy+force считаются вместе за один обход соседей.
- `src/kmc.py`: центр структуры и `cKDTree` переиспользуются в выборе события.

Текущий короткий benchmark после оптимизации: около `1.63 steps/s` на 50 шагах с 64 кандидатами.
