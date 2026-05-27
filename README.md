# PuO2 ML-kMC

Проект демонстрирует ML-kMC workflow для моделирования локального упорядочения и дефектной релаксации структуры PuO2. ML-модель используется как быстрый ранжировщик candidate events: она предсказывает изменение энергии и помогает выбрать наиболее перспективные локальные перестройки. После этого выбранные события проверяются exact-оценкой энергии на основе потенциала MOX-07.

## Главный результат

Основной результат проекта - двухстадийный seeded ML-kMC расчет:

```text
results/05_seeded_stage1_aggressive/
results/06_seeded_stage2_polish_1000K/
```

Интерпретация стадий:

- `results/05_seeded_stage1_aggressive/` - агрессивный этап поиска низкоэнергетической структуры.
- `results/06_seeded_stage2_polish_1000K/` - финальная низкотемпературная polishing-релаксация.

Ключевые итоговые метрики для всей цепочки:

| Metric | Initial | Final |
| --- | ---: | ---: |
| Energy, eV/PuO2 | -46.6442 | -47.5877 |
| Delta E, eV/PuO2 |  | -0.9435 |
| Bulk fluorite order score | 0.63 | 0.88 |
| Mean coordination error | 1.56 | 1.12 |
| Close-contact safety | False | True |

Полученный результат следует интерпретировать как частичный отжиг дефектов, снижение энергии и локальное флюоритоподобное упорядочение структуры PuO2, а не как полную рекристаллизацию всего кластера.

## Дополнительные сравнения

Обязательные папки результатов:

```text
results/01_model_best/
results/02_model_validation/
results/03_baseline_kmc/
results/04_production_5000_steps/
results/05_seeded_stage1_aggressive/
results/06_seeded_stage2_polish_1000K/
results/07_crystal_visualization/
```

Роль папок:

- `results/01_model_best/` - выбранная ML-модель.
- `results/02_model_validation/` - независимая проверка качества ML-модели.
- `results/03_baseline_kmc/` - baseline-расчет без основного seeded two-stage workflow.
- `results/04_production_5000_steps/` - дополнительный production-запуск на 5000 шагов.
- `results/05_seeded_stage1_aggressive/` и `results/06_seeded_stage2_polish_1000K/` - главный двухстадийный seeded workflow.
- `results/07_crystal_visualization/` - отдельные визуализации кристаллической структуры для отчета.

Промежуточные smoke, benchmark, candidate и seed-screen результаты перенесены в:

```text
results/99_archive/
```

## Сравнение с baseline

Baseline-расчет снижает энергию только на `-0.1881 eV/PuO2`, тогда как основной seeded two-stage ML-kMC workflow снижает энергию примерно на `-0.9435 eV/PuO2`. Это показывает, что ML-ранжирование событий существенно повышает эффективность поиска низкоэнергетических перестроек.

Краткая таблица главных метрик вынесена в [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

## Структура проекта

```text
PuO2_ML_KMC_clean/
  input/
    PuO2_324.xyz

  src/
    analysis.py
    dataset.py
    events.py
    features.py
    io_xyz.py
    kmc.py
    ml_model.py
    potentials.py
    publication_plots.py
    repair.py
    seeded.py
    validation.py
    visualization.py
    main_generate_dataset.py
    main_train_ml.py
    main_validate_ml.py
    main_run_kmc.py
    main_run_seeded_crystallization.py
    main_run_baseline.py
    main_compare.py
    main_active_learning.py

  tests/
    pytest tests

  results/
    01_model_best/
    02_model_validation/
    03_baseline_kmc/
    04_production_5000_steps/
    05_seeded_stage1_aggressive/
    06_seeded_stage2_polish_1000K/
    07_crystal_visualization/
    99_archive/
```

## Установка

Откройте PowerShell или терминал в папке проекта:

```powershell
cd path/to/PuO2_ML_KMC_clean
```

Установите зависимости:

```powershell
python -m pip install -r requirements.txt
```

Для полной фиксации локального окружения можно создать lock-файл:

```powershell
python -m pip freeze > requirements-lock.txt
```

## Quick Check

Запуск unit tests:

```powershell
python -m pytest -q
```

Full ML pipeline test trains several tree-based models. In unit tests it uses a small `fast_mode` configuration so the complete test suite remains suitable as a quick check.

Короткий smoke-run ML-kMC:

```powershell
python -m src.main_run_kmc --xyz input/PuO2_324.xyz --model-dir results/01_model_best --steps 20 --out-dir results/99_archive/smoke_manual --n-candidates-per-step 32 --exact-shortlist-size 8 --uncertainty-shortlist-size 4 --exact-check-interval 5 --reject-exact-delta-above 0 --pre-relaxation-steps 100 --save-xyz-interval 10 --seed 9100
```

Короткая проверка модели на 100 событиях:

```powershell
python -m src.main_validate_ml --xyz input/PuO2_324.xyz --model-dir results/01_model_best --out-dir results/99_archive/validation_smoke --n-events 100 --seed 9101
```

Пересоздать итоговые рисунки структуры и карты дефектов:

```powershell
python -m src.crystal_visualization --initial-xyz input/PuO2_324.xyz --final-xyz results/06_seeded_stage2_polish_1000K/final.xyz --out-dir results/07_crystal_visualization --zoom-radius 6 --grid-size 12 --initial-energy-per-puo2 -46.6442 --final-energy-per-puo2 -47.5877
```

То же самое можно запустить готовыми скриптами:

```powershell
.\test.ps1
.\run_smoke.ps1
.\validate.ps1
```

## Reproducing The Main Result

The final reported result is already stored in:

```text
results/05_seeded_stage1_aggressive/
results/06_seeded_stage2_polish_1000K/
```

These folders contain the saved trajectories, final XYZ structure, history tables, plots, and summary files used for the reported metrics.

To reproduce a shorter demonstration run, use:

```powershell
.\run_smoke.ps1
```

The full production workflow is computationally heavier and may take substantially longer than the smoke run. The main reported numbers should therefore be read from the saved result folders above unless a full rerun is explicitly required.

## Ключевые параметры запуска

- `--lambda-stage1`, `--lambda-stage2` в seeded workflow и `--order-bias-lambda` в обычном ML-kMC задают силу bias к росту структурного порядка. Внутренний score события имеет вид примерно `delta_E - lambda * delta_order`: большее `lambda` сильнее поощряет события, которые улучшают флюоритоподобную координацию, даже если энергетический выигрыш не самый большой.
- `--seed-radius` задает радиус исходного флюоритоподобного seed-фрагмента в ангстремах. Больший радиус создает более крупное кристаллическое ядро перед kMC, но может сильнее навязать структуру; меньший радиус оставляет больше свободы последующей релаксации.
- `--exact-shortlist-size` задает, сколько лучших ML-кандидатов на каждом шаге дополнительно проверяются exact-энергией MOX-07. Большее значение повышает надежность выбора события, но делает запуск медленнее; меньшее ускоряет расчет, но сильнее доверяет ML-ранжированию.

## Научно-методическое позиционирование

Проект корректнее формулировать как **ML-guided/off-lattice kinetic Monte Carlo workflow for searching lower-energy defect-relaxed configurations of PuO2 clusters**.

Это не строгий предсказательный kMC для реальной кинетики PuO2. В расчете используются конечный кластер, парный потенциал MOX-07, ML-ранжирование событий, bias к структурному порядку и seeded workflow. Поэтому результат показывает эффективный поиск более низкоэнергетических дефектно-релаксированных конфигураций и локальное флюоритоподобное упорядочение, а не прямое моделирование экспериментальных времен, барьеров переходов и полной термодинамической рекристаллизации.

## Формулировка для защиты

В проекте реализован ML-ускоренный workflow кинетического Монте-Карло для моделирования дефектной релаксации и локального флюоритоподобного упорядочения структуры PuO2. ML-модель используется как быстрый ранжировщик возможных локальных перестроек, а выбранные события дополнительно проверяются exact-оценкой энергии на основе потенциала MOX-07. Основной результат - двухстадийный seeded ML-kMC расчет, в котором энергия снизилась примерно на `0.94 eV/PuO2`, bulk fluorite order score вырос примерно с `0.63` до `0.88`, а средняя ошибка координации уменьшилась. Результат интерпретируется как частичный отжиг дефектов и локальное упорядочение, а не как полная рекристаллизация.

## Главные изображения

Базовая пара структур для сравнения:

```text
input/PuO2_324.xyz
results/06_seeded_stage2_polish_1000K/final.xyz
```

Для удобства эти же XYZ-файлы продублированы в папке визуализации:

```text
results/07_crystal_visualization/initial_crystal.xyz
results/07_crystal_visualization/final_crystal.xyz
```

Для отчета удобно использовать:

```text
results/05_seeded_stage1_aggressive/initial_final_structure.png
results/05_seeded_stage1_aggressive/seeded_stage_comparison.png
results/06_seeded_stage2_polish_1000K/initial_final_structure.png
results/07_crystal_visualization/initial_crystal_visualization.png
results/07_crystal_visualization/final_crystal_visualization.png
results/07_crystal_visualization/initial_final_comparison.png
results/07_crystal_visualization/final_density_heatmap.png
results/07_crystal_visualization/initial_defect_map.png
results/07_crystal_visualization/final_defect_map.png
results/04_production_5000_steps/publication_figures/main_result_summary.png
results/04_production_5000_steps/publication_figures/model_diagnostics.png
```

Рекомендуемые подписи:

- Initial amorphized PuO2 structure.
- Final structure after seeded two-stage ML-kMC relaxation.
- Initial/final visual comparison of the PuO2 cluster.
- Final metrics: Energy `-47.5877 eV/PuO2`, bulk order score `0.8788`, mean coordination error `1.1190`, close-contact safety `True`.

Новые визуализации кристаллической структуры можно пересоздать командой. Скрипт обновляет копии XYZ в `results/07_crystal_visualization/`, делает отдельные начальную и финальную 3D-картинки, панель initial/final comparison, 2D-проекцию плотности и карты дефектов координации. Для презентационных рисунков сохраняются PNG и PDF.

```powershell
python -m src.crystal_visualization --initial-xyz input/PuO2_324.xyz --final-xyz results/06_seeded_stage2_polish_1000K/final.xyz --out-dir results/07_crystal_visualization --zoom-radius 6 --grid-size 12 --initial-energy-per-puo2 -46.6442 --final-energy-per-puo2 -47.5877
```

## Ограничения метода

1. Используется парный потенциал MOX-07, поэтому многочастичные эффекты не учитываются явно.
2. Система конечная, поэтому поверхностные атомы искажают координационные метрики.
3. ML-модель применяется только для ранжирования событий, а не как окончательный источник энергии.
4. Результат показывает локальное упорядочение и отжиг дефектов, но не доказывает полную термодинамическую кристаллизацию.
5. Для более строгого вывода нужны несколько запусков с разными random seed.

## Научный вывод

Основной двухстадийный seeded ML-kMC workflow снижает энергию структуры PuO2, улучшает bulk fluorite order score, уменьшает среднюю ошибку координации и сохраняет close-contact safety после финальной релаксации. Поэтому результат корректно описывать как локальное флюоритоподобное упорядочение и отжиг дефектов в конечном PuO2-кластере.
