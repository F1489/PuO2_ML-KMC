# Notes

Это clean-версия проекта PuO2 ML-kMC перед сдачей.

## Главное для защиты

Главным результатом считать двухстадийный seeded ML-kMC workflow:

```text
results/05_seeded_stage1_aggressive/
results/06_seeded_stage2_polish_1000K/
```

Краткая формулировка:

```text
Основной двухстадийный seeded ML-kMC расчет показывает снижение энергии,
частичный отжиг дефектов и локальное флюоритоподобное упорядочение PuO2.
```

Не формулировать результат как полную рекристаллизацию всего кластера.

## Что оставить наверху results

```text
results/01_model_best/
results/02_model_validation/
results/03_baseline_kmc/
results/04_production_5000_steps/
results/05_seeded_stage1_aggressive/
results/06_seeded_stage2_polish_1000K/
results/07_crystal_visualization/
```

`results/04_production_5000_steps/` оставлен как дополнительный production-запуск, но не как главный результат.

Промежуточные benchmark, smoke, candidate, old validation и seed-screen результаты лежат в:

```text
results/99_archive/
```

## Ключевые файлы

- `README.md` - основная инструкция и научная формулировка.
- `RESULTS_SUMMARY.md` - короткая таблица метрик для преподавателя.
- `src/crystal_visualization.py` - генерация 3D-визуализаций структуры, density heatmap и карт дефектов координации.
- `results/07_crystal_visualization/initial_crystal.xyz` и `final_crystal.xyz` - начальный и финальный кристалл для визуального сравнения.
- `requirements.txt` - минимальные версии зависимостей.
- `test.ps1` - запуск unit tests.
- `run_smoke.ps1` - короткий проверочный kMC запуск.
- `validate.ps1` - короткая validation-проверка модели.

## Контакты автора

- Telegram: <https://t.me/hytalegoekb>
- Email: <a.a.shatunov@urfu.me>
