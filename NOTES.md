# Notes

Это clean-версия расширенного архива. Имена пользовательских папок упрощены:

Контакты автора:

- Telegram: <https://t.me/hytalegoekb>
- Email: <a.a.shatunov@urfu.me>

- `input/` вместо `data/input/`
- `results/` вместо `data/output/`
- `results/final_5000_steps` вместо `final_5000_optimized_ml_kmc_run`
- `results/best_model` вместо `models`
- `results/crystallization_seeded` вместо `seeded_production_run`

Главный результат: `results/final_5000_steps`.

Лучшая модель: `results/best_model`.

Кристаллизация: `results/crystallization_seeded` как supplementary seeded/local-ordering evidence.

Красивые графики для отчета:

- `results/final_5000_steps/publication_figures/main_result_summary.png`
- `results/final_5000_steps/publication_figures/model_diagnostics.png`
- `results/crystallization_seeded/publication_figures/seeded_crystallization_summary.png`

Проверка:

```powershell
.\test.ps1
```

Короткий запуск:

```powershell
.\run_smoke.ps1
```

Файлы в `src/` не переименованы намеренно: это Python-модули, и их имена участвуют в imports.
