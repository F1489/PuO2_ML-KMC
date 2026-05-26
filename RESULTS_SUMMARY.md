# Results Summary

Главный результат проекта - двухстадийный seeded ML-kMC workflow:

```text
results/05_seeded_stage1_aggressive/
results/06_seeded_stage2_polish_1000K/
```

| Расчет | Energy start, eV/PuO2 | Energy final, eV/PuO2 | Delta E, eV/PuO2 | Bulk order | Mean coord. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | -46.6442 | -46.8323 | -0.1881 | 0.6426 -> 0.6469 | 1.5333 -> 1.5333 |
| Final 5000 steps | -46.6442 | -47.3166 | -0.6724 | 0.6426 -> 0.7914 | 1.5381 -> 1.3000 |
| Seeded + polish | -46.6442 | -47.5877 | -0.9435 | 0.63 -> 0.88 | 1.56 -> 1.12 |

`Seeded + polish` is a sequential workflow:

```text
05_seeded_stage1_aggressive -> 06_seeded_stage2_polish_1000K
```

The second stage starts from the final structure produced by stage 1, not from the original input structure.

## Interpretation

Основной seeded two-stage ML-kMC workflow дает наиболее сильное снижение энергии и наиболее заметное улучшение локального флюоритоподобного порядка. Baseline нужен как контроль: он показывает, что простой расчет снижает энергию существенно слабее, чем seeded ML-kMC workflow.

Результат следует формулировать аккуратно: это частичный отжиг дефектов, снижение энергии и локальное флюоритоподобное упорядочение структуры PuO2, а не доказательство полной рекристаллизации всего кластера.

## ML Role

ML-модель не заменяет физическую модель. Она используется как быстрый ранжировщик candidate events: предсказывает изменение энергии, помогает выбрать перспективные события, а выбранные события затем проверяются exact-оценкой энергии на потенциале MOX-07.

## Main Final Metrics

For the final structure after seeded two-stage ML-kMC relaxation:

- Energy: `-47.5877 eV/PuO2`
- Bulk order score: `0.8788`
- Mean coordination error: `1.1190`
- Close-contact safety: `True`
