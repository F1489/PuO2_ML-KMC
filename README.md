# PuO2 ML-kMC Clean Package

Подробная инструкция к проекту ML-kMC для PuO2. Пакет содержит код, тесты, обученную модель, датасет, независимую валидацию, основной расчет на 5000 шагов, seeded crystallization workflow, baseline-сравнения и красивые графики для отчета.

## Author Contacts

- Telegram: <https://t.me/hytalegoekb>
- Email: <a.a.shatunov@urfu.me>

Главная идея проекта: использовать ML-модели для быстрого ранжирования локальных kMC-событий в кластере PuO2, а самые важные/сомнительные события проверять exact-расчетом энергии на потенциале MOX-07. Это ускоряет выбор событий и при этом сохраняет физический контроль через exact shortlist checks и close-contact safety.

## 1. Что считать главным результатом

Главный финальный результат:

```text
results/final_5000_steps
```

Лучшая текущая модель:

```text
results/best_model
```

Главные красивые графики для отчета:

```text
results/final_5000_steps/publication_figures/main_result_summary.png
results/final_5000_steps/publication_figures/model_diagnostics.png
results/crystallization_seeded/publication_figures/seeded_crystallization_summary.png
```

Основная научная формулировка:

```text
The ML-kMC workflow lowers the MOX-07 energy, removes unsafe close contacts after repair, reduces coordination error and improves fluorite-like local order metrics. The final 5000-step run supports partial defect annealing and local ordering, not complete PuO2 recrystallization.
```

По-русски: модель и kMC pipeline показывают энергетическую релаксацию, отжиг локальных дефектов и частичное флюоритоподобное локальное упорядочение. Не нужно утверждать полную рекристаллизацию всего PuO2-кластера.

## 2. Структура пакета

```text
PuO2_ML_KMC_clean/
  README.md                         подробная инструкция
  NOTES.md                          краткая памятка
  PATHS.md                          соответствие старых и новых путей
  requirements.txt                  Python-зависимости
  test.ps1                          запуск unit tests
  validate.ps1                      короткая validation-проверка модели
  run_smoke.ps1                     короткий kMC smoke run

  input/
    PuO2_324.xyz                    исходная PuO2-структура

  src/
    analysis.py                     структурные метрики, RDF, close-contact checks
    dataset.py                      генерация обучающих событий и mixed dataset
    events.py                       генерация kMC-событий
    features.py                     признаки событий для ML
    io_xyz.py                       чтение/запись XYZ
    kmc.py                          ML-kMC engine
    main_generate_dataset.py        CLI генерации dataset
    main_train_ml.py                CLI обучения моделей
    main_validate_ml.py             CLI независимой validation
    main_run_kmc.py                 CLI обычного ML-kMC расчета
    main_run_seeded_crystallization.py seeded crystallization CLI
    ml_model.py                     обучение, выбор моделей, prediction helpers
    potentials.py                   MOX-07 потенциал
    publication_plots.py            красивые графики для отчета
    repair.py                       repair близких контактов
    seeded.py                       наложение флюоритного seed
    validation.py                   validation metrics и ranking reports
    visualization.py                стандартные диагностические графики

  tests/
    pytest-тесты

  results/
    best_model/                     финальная выбранная ML-модель
    training_events.csv             mixed dataset
    validation/                     независимая validation выбранной модели
    final_5000_steps/               главный расчет на 5000 шагов
    crystallization_seeded/         seeded crystallization workflow
    baseline/                       baseline-сравнение
    production_old/                 старый production-like расчет
    short_ml_run/                   короткий ML-kMC comparison run
    smoke_best_model/               smoke run текущей модели
    candidate_model/                active-learning candidate model
    validation_candidate_model/     validation candidate model
```

Файлы в `src/` оставлены с техническими именами, потому что эти имена участвуют в Python imports. Пользовательские папки результатов переименованы проще.

## 3. Установка

Откройте PowerShell в папке проекта:

```powershell
cd C:\Users\Sasha1489\Desktop\py\PuO2_ML_KMC_clean
```

Установите зависимости:

```powershell
python -m pip install -r requirements.txt
```

Основные зависимости:

```text
numpy
pandas
scipy
scikit-learn
joblib
matplotlib
pytest
catboost
xgboost
lightgbm
```

Если PowerShell запрещает запуск `.ps1`, используйте:

```powershell
powershell -ExecutionPolicy Bypass -File .\test.ps1
```

## 4. Проверка проекта

Unit tests:

```powershell
python -m pytest -q
```

Или через скрипт:

```powershell
powershell -ExecutionPolicy Bypass -File .\test.ps1
```

Текущий результат:

```text
11 passed
```

Что проверяют тесты:

- чтение/запись XYZ;
- энергию MOX-07;
- генерацию событий;
- признаки событий;
- structural/validation helpers;
- загрузку новых моделей через `load_models()`;
- `predict_delta_E()`;
- `predict_good_event_probability()`;
- uncertainty для model wrapper;
- корректность mixed dataset pipeline.

## 5. Физическая модель

Используется парный потенциал MOX-07:

```text
U_ij(R_ij) = K_E * q_i * q_j / R_ij + A_ij * exp(-B_ij * R_ij) - C_ij / R_ij^6
```

Единицы:

```text
distance: angstrom, A
energy: eV
```

Заряды:

```text
q_Pu = +2.745
q_O  = -1.3725
```

Короткодействующие параметры:

| Pair | A, eV | B, A^-1 | C, eV A^6 |
|---|---:|---:|---:|
| O-O | 50212.0 | 5.5200 | 74.796 |
| O-Pu / Pu-O | 871.79 | 2.8079 | 0.0 |
| Pu-Pu | 0.0 | 0.0 | 0.0 |

Расчет идет для свободного конечного кластера, не для периодической bulk-ячейки.

## 5.1. Физический смысл процесса

Проект моделирует не обычную молекулярную динамику, а coarse-grained кинетический процесс отжига локальных дефектов. Исходная структура `PuO2_324.xyz` рассматривается как разупорядоченный или частично расплавленный finite cluster PuO2 после высокотемпературной стадии. Дальше kMC ищет последовательность локальных перестроек атомов, которые могут снижать энергию, улучшать координационное окружение и повышать флюоритоподобный порядок.

Главные физические процессы в модели:

- локальная релаксация атомов в поле сил MOX-07;
- устранение слишком близких Pu-O, O-O и Pu-Pu контактов;
- уменьшение координационных дефектов;
- поверхностная перестройка конечного кластера;
- частичное восстановление fluorite-like локального порядка;
- seeded local ordering в отдельном crystallization workflow.

Важно: модель не доказывает полную bulk-кристаллизацию PuO2. Для такого утверждения нужны более строгие условия: больший размер системы, PBC или carefully defined finite-size protocol, длительная динамика, сравнение с MD/DFT/экспериментом и устойчивые bulk order/RDF признаки. В этом проекте корректный вывод более узкий: энергетическая релаксация, дефектный отжиг и частичное локальное упорядочение.

## 5.2. Почему kMC, а не обычная MD

Молекулярная динамика интегрирует уравнения движения атомов с малым временным шагом. Это хорошо для колебаний и короткой динамики, но медленно для редких событий: перескоков, локальных перестроек, отжига дефектов и медленного упорядочения.

kMC работает иначе:

1. Не интегрирует все атомные колебания.
2. Генерирует набор возможных событий.
3. Оценивает rate каждого события.
4. Вероятностно выбирает событие пропорционально rate.
5. Увеличивает kMC-time на случайный интервал.

Физическая интерпретация: kMC приближенно моделирует последовательность редких активированных перестроек. В этом проекте события не являются строгими saddle-point transitions из NEB/ARTn. Это heuristic off-lattice events, ранжированные ML и проверяемые exact energy shortlist. Поэтому метод лучше называть ML-assisted off-lattice kMC relaxation/annealing, а не точным transition-state kMC.

## 5.3. Off-lattice события

В классическом lattice kMC атомы прыгают между заранее заданными узлами решетки. Здесь используется off-lattice подход: атомы двигаются в непрерывном 3D-пространстве, а не только по узлам.

Типы событий:

- `random_bulk`: случайное объемное смещение;
- `random_surface`: случайное поверхностное смещение;
- `surface`: поверхностно-смещенные события;
- `surface_compression`: мягкое сжатие поверхности внутрь;
- `coordination`: смещения, улучшающие локальную координацию;
- `relaxation`: смещения вдоль направления силы;
- `snap_to_fluorite_site`: движение к локально флюоритоподобной позиции;
- `growth_front`: события на границе ordered/disordered regions;
- `local_cluster_affine`: коллективная локальная перестройка небольшой группы атомов.

Обычный финальный расчет использует главным образом energy/relaxation-oriented события. Seeded crystallization workflow дополнительно включает order-biased события, которые помогают исследовать локальное флюоритоподобное упорядочение.

## 5.4. Энергия, барьер и rate

ML-регрессор предсказывает изменение потенциальной энергии:

```text
Delta E = E_after - E_before
```

Если `Delta E < 0`, событие энергетически выгодное. Если `Delta E > 0`, событие повышает энергию и должно иметь меньшую вероятность.

В kMC используется эффективный барьер:

```text
E_a = E0 + max(0, Delta E)
```

где:

- `E0` / `base_barrier` - базовый барьер события;
- `Delta E` - ML или exact оценка изменения энергии;
- `max(0, Delta E)` штрафует энергетически повышающие события.

Rate:

```text
rate = nu0 * exp(-E_a / (k_B * T))
```

где:

- `nu0` / `attempt_frequency` - эффективная частота попыток;
- `k_B` - постоянная Больцмана в eV/K;
- `T` - эффективная температура kMC.

В проекте `temperature = 3600 K` для основного расчета. Это не обязательно означает прямую MD-температуру реального эксперимента; здесь это effective kinetic temperature, управляющая вероятностью принятия энергетически дорогих событий.

## 5.5. Роль ML в физическом процессе

ML не заменяет полностью физику потенциала. Он нужен для ускорения:

- быстро оценить много candidate events;
- отсортировать события по полезности;
- выбрать top-ranked candidates;
- направить exact checks на самые важные события.

Физический контроль сохраняется через:

- exact recalculation для shortlist событий;
- rejection по exact energy guard;
- close-contact thresholds;
- диагностику min pair distances;
- validation на независимых событиях;
- uncertainty tracking.

Поэтому ML здесь является surrogate/ranking layer, а не самостоятельным физическим потенциалом.

## 5.6. Почему важны close-contact thresholds

Исходная структура содержит слишком близкие пары атомов. В парном потенциале такие контакты могут давать нефизичные энергии, большие силы и плохую динамику.

Используются pair-specific thresholds:

```text
Pu-O >= 1.9 A
O-O  >= 2.0 A
Pu-Pu >= 2.8 A
```

В начале `initial_close_contact_thresholds_satisfied = false`, поэтому перед kMC запускается repair stage. После repair и в финальном состоянии thresholds соблюдены:

```text
after_repair_close_contact_thresholds_satisfied = true
final_close_contact_thresholds_satisfied = true
```

Физический смысл: сначала убираются явно опасные геометрические артефакты, затем kMC работает уже в допустимой области конфигурационного пространства.

## 5.7. Fluorite PuO2 и координация

Идеальная флюоритная структура PuO2 имеет характерные локальные окружения:

- Pu окружен примерно 8 атомами O;
- O окружен примерно 4 атомами Pu.

Поэтому используются метрики:

- `fraction_pu_with_8_o`;
- `fraction_o_with_4_pu`;
- `fraction_pu_with_7_to_9_o_neighbors`;
- `fraction_o_with_3_to_5_pu_neighbors`;
- `mean_abs_coordination_error`.

`mean_abs_coordination_error` показывает среднее отклонение от идеальной координации. Уменьшение этой метрики означает, что локальные окружения становятся ближе к флюоритоподобным.

В финальном 5000-step расчете:

```text
mean_abs_coordination_error: 1.5333 -> 1.3000
fraction_pu_with_7_to_9_o:  0.2714 -> 0.3643
fraction_o_with_3_to_5_pu:  0.6500 -> 0.7500
```

Это хороший признак локального упорядочения, но не самодостаточное доказательство полной кристаллизации.

## 5.8. Order metrics

В проекте есть несколько order metrics:

```text
fluorite_order_score
bulk_fluorite_order_score
soft_coordination_order_score
```

Смысл:

- `fluorite_order_score`: строгая глобальная оценка флюоритоподобной координации;
- `bulk_fluorite_order_score`: оценка для более внутренней части кластера, менее чувствительная к поверхности;
- `soft_coordination_order_score`: мягкая оценка, допускающая неидеальные, но близкие окружения.

Почему нужен `bulk_fluorite_order_score`: у конечного кластера большая поверхность, а поверхность почти всегда имеет недокоординированные атомы. Если оценивать весь кластер слишком строго, surface defects могут скрыть реальное улучшение внутренней области.

Финальный 5000-step результат:

```text
fluorite_order_score:      0.5238 -> 0.6214
bulk_fluorite_order_score: 0.6426 -> 0.7914
soft_coordination_score:   0.7122 -> 0.7545
```

Физический вывод: bulk-like часть кластера становится более флюоритоподобной.

## 5.9. RDF и структурная интерпретация

RDF, radial distribution function, показывает распределение межатомных расстояний. В проекте строятся:

- `rdf_initial_final.png`;
- `rdf_pu_o_initial_final.png`;
- `rdf_pu_pu_initial_final.png`;
- `rdf_o_o_initial_final.png`.

Физически:

- более выраженные пики RDF обычно означают более упорядоченную структуру;
- широкий RDF соответствует разупорядоченному или аморфному состоянию;
- Pu-O RDF особенно важен для локальной координации PuO2;
- Pu-Pu и O-O RDF помогают увидеть дальний/средний порядок.

Но для конечного кластера RDF нужно интерпретировать осторожно:

- нет периодических границ;
- сильный surface contribution;
- конечный размер сглаживает пики;
- локальное улучшение координации может не дать идеального bulk RDF.

Поэтому RDF используется вместе с energy, coordination, order metrics и min-distance diagnostics.

## 5.10. Surface effects

PuO2 cluster имеет большую долю поверхностных атомов. Поверхность физически отличается от bulk:

- меньше соседей;
- больше координационных дефектов;
- выше подвижность;
- сильнее локальные релаксации;
- возможны surface compression/reconstruction.

Поэтому в analysis есть разделение:

```text
bulk_coordination_defects
surface_coordination_defects
```

Если surface defects растут, это не обязательно означает ухудшение bulk ordering. Поверхность может перестраиваться, пока внутренняя область становится более флюоритоподобной. Поэтому для выводов важно смотреть сразу несколько метрик.

## 5.11. Seeded crystallization

Seeded workflow нужен, чтобы проверить, может ли система поддержать и развить локальное флюоритоподобное ядро.

Схема:

1. Исходная структура.
2. Repair close contacts.
3. Наложение центрального fluorite-like seed.
4. Post-seed repair.
5. Stage 1: ordering с большим `lambda`.
6. Stage 2: annealing с меньшим `lambda`.
7. Оценка энергии, bulk order, coordination error, fractions Pu/O ideal coordination.

Order-biased score:

```text
S = Delta E - lambda * Delta Q
```

где:

- `Delta E` - изменение энергии;
- `Delta Q` - улучшение order metric;
- `lambda` - вес структурного порядка.

При `lambda > 0` kMC может предпочесть событие, которое не только снижает энергию, но и улучшает флюоритоподобный порядок. Чтобы это не стало нефизичным, используется exact energy guard:

```text
--max-exact-delta-e-above 0.2
```

Это ограничивает принятие событий, которые слишком повышают exact energy.

Интерпретация seeded result:

```text
local seeded ordering: yes
complete crystallization proof: no
```

То есть seeded workflow показывает, что локальное флюоритоподобное ядро и order-biased события могут улучшать bulk order и координации, но это не доказывает полную рекристаллизацию всего конечного кластера.

## 5.12. Ограничения физической модели

Главные ограничения:

1. Потенциал MOX-07 является парным и эмпирическим.
2. Нет явной электронной структуры, зарядового переноса и DFT-точности.
3. Нет periodic boundary conditions.
4. Система является конечным кластером с большой поверхностью.
5. kMC events эвристические, а не строго найденные saddle-point transitions.
6. Effective barrier model упрощен.
7. ML-регрессор аппроксимирует `Delta E`, а не является физическим потенциалом.
8. Полная кристаллизация требует более сильного доказательства, чем рост coordination/order metrics.

Зачем тогда метод полезен:

- быстро исследует большое число локальных перестроек;
- показывает направление энергетической релаксации;
- выявляет полезные события для отжига дефектов;
- позволяет сравнивать ML-ranking, exact shortlist и seeded ordering;
- дает воспроизводимый computational pipeline с тестами, validation и диагностикой.

## 6. Как работает ML-kMC

На каждом kMC-шаге:

1. Генерируется набор candidate events.
2. Для каждого события считаются локальные признаки.
3. ML-регрессор предсказывает `Delta E`.
4. ML-классификатор предсказывает вероятность полезного события.
5. По `Delta E` строится kMC rate:

```text
E_a = E0 + max(0, Delta E)
rate = nu0 * exp(-E_a / (k_B * T))
```

6. Классификатор мягко корректирует rate:

```text
factor = (1 - w) + w * P(good event)
```

7. Top-ranked и high-uncertainty события попадают в exact shortlist.
8. Для shortlisted событий exact `Delta E` пересчитывается потенциалом MOX-07.
9. Событие отклоняется, если оно нарушает close-contact thresholds или exact energy guard.
10. Принятое событие меняет координаты.

Close-contact thresholds:

```text
Pu-O >= 1.9 A
O-O  >= 2.0 A
Pu-Pu >= 2.8 A
```

## 7. Лучшая модель

Папка:

```text
results/best_model
```

Ключевые файлы:

```text
regressor.joblib
classifier.joblib
feature_columns.joblib
classifier_threshold.json
metrics.json
model_selection_report.json
regressor_comparison.json
classifier_comparison.json
feature_importance.csv
feature_importance.png
predicted_vs_exact_delta_E.png
```

Текущая выбранная модель:

```text
Delta E regressor: LightGBM ensemble wrapper
Good-event classifier: LightGBM classifier
Classifier threshold: 0.375
```

Независимая validation:

```text
sign_accuracy: 0.839
recall_energy_lowering: 0.862
R2: 0.7446
top_5_precision: 0.96
```

Что значит:

- `sign_accuracy`: как часто модель правильно предсказывает знак `Delta E`;
- `recall_energy_lowering`: сколько реально energy-lowering событий модель не пропускает;
- `R2`: качество регрессии `Delta E`;
- `top_5_precision`: насколько хороши топовые события, которые модель ставит наверх ranking.

## 8. Главный расчет на 5000 шагов

Папка:

```text
results/final_5000_steps
```

Команда, которой был получен результат:

```powershell
python -m src.main_run_kmc --xyz input/PuO2_324.xyz --model-dir results/best_model --steps 5000 --out-dir results/final_5000_steps_new --n-candidates-per-step 64 --exact-shortlist-size 8 --uncertainty-shortlist-size 4 --exact-check-interval 25 --reject-exact-delta-above 0 --pre-relaxation-steps 300 --save-xyz-interval 1000 --seed 20260527
```

Главные итоги:

| Metric | Value |
|---|---:|
| Steps | 5000 |
| Speed | 1.47 steps/s |
| Applied events | 4911 |
| Rejected selected events | 89 |
| Acceptance ratio | 0.9822 |
| Initial E/PuO2 | -46.6442 eV |
| After repair E/PuO2 | -46.7132 eV |
| Final E/PuO2 | -47.3166 eV |
| Delta repair/PuO2 | -0.0690 eV |
| Delta kMC/PuO2 | -0.6034 eV |
| Delta total/PuO2 | -0.6724 eV |
| Initial close-contact safe | false |
| After repair close-contact safe | true |
| Final close-contact safe | true |
| Final min Pu-O | 1.9003 A |
| Final min O-O | 2.0386 A |
| Final min Pu-Pu | 2.9156 A |

Структурные метрики:

| Metric | Initial | Final |
|---|---:|---:|
| fluorite_order_score | 0.5238 | 0.6214 |
| bulk_fluorite_order_score | 0.6426 | 0.7914 |
| soft_coordination_order_score | 0.7122 | 0.7545 |
| mean_abs_coordination_error | 1.5333 | 1.3000 |
| fraction Pu with 7-9 O neighbors | 0.2714 | 0.3643 |
| fraction O with 3-5 Pu neighbors | 0.6500 | 0.7500 |

Смысл:

- энергия заметно падает;
- repair убирает исходные близкие контакты;
- kMC дополнительно снижает энергию после repair;
- bulk fluorite-like order растет;
- средняя ошибка координации уменьшается;
- финальные close-contact thresholds соблюдены.

## 9. Какие файлы смотреть в final_5000_steps

```text
summary.json                         итоговые метрики
history.csv                          история всех kMC-шагов
final.xyz                            финальная структура
snapshots/                           промежуточные XYZ
pre_relaxation_history.csv           repair близких контактов
active_learning_events.csv           события, полезные для active learning
publication_figures/                 красивые графики
energy_per_puo2_vs_step.png          энергия на PuO2
min_pair_distances_vs_step.png       safety по расстояниям
crystallization_order_scores_vs_step.png order metrics
bulk_surface_defects_vs_step.png     дефекты bulk/surface
event_kind_counts.png                какие события выбирались
uncertainty_vs_step.png              uncertainty модели
rdf_initial_final.png                RDF до/после
initial_final_structure.png          3D структура до/после
```

## 10. Красивые графики

Папка:

```text
results/final_5000_steps/publication_figures
```

Файлы:

```text
main_result_summary.png
model_diagnostics.png
```

Для seeded crystallization:

```text
results/crystallization_seeded/publication_figures/seeded_crystallization_summary.png
```

Перегенерировать:

```powershell
python -m src.publication_plots results/final_5000_steps
python -m src.publication_plots results/crystallization_seeded --seeded
```

Эти графики лучше использовать в отчете: они сделаны в спокойном scientific style, с белым фоном, аккуратной сеткой, аннотациями и summary-блоками.

## 11. Validation

Папка:

```text
results/validation
```

Запуск короткой validation:

```powershell
powershell -ExecutionPolicy Bypass -File .\validate.ps1
```

Или вручную:

```powershell
python -m src.main_validate_ml --xyz input/PuO2_324.xyz --model-dir results/best_model --out-dir results/validation_smoke --n-events 100 --seed 9101
```

Главные файлы validation:

```text
validation_events.csv
validation_metrics.json
validation_error_summary.json
ranking_report.json
```

Что это значит:

- `validation_events.csv`: независимые события, exact `Delta E`, ML prediction, uncertainty;
- `validation_metrics.json`: MAE/RMSE/R2/sign accuracy/precision/recall;
- `validation_error_summary.json`: где модель ошибается сильнее;
- ranking report: насколько хорошо модель выбирает top-ranked события.

## 12. Короткий smoke run

Запуск:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

Или вручную:

```powershell
python -m src.main_run_kmc --xyz input/PuO2_324.xyz --model-dir results/best_model --steps 20 --out-dir results/smoke_manual --n-candidates-per-step 32 --exact-shortlist-size 8 --uncertainty-shortlist-size 4 --exact-check-interval 5 --reject-exact-delta-above 0 --pre-relaxation-steps 100 --save-xyz-interval 10 --seed 9100
```

Smoke run нужен не для физического вывода, а чтобы быстро проверить:

- модель загружается;
- kMC запускается;
- события выбираются;
- close-contact safety работает;
- `summary.json`, `history.csv`, `final.xyz` создаются.

## 13. Генерация mixed dataset

В пакете уже есть готовый датасет:

```text
results/training_events.csv
```

Пример генерации нового:

```powershell
python -m src.main_generate_dataset --xyz input/PuO2_324.xyz --xyz results/final_5000_steps/final.xyz --out results/training_events_new.csv --n-events 10000 --mixed --order-biased-events --seed 20260522
```

Что хранит dataset:

```text
delta_E
is_energy_lowering
atom_type
event_kind
delta_fluorite_order
delta_bulk_order
delta_coordination_error
is_crystallizing_event
ML feature columns
```

Зачем mixed dataset:

- модель видит не только начальную структуру;
- события берутся из разных состояний;
- есть обычные и order-biased events;
- качество лучше переносится на реальные kMC trajectories.

## 14. Обучение моделей

Переобучить модели:

```powershell
python -m src.main_train_ml --dataset results/training_events.csv --model-dir results/best_model_new --classifier-target is_energy_lowering
```

После обучения появятся:

```text
regressor.joblib
classifier.joblib
feature_columns.joblib
classifier_threshold.json
metrics.json
model_selection_report.json
regressor_comparison.json
classifier_comparison.json
feature_importance.csv
feature_importance.png
predicted_vs_exact_delta_E.png
```

Важно: если обучить новую модель, нужно заново прогнать validation и хотя бы smoke run.

## 15. Кристаллизация

Папка seeded workflow:

```text
results/crystallization_seeded
```

Код:

```text
src/main_run_seeded_crystallization.py
src/seeded.py
```

Что делает seeded workflow:

1. Берет исходную PuO2-структуру.
2. Делает repair close contacts.
3. Накладывает центральный fluorite-like seed.
4. Еще раз чинит близкие контакты после seed.
5. Запускает stage 1 с order bias и growth-front events.
6. Запускает stage 2 с меньшим order bias для annealing.
7. Сохраняет stage comparison и crystallization diagnostics.

Пример запуска:

```powershell
python -m src.main_run_seeded_crystallization --xyz input/PuO2_324.xyz --model-dir results/best_model --out-dir results/crystallization_seeded_new --steps-stage1 5000 --steps-stage2 5000 --lambda-stage1 4.0 --lambda-stage2 0.75 --temperature-stage1 3300 --temperature-stage2 2200 --seed-radius 5.0 --seed-blend 0.5 --pre-relaxation-steps 1000 --post-seed-repair-steps 1000 --n-candidates-per-step 128 --exact-shortlist-size 12 --max-exact-delta-e-above 0.2
```

Кристаллизационные механизмы:

```text
order-biased score: S = Delta E - lambda * Delta Q
events: snap_to_fluorite_site, growth_front, local_cluster_affine
metrics: bulk_fluorite_order_score, soft_coordination_order_score, crystalline_core_size, growth_front_size
```

Как интерпретировать:

- `bulk_fluorite_order_score` растет: больше bulk-like локального порядка;
- `mean_abs_coordination_error` падает: атомы ближе к идеальным координациям;
- `fraction_pu_with_8_o` и `fraction_o_with_4_pu` растут: локальные окружения становятся ближе к fluorite PuO2;
- это supplementary evidence для локального seeded ordering, не доказательство полной кристаллизации конечного кластера.

## 16. Baseline и дополнительные результаты

Дополнительные папки:

```text
results/baseline
results/production_old
results/short_ml_run
results/smoke_best_model
results/candidate_model
results/validation_candidate_model
```

Смысл:

- `baseline`: контроль без основной ML-ranking логики;
- `production_old`: старый production-like расчет, оставлен для истории сравнения;
- `short_ml_run`: короткий ML-kMC comparison;
- `smoke_best_model`: smoke текущей модели;
- `candidate_model`: active-learning candidate, не главный финальный выбор;
- `validation_candidate_model`: validation candidate model.

Для защиты/отчета главным считать `results/final_5000_steps`, остальные папки использовать как supplementary.

## 17. Оптимизация скорости

В проекте ускорены горячие места:

- `src/potentials.py`: локальная энергия и сила для trial-position без полной пересборки структуры;
- `src/features.py`: energy+force считаются вместе за один обход соседей;
- `src/kmc.py`: `cKDTree` и center переиспользуются при выборе события;
- exact checks ограничены shortlist-ом, а не выполняются для всех candidates.

Короткий benchmark после оптимизации:

```text
about 1.63 steps/s on 50 steps with 64 candidates
```

Финальный 5000-step run:

```text
1.47 steps/s
```

## 18. Частые команды

Проверить проект:

```powershell
python -m pytest -q
```

Запустить smoke:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

Запустить validation smoke:

```powershell
powershell -ExecutionPolicy Bypass -File .\validate.ps1
```

Сгенерировать красивые графики:

```powershell
python -m src.publication_plots results/final_5000_steps
python -m src.publication_plots results/crystallization_seeded --seeded
```

Повторить 5000-step расчет:

```powershell
python -m src.main_run_kmc --xyz input/PuO2_324.xyz --model-dir results/best_model --steps 5000 --out-dir results/final_5000_steps_new --n-candidates-per-step 64 --exact-shortlist-size 8 --uncertainty-shortlist-size 4 --exact-check-interval 25 --reject-exact-delta-above 0 --pre-relaxation-steps 300 --save-xyz-interval 1000 --seed 20260527
```

Переобучить модель:

```powershell
python -m src.main_train_ml --dataset results/training_events.csv --model-dir results/best_model_new --classifier-target is_energy_lowering
```

Запустить seeded crystallization:

```powershell
python -m src.main_run_seeded_crystallization --xyz input/PuO2_324.xyz --model-dir results/best_model --out-dir results/crystallization_seeded_new --steps-stage1 5000 --steps-stage2 5000 --lambda-stage1 4.0 --lambda-stage2 0.75 --temperature-stage1 3300 --temperature-stage2 2200 --seed-radius 5.0 --seed-blend 0.5 --pre-relaxation-steps 1000 --post-seed-repair-steps 1000 --n-candidates-per-step 128 --exact-shortlist-size 12 --max-exact-delta-e-above 0.2
```

## 19. Что писать в отчете

Рекомендуемая формулировка:

```text
В работе реализован ML-ускоренный off-lattice kMC pipeline для PuO2 на потенциале MOX-07. Модель обучена на mixed dataset из локальных событий и выбирается по kMC-oriented метрикам, включая sign accuracy, recall energy-lowering events и ranking quality. Финальный 5000-step расчет снижает энергию на -0.6724 eV/PuO2, сохраняет close-contact safety и улучшает bulk_fluorite_order_score с 0.6426 до 0.7914. Seeded workflow дополнительно демонстрирует возможность локального seeded fluorite-like ordering. Результаты следует интерпретировать как энергетическую релаксацию, отжиг дефектов и частичное локальное упорядочение, а не как полную рекристаллизацию конечного PuO2-кластера.
```
