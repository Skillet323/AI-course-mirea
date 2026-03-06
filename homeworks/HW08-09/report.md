# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано

- Датасет: **CIFAR10** (выбран для экспериментов по сравнению методов регуляризации и оптимизации).
- В части A (регуляризация) сравнивалось: E1 (base), E2 (Dropout), E3 (BatchNorm), E4 (лучший из E2/E3 + EarlyStopping).
- В части B (оптимизация) делались диагностики LR и сравнение оптимизаторов: O1 (LR слишком большой), O2 (LR слишком маленький), O3 (SGD+momentum + weight decay).

## 2. Среда и воспроизводимость

- Python: 3.12.3
- torch: 2.10.0+cu130
- torchvision: 0.25.0+cu130
- Устройство (на момент запуска ноутбука): cuda
- Seed: 42
- Как запустить: открыть `homeworks/HW08-09/HW08-09.ipynb` и выполнить Run All.

## 3. Данные

- Датасет: **CIFAR10**
- Разделение: train/val/test — train разбит 80/20 (воспроизводимо с seed=42); test используется из torchvision.
- Трансформации: `ToTensor()` + `Normalize(...)` (см. ноутбук).
- Комментарий: CIFAR10 — 10 классов, цветные 32x32 изображения; MLP подходит плохо для изображений, поэтому ожидаемо невысокая точность, но задача — сравнения методов.

## 4. Базовая модель и обучение

- Модель MLP: скрытые слои = [512, 256, 128], активация ReLU, опциональные Dropout/BatchNorm.
- Loss: CrossEntropyLoss
- Базовый Optimizer (часть A): Adam (lr=0.001)
- Batch size: 128
- Epochs (макс): 10
- EarlyStopping: patience = 4 (метрика — val_accuracy)

## 5. Часть A (S08): регуляризация (E1-E4)

- E1 (base): MLP [512, 256, 128], без Dropout/BatchNorm
- E2 (Dropout): как E1 + Dropout(p=0.3)
- E3 (BatchNorm): как E1 + BatchNorm
- E4 (EarlyStopping): лучший из (E2/E3) с EarlyStopping (patience=4); E4 сохранён как best_model.pt

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1-O3)

- O1: Adam, lr=1e-1 (слишком большой) — короткий прогон val_acc=0.5059
- O2: Adam, lr=1e-5 (слишком маленький) — короткий прогон val_acc=0.4745
- O3: SGD + momentum=0.9 + weight_decay=1e-4, lr=1e-2 — прогон 12 эпох (см. runs.csv)

## 7. Результаты

Файлы:
- `./homeworks/HW08-09/artifacts/runs.csv`
- `./homeworks/HW08-09/artifacts/best_model.pt`
- `./homeworks/HW08-09/artifacts/best_config.json`
- `./homeworks/HW08-09/artifacts/figures/curves_best.png`
- `./homeworks/HW08-09/artifacts/figures/curves_lr_extremes.png`

Короткая сводка:
- Лучший эксперимент части A: **E4**
- Лучшая val_accuracy: **0.5548**
- Итоговая test_accuracy (для лучшей модели): **0.5419**
- O1 (large LR) val_acc: 0.5059
- O2 (small LR) val_acc: 0.4745
- O3 (SGD+momentum+wd) val_acc: 0.5415

## 8. Анализ

- По runs.csv лучший по val_accuracy оказался эксперимент **E4** (val_accuracy=0.5548). В части A заметно, что BatchNorm/Dropout влияют как регуляризаторы: в наших прогонах BatchNorm (E3) показал лучший результат среди базовых вариантов. Эксперимент E4 использует ту же архитектуру, но с EarlyStopping, что позволило сохранить лучшую модель и получить максимальную val_accuracy.
- EarlyStopping настроен с patience=4; в данном прогоне E4 обучался до максимального числа эпох (10), EarlyStopping не сработал, так как метрика продолжала улучшаться.
- O1 (слишком большой lr) показал заметные колебания/плохой прогресс и пониженное val_accuracy; O2 (слишком маленький lr) почти не обучался (малая скорость уменьшения loss). O3 (SGD+momentum+wd) показал поведение, близкое к Adam, но с чуть иным сходством по кривым (см. curves).
- Weight decay в O3 привёл к небольшому регуляризующему эффекту (см. val_loss/val_acc в runs.csv).
- Для CIFAR10 MLP ограничен архитектурно; результаты полезны для сравнения техник, но не для SOTA.

## 9. Итоговый вывод

- В качестве базового конфига для дальнейших экспериментов возьмём E4 (модель [512, 256, 128], batchnorm=True, dropout=0.0, optimizer=Adam, lr=0.001).
- Что пробовать дальше: 1) сверточную архитектуру вместо MLP; 2) подбор lr через lr-scheduler или lr-range test.

## 10. Приложение

- Доп. сравнения и графики лежат в `homeworks/HW08-09/artifacts/figures/`.

