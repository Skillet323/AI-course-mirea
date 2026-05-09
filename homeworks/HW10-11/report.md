# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Часть A выполнена на датасете `Flowers102`, потому что это fine-grained задача и на ней хорошо видно преимущество transfer learning.
- Часть B выполнена на `Pascal VOC` в треке `detection`, потому что этот датасет подходит для демонстрации bounding-box детекции и базовой метрики качества.
- В части A сравнивались C1–C4: простая CNN без аугментаций, та же CNN с аугментациями, ResNet18 head-only и ResNet18 partial fine-tune.
- Во второй части сравнивались два порога уверенности: `V1 = 0.3` и `V2 = 0.7`.

## 2. Среда и воспроизводимость

- Python: версия среды, в которой выполнен ноутбук
- torch / torchvision: версии, установленные в среде
- Устройство (CPU/GPU): определяется автоматически в ноутбуке
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `Flowers102`
- Разделение: train / val / test через официальные split'ы torchvision
- Базовые transforms: Resize((224, 224)), ToTensor(), Normalize(ImageNet mean/std)
- Augmentation transforms: RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAffine
- Комментарий: `Flowers102` содержит 102 класса и требует fine-grained различения. Для такой задачи простая CNN с нуля обычно уступает pretrained моделям, а аугментации помогают лишь частично. Официальные split'ы удобны тем, что не требуют ручного разбиения и дают корректную валидацию.

### 3.2. Часть B: structured vision

- Датасет: `Pascal VOC`
- Трек: `detection`
- Что считается ground truth: bounding boxes из XML-разметки Pascal VOC
- Какие предсказания использовались: bounding boxes, class labels и confidence score от pretrained Faster R-CNN
- Комментарий: Pascal VOC — стандартный датасет для object detection. Использование pretrained модели разумно, потому что задача здесь состоит в корректной визуализации предсказаний, сопоставлении prediction ↔ ground truth и анализе влияния score threshold.

## 4. Часть A: модели и обучение (C1-C4)

- C1 (simple-cnn-base): простая CNN без аугментаций.
- C2 (simple-cnn-aug): та же CNN, но с аугментациями.
- C3 (resnet18-head-only): pretrained ResNet18, backbone заморожен, обучается только классификационная голова.
- C4 (resnet18-finetune): pretrained ResNet18, разморожены `layer4` и `fc`.

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer: Adam
- Batch size: 32
- Epochs (max): 10
- Критерий выбора лучшей модели: best validation accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Detection track

- Модель: FasterRCNN_ResNet50_FPN pretrained
- V1: `score_threshold = 0.3`
- V2: `score_threshold = 0.7`
- Как считался IoU: IoU между предсказанным bbox и GT bbox того же класса, затем greedy matching с порогом `IoU >= 0.5`
- Как считались precision / recall: `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализация второй части: `./artifacts/figures/detection_examples.png`
- Метрики второй части: `./artifacts/figures/detection_metrics.png`

Короткая сводка:

- Лучший эксперимент части A: C4
- Лучшая `val_accuracy`: 0.8598
- Итоговая `test_accuracy` лучшего классификатора: 0.8429
- Что дали аугментации (C2 vs C1): сравнение видно по `classification_compare.png`; аугментации могут дать небольшой прирост, но не всегда решают проблему обучения с нуля
- Что дал transfer learning (C3/C4 vs C1/C2): pretrained ResNet18 обычно даёт сильный выигрыш по сравнению с CNN, обученной с нуля
- Что оказалось лучше: head-only или partial fine-tuning: это определяется по `runs.csv`, в текущем прогоне лучший вариант — C4
- Что показал режим V1 во второй части: больше предсказаний, выше recall, ниже precision
- Что показал режим V2 во второй части: меньше предсказаний, выше precision, ниже recall
- Как интерпретируются метрики второй части: они показывают trade-off между полнотой и точностью обнаружения объектов

## 7. Анализ

SimpleCNN без transfer learning обычно сильно уступает pretrained ResNet18 на fine-grained задаче Flowers102, потому что обучается с нуля на сравнительно небольшом наборе данных. Аугментации помогают повысить устойчивость, но сами по себе не заменяют хорошие признаки. Head-only fine-tuning даёт быстрый и стабильный baseline, а partial fine-tuning обычно позволяет дополнительно адаптировать верхние слои под новый домен. Во второй части изменение `score_threshold` напрямую меняет баланс precision и recall: при низком пороге модель находит больше объектов, но делает больше ложных срабатываний. При высоком пороге число ложных срабатываний снижается, но часть объектов теряется. Для detection метрика IoU важна, потому что она отражает не только факт наличия бокса, но и качество его локализации. Потенциальные утечки данных в классификации были бы связаны с неправильным использованием train/val/test или с обучением scaler/augmentation-пайплайна на тесте. Здесь этого избегали за счёт официальных split'ов и раздельного использования test только после выбора лучшей модели по validation. Для detection-части важно явно описывать, как сопоставляются predicted boxes и ground truth, иначе precision и recall можно легко исказить. Типичные ошибки детектора — дублирование боксов, пропуск мелких объектов и ложные срабатывания на похожие классы.

## 8. Итоговый вывод

В качестве базового классификатора для такой задачи я бы взял pretrained ResNet18 с partial fine-tuning, потому что он сочетает качество и умеренную сложность обучения. Простая CNN полезна как контрольный baseline, но на Flowers102 обычно слабее. В detection-задачах ключевое значение имеют корректная визуализация, IoU и выбор порога уверенности в зависимости от баланса precision/recall. Главное, что здесь важно, — не использовать test как часть подбора модели и не смешивать его с validation.
