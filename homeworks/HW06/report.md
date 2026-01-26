# HW06 – Report
## 1. Dataset
- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: 18000 строк, 37 признаков
- Целевая переменная: `target`
  - класс 0: 13273 (73.74%)
  - класс 1: 4727 (26.26%)
- Признаки: все числовые (при необходимости некоторые были приведены к числовому типу).

## 2. Protocol
- Разбиение: train/test = 0.75/0.25, random_state=42
- Подбор: GridSearchCV на train (см. search_summaries.json); CV-оценки использованы для выбора модели, тест использован один раз для финальной оценки.
- Метрики: accuracy, F1, ROC-AUC (для бинарных). Для CV использовался scoring='roc_auc'.

## 3. Models
- DummyClassifier (most_frequent) — baseline
- LogisticRegression (Pipeline: StandardScaler + LogisticRegression), C подобран через GridSearchCV
- DecisionTreeClassifier — подбор max_depth + min_samples_leaf через GridSearchCV
- RandomForestClassifier — подбор max_depth/max_features/min_samples_leaf через GridSearchCV
- GradientBoostingClassifier — подбор learning_rate/max_depth через GridSearchCV
- StackingClassifier (опционально) — обучён на train, оценён на test

## 4. Results
Таблица финальных метрик на test:

| model              |   accuracy |   roc_auc |         f1 |
|:-------------------|-----------:|----------:|-----------:|
| Dummy              |   0.737333 |  0.5      | nan        |
| LogisticRegression |   0.816222 |  0.80089  |   0.571724 |
| DecisionTree       |   0.826    |  0.82983  |   0.62446  |
| RandomForest       |   0.889333 |  0.926607 |   0.751992 |
| GradientBoosting   |   0.895778 |  0.922107 |   0.77846  |
| Stacking           |   0.906444 |  0.927999 |   0.810104 |

- Победитель (по CV, затем по тестовой метрике): **Stacking** 

## 5. Analysis
- Устойчивость: при желании запустите несколько прогонов с разными random_state (опционально, не включено в код по умолчанию).
- Ошибки: confusion matrix сохранена в artifacts/figures/confusion_matrix_best.png (если best_model поддерживает predict).
- Интерпретация: permutation importance сохранён в artifacts/feature_importance.csv и artifacts/figures/permutation_importance.png (если был успешно рассчитан).

## 6. Conclusion
- Ансамбли (RF/GB/Stacking) обычно дают лучший баланс bias/variance по сравнению с одиночными деревьями или линейными моделями на этих данных.
- Для честного эксперимента подбор гиперпараметров выполняйте только на train, используйте CV, и тест применяйте один раз для финальной оценки.
- Сохраняйте модели и метаданные (best_model.joblib, best_model_meta.json) для воспроизводимости.
- Для дисбалансных задач дополнительно смотрите PR-кривую и average_precision_score.
