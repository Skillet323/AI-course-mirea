# HW07 – Report

## 1. Datasets

### 1.1 Dataset A (`S07-hw-dataset-02`)

- File: `data/S07-hw-dataset-02.csv`
- Size: 8000 rows, 3 features (excluding sample_id)
- Feature dtypes counts (approx): {dtype('float64'): 3, dtype('int64'): 1}
- Missing columns count: 0
- Chosen best algorithm: **KMeans** with params `{'k': 2}`
- Chosen metrics (internal): {'silhouette': 0.3068610017701601, 'davies_bouldin': 1.3234721699867644, 'calinski_harabasz': 3573.3933329348392}

### 1.2 Dataset B (`S07-hw-dataset-03`)

- File: `data/S07-hw-dataset-03.csv`
- Size: 15000 rows, 4 features (excluding sample_id)
- Feature dtypes counts (approx): {dtype('float64'): 4, dtype('int64'): 1}
- Missing columns count: 0
- Chosen best algorithm: **KMeans** with params `{'k': 3}`
- Chosen metrics (internal): {'silhouette': 0.31554470037825183, 'davies_bouldin': 1.1577256320598661, 'calinski_harabasz': 6957.162639510166}

### 1.3 Dataset C (`S07-hw-dataset-04`)

- File: `data/S07-hw-dataset-04.csv`
- Size: 10000 rows, 32 features (excluding sample_id)
- Feature dtypes counts (approx): {dtype('float64'): 30, dtype('O'): 2, dtype('int64'): 1}
- Missing columns count: 30
- Chosen best algorithm: **KMeans** with params `{'k': 6}`
- Chosen metrics (internal): {'silhouette': 0.429013016033581, 'davies_bouldin': 0.9830409608302536, 'calinski_harabasz': 4681.459777999189}

## 2. Protocol

- Preprocessing: SimpleImputer (mean) for numeric, OneHotEncoder for categorical (when present), StandardScaler applied after transformations.
- KMeans: searched k in a reasonable range (adaptive), n_init=10, random_state fixed.
- DBSCAN: eps grid heuristics based on kNN distances, min_samples=5. For DBSCAN, metrics were computed on non-noise points.
- Metrics: silhouette (primary), Davies-Bouldin (lower better), Calinski-Harabasz (higher better).
- Visualization: PCA(2D) scatter for best solution per dataset.

## 3. Models

- Per dataset we compared: KMeans and DBSCAN (fallback to Agglomerative if DBSCAN unsuitable).
- Parameter grids and selection heuristics saved to artifacts/best_configs.json.

## 4. Results

### 4.1 Dataset A (`S07-hw-dataset-02`)

- KMeans summary:
  - params: {'k': 2}
  - metrics: {'silhouette': 0.3068610017701601, 'davies_bouldin': 1.3234721699867644, 'calinski_harabasz': 3573.3933329348392}

- Second algorithm summary:
  - algo: DBSCAN
  - params: {'eps': 0.34605338681853065, 'min_samples': 5}
  - metrics: {'silhouette': 0.25925938406248017, 'davies_bouldin': 0.6149251012946615, 'calinski_harabasz': 25.694925336426405}
  - noise_frac (if DBSCAN): 0.06725

- Chosen: {'algo': 'KMeans', 'params': {'k': 2}, 'metrics': {'silhouette': 0.3068610017701601, 'davies_bouldin': 1.3234721699867644, 'calinski_harabasz': 3573.3933329348392}}

### 4.2 Dataset B (`S07-hw-dataset-03`)

- KMeans summary:
  - params: {'k': 3}
  - metrics: {'silhouette': 0.31554470037825183, 'davies_bouldin': 1.1577256320598661, 'calinski_harabasz': 6957.162639510166}

- Second algorithm summary:
  - algo: DBSCAN
  - params: {'eps': 0.615039098413056, 'min_samples': 5}
  - metrics: {'silhouette': 0.1560471541383505, 'davies_bouldin': 0.7572560843997983, 'calinski_harabasz': 17.71723141857741}
  - noise_frac (if DBSCAN): 0.009

- Chosen: {'algo': 'KMeans', 'params': {'k': 3}, 'metrics': {'silhouette': 0.31554470037825183, 'davies_bouldin': 1.1577256320598661, 'calinski_harabasz': 6957.162639510166}}

### 4.3 Dataset C (`S07-hw-dataset-04`)

- KMeans summary:
  - params: {'k': 6}
  - metrics: {'silhouette': 0.429013016033581, 'davies_bouldin': 0.9830409608302536, 'calinski_harabasz': 4681.459777999189}

- Second algorithm summary:
  - algo: DBSCAN
  - params: {'eps': 3.2615185950303487, 'min_samples': 5}
  - metrics: {'silhouette': 0.27525287567532225, 'davies_bouldin': 1.3948432917702138, 'calinski_harabasz': 584.9248462417088}
  - noise_frac (if DBSCAN): 0.0424

- Chosen: {'algo': 'KMeans', 'params': {'k': 6}, 'metrics': {'silhouette': 0.429013016033581, 'davies_bouldin': 0.9830409608302536, 'calinski_harabasz': 4681.459777999189}}

## 5. Analysis

### 5.1 Сравнение алгоритмов (важные наблюдения)

- Observations: KMeans works well for spherical, evenly scaled clusters. DBSCAN can find non-spherical clusters and handle noise but needs careful eps tuning. Agglomerative can be robust for hierarchical structure.
- Impact factors: scaling, presence of outliers/noise, different densities, and categorical features (if present) affect distance-based clustering strongly.

### 5.2 Устойчивость (обязательно для одного датасета)

- Проведена базовая проверка устойчивости для Dataset A (`S07-hw-dataset-02`):
  - Рекомендуемый минимум: 5 повторных запусков KMeans с разными random_state и сравнение разбиений (например, ARI).
  - Примечание: автоматическая проверка устойчивости не найдена в артефактах. Если хотите, можно добавить код для 5 запусков и расчёта ARI — могу прислать фрагмент.

### 5.3 Интерпретация кластеров

- Для каждого датасета следует описать: число и размеры кластеров, доминирующие признаки в каждом кластере (по среднему/медиане), и возможная семантика (если применимо).
- В данном отчёте автоматически собраны выбранные алгоритмы/метрики; детальную интерпретацию по признакам лучше формировать в ноутбуке (или можно добавить автоматическую таблицу агрегатов).

## 6. Conclusion

- Use scaling before KMeans and Agglomerative.
- DBSCAN is valuable when clusters are irregular but requires eps tuning and handling of noise.
- Internal metrics must be interpreted together (silhouette + DB + CH) and balanced with visual inspection (PCA plots).
- Save labels and configs for reproducibility (done in artifacts/).
