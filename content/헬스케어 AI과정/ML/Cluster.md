---
title: 041 Cluster
tag:
  - 헬스케어 ai
  - ML
description: 260610 수업 내용 정리
---

# 거리를 기반으로 하는 Clustering

거리(Distance)는 데이터 포인트가 **얼마나 비슷한가**를 수치로 표현하는 방법이다.  
거리가 가까울수록 비슷한 데이터, 멀수록 다른 데이터다. KNN과 군집화(Clustering) 모두 이 거리 개념 위에 세워진다.

---

## 1. 선형대수의 거리 — 벡터 간 거리 측정

두 데이터 포인트는 수치 공간의 벡터로 표현된다. 이 벡터 사이의 거리를 측정하는 방법이 여러 가지 있다.

### 스케일링이 반드시 필요한 이유

거리 기반 알고리즘에서 스케일링은 선택이 아닌 **필수**다.

```
예: 고객 데이터
  나이:   25세  vs  30세  → 차이 = 5
  연봉: 3000만  vs 5000만 → 차이 = 2000만

  스케일링 없이 거리 계산:
  d = √(5² + 2000만²) ≈ 2000만  ← 연봉이 거리를 완전히 지배!
  → 나이 정보가 사실상 무시됨

  스케일링 후:
  나이:   (25-25)/(30-25) = 0  vs 1 → 차이 = 1
  연봉: (3000만-3000만)/(2000만) = 0 vs 1 → 차이 = 1
  → 두 특성이 동등하게 반영됨
```

### 거리 측정 방법들

#### 유클리디안 거리 (Euclidean Distance) — 가장 기본

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

직선 거리다. 2차원에서는 피타고라스 정리와 동일하다. 가장 많이 사용되지만 이상치에 민감하고, 고차원에서는 모든 점이 비슷하게 멀어지는 **차원의 저주** 문제가 있다.

#### 맨해튼 거리 (Manhattan Distance)

$$d(p, q) = \sum_{i=1}^{n}|p_i - q_i|$$

격자(grid)처럼 수직/수평으로만 이동하는 거리다. 이상치에 덜 민감하다. 뉴욕의 블록처럼 대각선으로 갈 수 없는 상황을 모델링한다.

#### 코사인 유사도 (Cosine Similarity)

$$\text{cosine}(p, q) = \frac{p \cdot q}{|p| |q|}$$

크기가 아닌 **방향의 유사도**를 측정한다. 텍스트 분류, 추천 시스템에서 많이 쓴다.  
"많이 언급되냐"보다 "어떤 주제를 다루냐"가 중요할 때 사용한다.

#### 민코프스키 거리 (Minkowski Distance)

$$d(p, q) = \left(\sum_{i=1}^{n}|p_i - q_i|^r\right)^{1/r}$$

유클리디안($r=2$)과 맨해튼($r=1$)을 일반화한 거리다.

```python
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine

p = np.array([1, 2, 3])
q = np.array([4, 6, 8])

print(f'유클리디안: {euclidean(p, q):.4f}')   # √(9+16+25) = 7.416
print(f'맨해튼:     {cityblock(p, q):.4f}')    # 3+4+5 = 12
print(f'코사인 거리: {cosine(p, q):.4f}')       # 1 - 코사인유사도

# sklearn에서 거리 측정
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
X = np.array([[1, 2], [3, 4], [5, 6]])
print(euclidean_distances(X))     # 모든 쌍의 유클리디안 거리 행렬
print(cosine_similarity(X))       # 모든 쌍의 코사인 유사도 행렬
```

---

## 2. Clustering vs KNN 분류 — 비교

같은 "거리"라는 척도를 사용하지만 목적이 완전히 다르다.

|구분|Clustering|KNN Classification|
|---|---|---|
|**학습 방법**|비지도학습 (레이블 없음)|지도학습 (레이블 있음)|
|**척도**|거리|거리|
|**목적**|특성 파악 (복잡 → 단순)|정확한 분류|
|**평가**|군집 간 거리, 혼잡성|정확도|
|**후작업**|통계적 특성 파악 (중심 분석)|없음|

```
Clustering: "비슷한 것끼리 묶어봐, 그 안에 어떤 패턴이 있는지 발견해봐"
KNN:        "이 새 데이터가 어느 클래스인지 주변 이웃을 보고 결정해"
```

---

## 3. KNN (K-Nearest Neighbors)

### Instance-Based Learning (사례 기반 학습)

KNN은 **모델을 따로 만들지 않는다**. 훈련 데이터를 그대로 메모리에 저장해두고, 새 데이터가 오면 그때 가장 가까운 K개 이웃을 찾아 예측한다. 이것을 **게으른 학습(Lazy Learning)** 이라고도 한다.

```
일반 모델 (Eager Learning):
  훈련 → 내부 파라미터 학습 → 저장
  예측 → 파라미터로 빠르게 계산

KNN (Lazy Learning):
  훈련 → 데이터를 그냥 저장 (학습 없음)
  예측 → 전체 훈련 데이터와 거리 계산 → 가장 가까운 K개 선택
         → 다수결(분류) 또는 평균(회귀)
```

### 이웃 검색 알고리즘

데이터가 많을 때 매번 전체와 거리를 계산하면 느리다. 이를 효율화하는 세 가지 방법이 있다.

|알고리즘|방법|적합한 경우|
|---|---|---|
|**brute**|전체와 모두 비교|소량 데이터 또는 고차원|
|**ball_tree**|구(ball) 모양 트리로 공간 분할|저차원, 유클리디안/맨해튼|
|**kd_tree**|축(axis)으로 공간을 반씩 분할|저차원(d < 20), 균일한 분포|

```
kd_tree 예시 (2D):
  전체 공간을 x축 기준으로 반 분할
    왼쪽을 y축 기준으로 반 분할
      ... 반복
  트리 구조로 저장 → 검색 시 무관한 영역 건너뜀 → O(log n)
```

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 데이터 생성 및 분할
X, y = make_classification(n_samples=500, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN은 반드시 스케일링 필요
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

knn = KNeighborsClassifier(
    n_neighbors=5,         # K 값
    metric='minkowski',    # 거리 측정 방법 (p=2 → 유클리디안)
    p=2,
    algorithm='auto',      # 'auto', 'ball_tree', 'kd_tree', 'brute'
    weights='distance',    # 'uniform': 모두 동등, 'distance': 가까울수록 더 중요
    n_jobs=-1              # 전체 코어 사용
)
knn.fit(X_train_sc, y_train)
print(classification_report(y_test, knn.predict(X_test_sc)))

# 최적 K 찾기 — 엘보우 방법
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_k, X_train_sc, y_train, cv=5)
    k_scores.append(scores.mean())

best_k = k_range[k_scores.index(max(k_scores))]
print(f'최적 K: {best_k}, CV 정확도: {max(k_scores):.4f}')

plt.plot(k_range, k_scores, 'b-o')
plt.axvline(x=best_k, color='red', linestyle='--')
plt.xlabel('K 값')
plt.ylabel('CV 정확도')
plt.title('K에 따른 분류 성능')
plt.show()
```

### KNN의 특징

- **이상치 처리에 강함**: 이상치는 가까운 이웃이 별로 없어서 자연스럽게 처리됨
- **비선형 데이터**: 결정 경계를 학습하지 않아 복잡한 경계도 표현 가능
- **작은 데이터셋에 적합**: 큰 데이터에서는 예측이 느림
- **해석 가능**: 예측 결과에 어떤 이웃이 영향을 줬는지 확인 가능

---

## 4. K-Means 클러스터링

### 핵심 아이디어

**중심기반 클러스터링**이다. K개의 중심점(Centroid)을 정하고, 각 데이터를 가장 가까운 중심에 배정하는 과정을 반복한다.

```
직관적 설명:
  "학생 100명을 성적 패턴이 비슷한 K개 그룹으로 나눠라"
  
  레이블 없이, 비슷한 패턴끼리 모아서 구조를 발견하는 것이 목표다.
```

### 알고리즘 단계

```
1. 초기화: K개의 중심을 무작위로 선택

2. 배정 (Assignment):
   모든 데이터 포인트를 가장 가까운 중심에 배정
   클러스터 ← argmin_k d(xᵢ, cₖ)

3. 재계산 (Update):
   각 클러스터의 새 중심 = 클러스터 내 데이터의 평균
   cₖ ← mean(클러스터 k에 속한 데이터들)

4. 수렴까지 2~3 반복
   (중심이 더 이상 움직이지 않거나 최대 반복 횟수 도달)
```

### WSS (Within-Cluster Sum of Squares) — 목적 함수

$$WSS = \sum_{k=1}^{K} \sum_{x_i \in C_k} (x_i - c_k)^2$$

각 데이터와 자신이 속한 클러스터 중심 사이의 거리 제곱합이다. K-Means는 이 값을 최소화하는 방향으로 학습한다.

```
WSS가 작다 = 클러스터 내부가 빽빽하게 모여 있다 = 좋은 군집
WSS가 크다 = 클러스터 내부가 흩어져 있다 = 나쁜 군집
```

### 초기화: K-Means++

무작위 초기화의 문제점은 초기 중심이 모두 같은 영역에 모여 있을 수 있다는 것이다. K-Means++는 **서로 멀리 떨어진 초기 중심**을 선택해서 수렴을 빠르게 한다.

```
K-Means++ 초기화:
  1. 첫 번째 중심: 무작위 선택
  2. 두 번째 중심: 기존 중심과 거리에 비례한 확률로 선택
     (멀수록 선택 확률 높음)
  3. 반복 → K개 중심 완성
```

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 군집화할 데이터 생성 (3개 그룹)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# 스케일링 (군집화에서도 스케일링 필요)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# K-Means 적합
km = KMeans(
    n_clusters=3,          # 군집 수
    init='k-means++',      # 초기화 방법 (기본값, 더 안정적)
    n_init=10,             # 다른 초기값으로 10번 반복 → 최선 선택
    max_iter=300,          # 최대 반복 횟수
    random_state=42
)
km.fit(X_sc)

labels = km.labels_         # 각 데이터의 클러스터 레이블
centers = km.cluster_centers_  # 클러스터 중심
wss = km.inertia_           # WSS (낮을수록 좋음)
print(f'WSS: {wss:.4f}')

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_sc[:, 0], X_sc[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30)
axes[0].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', zorder=5, label='중심')
axes[0].set_title('K-Means 군집 결과')
axes[0].legend()

# 정답 레이블과 비교
axes[1].scatter(X_sc[:, 0], X_sc[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=30)
axes[1].set_title('실제 군집 (참고용)')

plt.tight_layout()
plt.show()
```

### 최적 K 선택 — 엘보우 방법 (Elbow Method)

K가 커질수록 WSS는 계속 줄어든다. K=데이터 수이면 WSS=0이 되므로 단순히 WSS가 낮다고 좋은 것이 아니다. **WSS 감소 폭이 급격히 줄어드는 꺾이는 지점(엘보우)**이 최적 K다.

```
WSS
 │\
 │ \
 │  \
 │   \___
 │        \___________
 └──────────────────── K
       ↑
   여기가 엘보우 → 최적 K
```

```python
# 엘보우 방법으로 최적 K 탐색
wss_list = []
k_range = range(1, 11)

for k in k_range:
    km_k = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km_k.fit(X_sc)
    wss_list.append(km_k.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, wss_list, 'bo-')
plt.xlabel('K (군집 수)')
plt.ylabel('WSS (군집 내 분산)')
plt.title('엘보우 방법으로 최적 K 선택')
plt.xticks(k_range)
plt.grid(alpha=0.3)
plt.show()
# 그래프에서 꺾이는 지점의 K가 최적
```

### K-Means의 한계

```
1. 원형(구형) 군집만 잘 처리:
   길쭉한 타원형이나 복잡한 모양의 군집은 잘못 분류됨

2. 이상치에 민감:
   중심 = 평균이므로 이상치 하나가 중심을 크게 이동시킴

3. K를 미리 정해야 함:
   적합한 K를 모르면 엘보우 방법 등으로 탐색해야 함

4. 스케일 의존:
   스케일링 없으면 단위가 큰 특성이 군집을 지배
```

---

## 5. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### 핵심 아이디어

K-Means가 중심까지의 거리로 군집을 나눈다면, DBSCAN은 **데이터가 밀집된 영역**을 군집으로 인식한다. 밀집되지 않은 영역의 점은 **노이즈(이상치)** 로 처리한다.

```
K-Means: "중심에서 가까운 것끼리 묶는다"
DBSCAN:  "빽빽하게 모여 있는 곳이 군집이다"
```

### 두 가지 핵심 파라미터

- **Epsilon (ε)**: 이웃으로 간주하는 최대 거리 (반경)
- **min_samples**: Core Point가 되기 위한 최소 이웃 수

### 점의 세 가지 유형

```
Core Point (핵심 점):
  ε 반경 안에 min_samples 이상의 점이 있는 점
  → 군집의 중심이 되는 점

Border Point (경계 점):
  ε 반경 안에 min_samples 미만의 점이 있지만
  Core Point의 ε 반경 안에 포함되는 점
  → 군집의 가장자리

Outlier (노이즈, 이상치):
  어떤 Core Point의 ε 반경에도 포함되지 않는 점
  → 레이블 = -1 (군집에 속하지 않음)
```

```
시각화:

     ● ●
   ●[●]●      ← [●] = Core Point (ε 안에 이웃 많음)
     ●  ●
           ·  ← · = Border Point (Core Point ε 안에 있지만 자신은 Core 아님)
               ★  ← ★ = Outlier (어떤 Core Point의 ε 안에도 없음)
```

### 알고리즘 흐름

```
1. 임의의 방문하지 않은 점 선택

2. Core Point인지 확인:
   ε 반경 안 이웃 수 ≥ min_samples → Core Point
     → 새 군집 시작
     → 이 Core Point의 모든 이웃을 군집에 추가
     → 이웃 중 Core Point도 있으면 그 이웃들도 재귀적으로 추가

3. Core Point가 아니면:
   현재 군집의 Border Point이거나 Outlier

4. 모든 점을 방문할 때까지 반복
```

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# DBSCAN이 빛나는 데이터: 복잡한 모양 + 이상치
X_moons, _ = make_moons(n_samples=200, noise=0.1, random_state=42)

# 이상치 추가
outliers = np.random.uniform(-2, 3, (15, 2))
X_with_outliers = np.vstack([X_moons, outliers])

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_with_outliers)

# DBSCAN 적합
db = DBSCAN(
    eps=0.3,            # ε: 이웃 반경 (핵심 파라미터)
    min_samples=5,      # 최소 이웃 수 (핵심 파라미터)
    metric='euclidean'
)
labels = db.fit_predict(X_sc)

# 결과 분석
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)
print(f'발견된 군집 수: {n_clusters}')
print(f'이상치 수:      {n_noise}')

# Core, Border, Outlier 구분
core_mask    = np.zeros(len(X_sc), dtype=bool)
core_mask[db.core_sample_indices_] = True
border_mask  = ~core_mask & (labels != -1)
outlier_mask = (labels == -1)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means와 비교
km = KMeans(n_clusters=2, random_state=42)
km_labels = km.fit_predict(X_sc)
axes[0].scatter(X_sc[:, 0], X_sc[:, 1], c=km_labels, cmap='viridis', alpha=0.7, s=30)
axes[0].set_title('K-Means (초승달 모양 실패)')

# DBSCAN 결과
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
for k in range(n_clusters):
    mask = (labels == k) & core_mask
    axes[1].scatter(X_sc[mask, 0], X_sc[mask, 1], s=50, alpha=0.8, label=f'군집{k+1} Core')
    mask = (labels == k) & border_mask
    axes[1].scatter(X_sc[mask, 0], X_sc[mask, 1], s=30, alpha=0.5)

axes[1].scatter(X_sc[outlier_mask, 0], X_sc[outlier_mask, 1],
                c='red', marker='x', s=100, linewidths=2, label='이상치')
axes[1].set_title('DBSCAN (초승달 모양 성공 + 이상치 제거)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Epsilon과 min_samples 선택 방법

```python
# ε 선택: k-distance graph
# 각 포인트에서 k번째 이웃까지의 거리를 정렬했을 때 급격히 올라가는 지점
from sklearn.neighbors import NearestNeighbors

k = 5  # min_samples와 같게 설정
nbrs = NearestNeighbors(n_neighbors=k).fit(X_sc)
distances, _ = nbrs.kneighbors(X_sc)
distances = np.sort(distances[:, k-1])   # k번째 이웃까지의 거리 정렬

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('데이터 포인트 (거리 순 정렬)')
plt.ylabel(f'{k}번째 이웃까지의 거리')
plt.title('k-Distance Graph — 급격히 올라가는 지점이 ε')
plt.axhline(y=0.3, color='red', linestyle='--', label='선택한 ε=0.3')
plt.legend()
plt.show()
```

### DBSCAN의 특징

```
장점:
  - K(군집 수)를 미리 정하지 않아도 됨
  - 이상치를 자동으로 탐지 (레이블 = -1)
  - 원형이 아닌 복잡한 모양의 군집도 찾을 수 있음
  - 밀도가 다른 여러 군집도 처리 가능

단점:
  - ε과 min_samples를 잘 설정해야 함 (도메인 지식 필요)
  - 밀도가 다른 군집이 섞이면 약함 (HDBSCAN으로 해결 가능)
  - 고차원 데이터에서 ε 선택이 어려움
  - 구현이 K-Means보다 복잡
```

---

## 6. 군집 평가 — 실루엣 방법 (Silhouette Method)

레이블이 없는 비지도학습에서 "군집이 잘 됐는가"를 평가하는 방법이다.

### 실루엣 계수 (Silhouette Coefficient)

데이터 포인트 $i$에 대해:

- $a(i)$: 같은 클러스터 내 **다른 점들과의 평균 거리** (응집성, 작을수록 좋음)
- $b(i)$: 가장 가까운 **다른 클러스터 점들과의 평균 거리** (분리성, 클수록 좋음)

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

값의 범위: $-1$ ~ $+1$

```
s(i) = +1에 가까움: 자신의 클러스터에 잘 속해 있고, 다른 클러스터와 잘 분리됨
s(i) =  0에 가까움: 두 클러스터의 경계에 위치
s(i) = -1에 가까움: 잘못된 클러스터에 배정됨
```

**전체 평균 실루엣 계수**: 모든 데이터 포인트의 실루엣 계수 평균 → 전체 군집 품질

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 여러 K에 대한 실루엣 점수 비교
silhouette_scores = []
wss_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_sc)
    s_score = silhouette_score(X_sc, labels)
    silhouette_scores.append(s_score)
    wss_scores.append(km.inertia_)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(k_range, silhouette_scores, 'go-')
axes[0].set_xlabel('K')
axes[0].set_ylabel('실루엣 점수')
axes[0].set_title('실루엣 점수 (높을수록 좋음)')

axes[1].plot(k_range, wss_scores, 'bo-')
axes[1].set_xlabel('K')
axes[1].set_ylabel('WSS')
axes[1].set_title('엘보우 방법')

plt.tight_layout()
plt.show()

best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f'실루엣 기준 최적 K: {best_k}')
```

### 실루엣 플롯 — 군집별 품질 시각화

```python
def plot_silhouette(X, n_clusters):
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    cluster_labels = km.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_values  = silhouette_samples(X, cluster_labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10

    for k in range(n_clusters):
        k_values = np.sort(sample_values[cluster_labels == k])
        size_k = k_values.shape[0]
        y_upper = y_lower + size_k

        color = cm.nipy_spectral(float(k) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, k_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_k, str(k))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color='red', linestyle='--',
               label=f'전체 평균 = {silhouette_avg:.3f}')
    ax.set_xlabel('실루엣 계수')
    ax.set_ylabel('클러스터')
    ax.set_title(f'실루엣 플롯 (K={n_clusters})')
    ax.legend()
    plt.show()

plot_silhouette(X_sc, n_clusters=3)
# 각 군집의 막대가 빨간선(평균)보다 크고, 두께가 균일하면 좋은 군집
```

---

## 7. K-Means vs DBSCAN 비교 및 선택 가이드

|구분|K-Means|DBSCAN|
|---|---|---|
|**군집 수**|미리 지정 필요|자동 결정|
|**군집 모양**|원형/구형에 최적|임의의 모양|
|**이상치**|민감 (중심에 영향)|자동 탐지 (-1 레이블)|
|**속도**|빠름|보통|
|**파라미터**|K|ε, min_samples|
|**스케일 의존**|있음|있음|
|**빈 클러스터**|발생 가능|없음|
|**적합한 경우**|원형 군집, 빠른 분석|복잡한 형태, 이상치 탐지|

```python
# 세 가지 데이터 유형에서 K-Means vs DBSCAN 비교
from sklearn.datasets import make_blobs, make_moons, make_circles

datasets = [
    (make_blobs(n_samples=200, random_state=42)[0],     '구형 군집'),
    (make_moons(n_samples=200, noise=0.1, random_state=42)[0], '초승달'),
    (make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)[0], '동심원')
]

fig, axes = plt.subplots(3, 2, figsize=(12, 15))

for i, (X_d, title) in enumerate(datasets):
    X_d_sc = StandardScaler().fit_transform(X_d)

    # K-Means
    km_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_d_sc)
    axes[i, 0].scatter(X_d_sc[:, 0], X_d_sc[:, 1], c=km_labels, cmap='viridis', s=30)
    axes[i, 0].set_title(f'{title} — K-Means')

    # DBSCAN
    db_labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(X_d_sc)
    axes[i, 1].scatter(X_d_sc[:, 0], X_d_sc[:, 1], c=db_labels, cmap='viridis', s=30)
    n_noise = (db_labels == -1).sum()
    axes[i, 1].set_title(f'{title} — DBSCAN (이상치 {n_noise}개)')

plt.tight_layout()
plt.show()
# 구형: K-Means 성공 / 초승달·동심원: DBSCAN 성공
```

---

## 8. 계층적 군집화 (Hierarchical Clustering) — 연결법

K-Means와 DBSCAN 외에 **군집을 계층적으로 합쳐가는 방법**이다.  
K를 미리 지정하지 않아도 되고, 어떻게 합쳐지는지 **덴드로그램(Dendrogram)** 으로 시각화할 수 있다.

### 작동 방식 (Agglomerative — 병합 방식)

```
시작: 모든 데이터 포인트가 각자 하나의 군집

반복:
  가장 가까운 두 군집을 찾아 합침
  → 하나의 군집이 남을 때까지 반복

덴드로그램을 보고 적절한 높이에서 잘라 군집 수 결정
```

### 연결법 (Linkage) 4종 — 두 군집 사이의 거리를 어떻게 정의하느냐

아래 그림을 머릿속에 그리면서 이해하자.

```
군집 A: 점들 [1, 2, 3]
군집 B: 점들 [6, 7, 8]

            군집 A         군집 B
            1  2  3    |    6  7  8
                       |
```

#### 최단 연결법 (Single Linkage)

두 군집 사이의 거리 = **가장 가까운 두 점 사이의 거리**

```
군집 A와 B 사이 거리 = d(3, 6)  ← A에서 가장 오른쪽, B에서 가장 왼쪽

장점: 길쭉하거나 연결된 형태의 군집도 잘 잡음
단점: 체인 효과(Chaining Effect) — 두 군집이 하나의 점으로만 연결돼도
      합쳐버려 군집이 길게 늘어지는 현상
```

#### 최장 연결법 (Complete Linkage)

두 군집 사이의 거리 = **가장 먼 두 점 사이의 거리**

```
군집 A와 B 사이 거리 = d(1, 8)  ← A에서 가장 왼쪽, B에서 가장 오른쪽

장점: 비슷한 크기의 컴팩트한 군집 생성, 체인 효과 방지
단점: 이상치에 민감 — 이상치 하나가 거리를 크게 늘림
```

#### 평균 연결법 (Average Linkage)

두 군집 사이의 거리 = **모든 쌍의 거리 평균**

$$d(A, B) = \frac{1}{|A| \cdot |B|} \sum_{a \in A} \sum_{b \in B} d(a, b)$$

```
군집 A = [1, 2, 3], 군집 B = [6, 7, 8]일 때:
  d(1,6), d(1,7), d(1,8)
  d(2,6), d(2,7), d(2,8)
  d(3,6), d(3,7), d(3,8)
  총 9개 쌍의 거리 평균

장점: 최단/최장의 극단값을 피하는 절충안
단점: 계산량이 많음
```

#### 와드법 (Ward's Method)

두 군집 사이의 거리 = **합칠 때 증가하는 WSS(클러스터 내 분산의 제곱합) 증가량**

$$\Delta(A, B) = WSS(A \cup B) - WSS(A) - WSS(B)$$

```
다른 연결법과 달리 "두 점 사이의 거리"가 아니라
"합쳤을 때 전체 분산이 얼마나 늘어나는가"를 거리로 사용

모든 점을 고려해서 분산이 가장 적게 증가하는 방향으로 합침

장점: 가장 컴팩트하고 균형 잡힌 군집 생성 (가장 많이 사용)
단점: 원형 군집 가정, 이상치에 민감
```

### 4가지 연결법 비교

|방법|거리 기준|특징|권장 상황|
|---|---|---|---|
|**최단 (Single)**|가장 가까운 두 점|체인 효과 있음|연결된 형태의 군집|
|**최장 (Complete)**|가장 먼 두 점|컴팩트한 군집|이상치 없는 구형 군집|
|**평균 (Average)**|모든 쌍의 평균|최단/최장 절충|일반적인 경우|
|**와드 (Ward)**|분산 증가량|가장 균형 잡힘|**대부분의 경우 권장**|

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 데이터 생성
X, y_true = make_blobs(n_samples=50, centers=3, cluster_std=0.8, random_state=42)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# ── 덴드로그램 4종 비교 ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
methods = ['single', 'complete', 'average', 'ward']
titles  = ['최단 연결법 (Single)', '최장 연결법 (Complete)',
           '평균 연결법 (Average)', '와드법 (Ward)']

for ax, method, title in zip(axes.ravel(), methods, titles):
    Z = linkage(X_sc, method=method)  # 연결 행렬 계산
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=20,
               leaf_rotation=45, leaf_font_size=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('샘플 인덱스')
    ax.set_ylabel('거리')
    # 수평선: 이 높이에서 자르면 몇 개 군집인지 결정
    ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='절단선')
    ax.legend()

plt.suptitle('4가지 연결법 덴드로그램 비교', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

### 덴드로그램 읽는 법

```
높이
  │
10│           ┌──────────────────┐
  │           │                  │
 7│      ┌───┘         ┌────────┘
  │      │             │
 5│  ┌───┘      ┌──────┘
  │  │          │
 2│  ├─┐      ┌─┤
  │  │ │      │ │
 0│  A B      C D  ...
  └──────────────────── 데이터 포인트

읽는 법:
  - A와 B가 가장 먼저 합쳐짐 (높이=2: 가장 비슷)
  - 그 다음 C와 D가 합쳐짐 (높이=2)
  - AB 군집과 CD 군집이 합쳐짐 (높이=5)
  - 최종적으로 모두 합쳐짐 (높이=10)

절단선을 어디서 그느냐에 따라 군집 수가 결정됨:
  높이=6에서 자르면 → 2개 군집
  높이=3에서 자르면 → 4개 군집
```

### sklearn으로 군집화

```python
# 와드법으로 3개 군집
agg = AgglomerativeClustering(
    n_clusters=3,          # 군집 수 (덴드로그램 보고 결정)
    linkage='ward',        # 'single', 'complete', 'average', 'ward'
    metric='euclidean'     # 와드법은 유클리디안만 지원
)
labels_agg = agg.fit_predict(X_sc)

# 시각화
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, method, title in zip(axes[:3], ['single', 'complete', 'ward'],
                              ['최단', '최장', '와드']):
    agg_m = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels_m = agg_m.fit_predict(X_sc)
    ax.scatter(X_sc[:, 0], X_sc[:, 1], c=labels_m, cmap='tab10', s=40, alpha=0.8)
    ax.set_title(f'{title} 연결법')

axes[3].scatter(X_sc[:, 0], X_sc[:, 1], c=y_true, cmap='tab10', s=40, alpha=0.8)
axes[3].set_title('실제 군집')
plt.tight_layout()
plt.show()

# 실루엣 점수 비교
from sklearn.metrics import silhouette_score
for method in ['single', 'complete', 'average', 'ward']:
    agg_m = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels_m = agg_m.fit_predict(X_sc)
    s = silhouette_score(X_sc, labels_m)
    print(f'{method:10s}: 실루엣 = {s:.4f}')
```

### scipy로 군집 수 자동 결정

```python
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# 와드법으로 연결 행렬 계산
Z = linkage(X_sc, method='ward')

# 방법 1: 군집 수 직접 지정
labels_3 = fcluster(Z, t=3, criterion='maxclust')  # 3개 군집

# 방법 2: 거리(높이) 기준으로 자르기
labels_dist = fcluster(Z, t=3.0, criterion='distance')  # 높이 3에서 절단

# 덴드로그램에서 절단 높이 시각화
plt.figure(figsize=(10, 5))
Z_plot = linkage(X_sc, method='ward')
dendrogram(Z_plot)
plt.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='절단 높이=3.0')
plt.xlabel('데이터 인덱스')
plt.ylabel('거리 (높이)')
plt.title('와드법 덴드로그램 — 절단선 설정')
plt.legend()
plt.show()
```

### 계층적 군집화 vs K-Means vs DBSCAN

|구분|K-Means|계층적 (Ward)|DBSCAN|
|---|---|---|---|
|군집 수|사전 지정|사전 지정 또는 덴드로그램으로 결정|자동 결정|
|군집 모양|원형|원형 (Ward)|임의 형태|
|이상치 처리|민감|민감|자동 제거|
|시각화|산점도|덴드로그램 (구조 파악 용이)|산점도|
|대용량 데이터|빠름|느림 ($O(n^2)$~$O(n^3)$)|보통|
|직관적 해석|중간|높음 (트리 구조)|높음|
|주요 파라미터|K|K, 연결법|ε, min_samples|

군집화 자체가 목적이 아니라 **군집 내 데이터의 특성을 파악하는 것이 진짜 목적**이다.

```python
import pandas as pd

# 군집 결과를 데이터프레임에 추가
df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
df['cluster'] = km.labels_

# 군집별 통계 (중심 분석)
print(df.groupby('cluster').agg(['mean', 'std', 'count']))

# 군집별 분포 시각화
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(['feature_1', 'feature_2', 'feature_3', 'feature_4']):
    for k in range(3):
        axes[i].hist(df[df['cluster']==k][col], bins=15, alpha=0.5, label=f'군집 {k}')
    axes[i].set_title(col)
    axes[i].legend()
plt.tight_layout()
plt.show()
```

---

## 9. 전체 흐름 정리

```
데이터
   ↓
스케일링 (필수 — 거리 기반이므로)
   ↓
군집화 알고리즘 선택
   ├─ K가 알려진 경우 / 원형 군집         → K-Means
   │   초기화: K-Means++
   │   최적 K: 엘보우(WSS) + 실루엣 점수
   │
   ├─ 군집 구조를 탐색하고 싶은 경우      → 계층적 군집화 (Hierarchical)
   │   연결법 선택: Single/Complete/Average/Ward
   │   덴드로그램으로 군집 수 시각적 결정
   │   대부분의 경우 와드법 권장
   │
   └─ K 모름 / 복잡한 형태 / 이상치      → DBSCAN
       파라미터: ε (k-distance graph), min_samples
       이상치: 레이블 = -1

평가
   ├─ WSS (낮을수록 클러스터 내부 응집)
   ├─ 실루엣 점수 (-1~1, 1에 가까울수록)
   └─ 실루엣 플롯 (클러스터별 품질 시각화)

후작업
   └─ 군집별 통계 분석 (평균, 표준편차)
       → 각 군집의 특성 파악 → 인사이트 도출
```