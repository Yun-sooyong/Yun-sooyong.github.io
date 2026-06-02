---
title: 036 scikit-learn
tag:
  - 국비교육
  - ML
description: 260602 수업 내용 정리
---

# scikit-learn 전체 구조와 ML 파이프라인

scikit-learn은 Python에서 가장 널리 쓰이는 머신러닝 라이브러리다.  
NumPy/SciPy 기반으로 만들어졌으며, **일관된 인터페이스**로 다양한 ML 알고리즘을 제공한다.

```python
import sklearn
print(sklearn.__version__)   # 버전 확인
```

---

## 1. ML 학습 방법의 분류

### 지도학습 (Supervised Learning)

**정답(레이블)이 있는 데이터**로 학습한다. 입력 $X$와 출력 $y$의 관계를 학습해서 새 데이터의 $y$를 예측한다.

- **회귀 (Regression)**: $y$가 연속형 수치. "집값 예측", "기온 예측"
- **분류 (Classification)**: $y$가 범주형. "스팸/정상 메일", "고양이/개 구분"

```
학습 데이터: (X₁, y₁), (X₂, y₂), ..., (Xₙ, yₙ)
목표: f(X) ≈ y 인 함수 f를 찾는다
```

### 비지도학습 (Unsupervised Learning)

**정답 없이** 데이터의 구조와 패턴을 스스로 발견한다.

- **군집화 (Clustering)**: 비슷한 데이터끼리 묶기. "고객 세분화"
- **차원 축소 (Dimensionality Reduction)**: 변수를 줄이면서 핵심 정보 유지. "PCA"
- **밀도 추정**: 데이터의 분포를 학습. "이상치 탐지"

### 준지도학습 (Semi-supervised Learning)

소량의 레이블 데이터와 대량의 레이블 없는 데이터를 함께 사용한다.  
레이블링 비용이 클 때 유용하다. (예: 의료 이미지 — 전문의 레이블링이 비쌈)

---

## 2. scikit-learn의 특징

### 일관성 있는 인터페이스

모든 모델이 동일한 메서드를 가진다. 모델을 바꿔도 코드 구조가 거의 같다.

```python
# 어떤 모델이든 동일한 패턴
model = SomeModel(hyperparameter=value)   # 1. 모델 생성
model.fit(X_train, y_train)               # 2. 학습
y_pred = model.predict(X_test)            # 3. 예측
score  = model.score(X_test, y_test)      # 4. 평가
```

### 주요 특징

- **분류, 회귀, 군집화, 차원 축소** 등 광범위한 알고리즘 지원
- **지속적으로 관리되는 패키지**: 모델 추가/삭제가 체계적으로 이루어짐
- **병렬 처리 지원**: `n_jobs=-1`로 모든 CPU 코어 활용
- **GPU 미사용**: CPU 기반 → 딥러닝보다 빠른 소규모 작업에 적합
- **NumPy 배열이 기본 입력 형식**

### scikit-learn 2.0의 주요 변화

- **차원 감소에 중점**: PCA, FA, ICA 등 강화
- **모델 결합(Composite Model)에 중점**: PCR(Principal Component Regression) 등
- 파이프라인과 메타 추정기의 기능 확장

---

## 3. Estimator와 Predictor 구조

scikit-learn의 모든 객체는 **Estimator** 기반이다.

### Estimator (추정기)

데이터에서 **파라미터를 학습**하는 객체. `fit()` 메서드를 가진다.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# 모두 Estimator: fit()으로 데이터에서 학습
scaler = StandardScaler()
scaler.fit(X)           # 평균, 표준편차 학습

pca = PCA(n_components=2)
pca.fit(X)              # 주성분 방향 학습

model = LinearRegression()
model.fit(X_train, y_train)  # 회귀 계수 학습
```

### Estimator의 5가지 유형

|유형|역할|주요 메서드|예시|
|---|---|---|---|
|**Transformers**|데이터 변환|`fit()`, `transform()`, `fit_transform()`|`StandardScaler`, `PCA`|
|**Regressors**|연속형 예측|`fit()`, `predict()`, `score()`|`LinearRegression`, `Ridge`|
|**Classifiers**|범주형 예측|`fit()`, `predict()`, `predict_proba()`|`LogisticRegression`, `SVC`|
|**Meta-estimators**|다른 모델을 감싸서 기능 확장|`fit()`, `predict()`|`Pipeline`, `GridSearchCV`, `BaggingClassifier`|
|**Clusterers**|군집화|`fit()`, `predict()`, `fit_predict()`|`KMeans`, `DBSCAN`|

---

## 4. ML 파이프라인 (make_pipeline)

### 파이프라인이란

**전처리 → 특성 추출 → 모델 학습**을 하나의 객체로 연결하는 구조다.

```
Data → Feature Extraction → Model Training → Final Model
```

파이프라인을 쓰는 이유:

1. **데이터 누수(Data Leakage) 방지**: 전처리가 훈련 데이터 기준으로만 학습됨
2. **코드 간결화**: 여러 단계를 하나의 객체로 관리
3. **교차검증에서 자동으로 각 fold에 맞게 전처리 재적용**

### 데이터 누수 문제

```python
# ❌ 잘못된 방식 (데이터 누수 발생)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)           # 전체 데이터로 스케일러 학습
X_train, X_test = train_test_split(X_scaled) # 이미 테스트 데이터가 스케일러에 영향을 줌

# ✅ 올바른 방식 (파이프라인 사용)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = make_pipeline(
    StandardScaler(),        # 훈련 데이터 기준으로 fit, 테스트는 transform만
    LinearRegression()
)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)  # 내부적으로 scaler.transform(X_test) 자동 처리
```

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np

# 예시: 완전한 파이프라인
pipe = Pipeline([
    ('scaler', StandardScaler()),         # 1. 정규화
    ('pca',    PCA(n_components=10)),     # 2. 차원 축소
    ('model',  Ridge(alpha=1.0))          # 3. 릿지 회귀
])

# make_pipeline: 이름 자동 지정 버전
pipe2 = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    Ridge(alpha=1.0)
)

# fit/predict가 전체 파이프라인에 자동 적용
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# 파이프라인의 특정 단계 접근
print(pipe.named_steps['pca'].explained_variance_ratio_)
print(pipe['model'].coef_)   # 2.0+ 방식
```

---

## 5. 전처리 (Preprocessing)

### 정규화 / 표준화

```python
from sklearn.preprocessing import (
    StandardScaler,    # Z-score 표준화 (평균 0, 표준편차 1)
    MinMaxScaler,      # [0, 1] 범위로 변환
    RobustScaler,      # 중앙값과 IQR 기반 (이상치에 강함)
    Normalizer         # 행 단위 L2 정규화 (벡터 단위 벡터화)
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)         # transform만 (fit 없음)

# 중요: 테스트 데이터는 transform만 해야 함 (훈련 기준 스케일 유지)
```

### 범주형 인코딩

```python
from sklearn.preprocessing import (
    LabelEncoder,       # 레이블을 정수로 (타깃 변수용)
    OrdinalEncoder,     # 서열형 범주 → 정수
    OneHotEncoder       # 범주 → 이진 벡터
)

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = ohe.fit_transform(X_categorical)
```

### 결측값 처리

```python
from sklearn.impute import SimpleImputer, KNNImputer

# 단순 보간
imputer = SimpleImputer(strategy='mean')      # 평균으로
imputer = SimpleImputer(strategy='median')    # 중위수로
imputer = SimpleImputer(strategy='most_frequent')  # 최빈값으로

# KNN 보간 (유사한 샘플 참조)
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
```

---

## 6. 데이터 변환 및 차원 축소

### 왜 차원 축소가 필요한가

변수(차원)가 많아질수록 다음 문제들이 생긴다.

```
차원의 저주 (Curse of Dimensionality):
  - 차원이 늘어날수록 데이터 공간이 지수적으로 커짐
  - 같은 N개 데이터라도 고차원에서는 모든 점이 서로 멀어짐
  - 거리 기반 알고리즘(KNN, SVM 등)의 성능 저하
  - 모델 학습에 필요한 데이터 양이 지수적으로 증가

다중공선성 (Multicollinearity):
  - 변수 간 상관이 높으면 회귀계수가 불안정해짐
  - 모델 해석이 어려워짐

계산 비용:
  - 변수가 많을수록 학습이 느림
  - 메모리 사용량 증가
```

차원 축소는 이 문제들을 해결하면서 **핵심 정보는 최대한 유지**하는 방법이다.

---

### PCA (Principal Component Analysis, 주성분 분석)

#### 핵심 아이디어

데이터가 **가장 많이 퍼진 방향(분산이 최대인 방향)** 을 새로운 축으로 삼아 데이터를 재표현한다.

```
원래 데이터 (2D 예시):
  y
  │     ●
  │  ●    ●
  │●   ●    ●
  │  ●    ●
  └────────── x

PC1 방향: 데이터가 가장 넓게 퍼진 방향 (↗ 방향)
PC2 방향: PC1과 직교하면서 그 다음으로 넓게 퍼진 방향 (↖ 방향)

PC1으로 투영하면 2D → 1D로 줄어들지만 대부분의 정보(분산)를 유지
```

#### 수학적 원리

1. 데이터를 **표준화** (평균 0, 분산 1)
2. **공분산 행렬** 계산: $C = \frac{1}{n-1}X^TX$
3. 공분산 행렬의 **고유값 분해**: $C = V \Lambda V^T$
    - 고유벡터($V$): 새로운 축의 방향 (주성분)
    - 고유값($\Lambda$): 각 축이 설명하는 분산의 크기
4. 고유값 큰 순서로 k개 고유벡터 선택
5. 데이터를 선택된 축으로 **투영**: $Z = XV_k$

```
고유값이 크다 = 그 방향으로 데이터가 많이 퍼져 있다 = 많은 정보를 담고 있다
고유값이 작다 = 그 방향으로 데이터가 거의 안 퍼져 있다 = 노이즈에 가깝다
```

#### 설명된 분산 비율 (Explained Variance Ratio)

$$\text{설명 비율}_k = \frac{\lambda_k}{\sum_{i=1}^{p}\lambda_i}$$

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 표준화 먼저 (PCA는 분산 기반이므로 스케일에 민감)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── PCA 수행 ───────────────────────────────────────────
pca_full = PCA()   # 전체 성분 유지 (분석용)
pca_full.fit(X_scaled)

# 각 주성분의 설명 분산 비율
evr = pca_full.explained_variance_ratio_
cumulative_evr = np.cumsum(evr)

print("주성분별 설명 분산 비율:")
for i, (var, cum) in enumerate(zip(evr[:10], cumulative_evr[:10])):
    print(f'  PC{i+1:2d}: {var:.4f} ({cum:.4f} 누적)')

# ── 스크리 플롯 (Scree Plot) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, len(evr[:20])+1), evr[:20], alpha=0.7)
axes[0].set_title('주성분별 설명 분산 비율 (Scree Plot)')
axes[0].set_xlabel('주성분 번호')
axes[0].set_ylabel('설명 분산 비율')

axes[1].plot(range(1, len(cumulative_evr)+1), cumulative_evr, 'b-o', markersize=3)
axes[1].axhline(y=0.90, color='red', linestyle='--', label='90% 기준선')
axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95% 기준선')
axes[1].set_title('누적 설명 분산 비율')
axes[1].set_xlabel('주성분 수')
axes[1].set_ylabel('누적 설명 분산 비율')
axes[1].legend()
plt.tight_layout()
plt.show()

# 90%, 95%, 99% 분산을 유지하는 데 필요한 성분 수
for threshold in [0.90, 0.95, 0.99]:
    n_comp = np.searchsorted(cumulative_evr, threshold) + 1
    print(f'{threshold*100:.0f}% 분산 유지: {n_comp}개 성분 필요 (전체 {X.shape[1]}개 중)')

# ── 실제 차원 축소 ────────────────────────────────────
pca = PCA(n_components=0.95)   # 95% 분산 유지하는 최소 성분 수 자동 결정
X_pca = pca.fit_transform(X_scaled)
print(f'\n차원 축소: {X_scaled.shape[1]}차원 → {X_pca.shape[1]}차원')
print(f'유지된 분산: {pca.explained_variance_ratio_.sum():.4f}')

# ── 로딩 행렬 분석 (각 원본 변수의 기여도) ──────────
loadings = pca.components_   # shape: (n_components, n_features)
print(f'\nPC1에 가장 크게 기여하는 변수:')
for idx in np.argsort(np.abs(loadings[0]))[::-1][:5]:
    print(f'  변수 {idx}: {loadings[0, idx]:+.4f}')
```

#### PCA 시각화

```python
# 2D로 축소해서 클래스 분리 시각화
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% 설명)')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% 설명)')
ax.set_title('PCA 2D 시각화')
plt.colorbar(scatter)
plt.show()
```

#### PCA 주의사항

```
1. 표준화 필수: 스케일이 다른 변수가 있으면 분산이 큰 변수가 주성분을 독점
   → 항상 StandardScaler 후 PCA 적용

2. 해석 어려움: 주성분은 원래 변수의 선형 결합 → 직관적 해석이 어려움
   → 예측 성능은 높아지지만 설명력은 떨어질 수 있음

3. 선형만 처리: 비선형 관계는 포착 못함 → 커널 PCA나 t-SNE 고려

4. 레이블 미사용: 비지도 방법이므로 y 정보를 활용하지 않음
   → 분류 성능 향상을 원하면 LDA(Linear Discriminant Analysis) 고려
```

---

### LDA (Linear Discriminant Analysis, 선형 판별 분석)

PCA와 달리 **클래스 레이블을 활용**해서 클래스 간 분리를 최대화하는 방향으로 차원을 축소한다.

```
PCA 목표: 전체 데이터의 분산을 최대화
LDA 목표: 클래스 간 분산 / 클래스 내 분산 을 최대화

→ LDA는 분류에 더 직접적으로 도움이 됨
→ 최대 성분 수 = 클래스 수 - 1 (이항 분류면 최대 1개)
```

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)   # 최대 클래스수-1
X_lda = lda.fit_transform(X_scaled, y)   # y를 사용! (지도 방식)

print(f'LDA 설명 분산 비율: {lda.explained_variance_ratio_}')

# 시각화 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (X_2d_comp, title) in zip(axes, [
    (pca_2d.fit_transform(X_scaled), 'PCA (비지도)'),
    (X_lda, 'LDA (지도)')
]):
    scatter = ax.scatter(X_2d_comp[:, 0], X_2d_comp[:, 1],
                         c=y, cmap='viridis', alpha=0.7)
    ax.set_title(title)
plt.tight_layout()
plt.show()
```

---

### FA (Factor Analysis, 요인 분석)

PCA가 "분산 최대화"를 목표로 한다면, FA는 **변수들이 공유하는 잠재 요인** 을 발견하는 것이 목표다.

```
가정: 관측 변수 = 잠재 요인들의 선형 결합 + 고유 분산(노이즈)

예: 학생 성적 데이터 (수학, 물리, 화학, 국어, 영어, 한국사)
    잠재 요인 1 (이공계 능력): 수학, 물리, 화학에 높은 기여
    잠재 요인 2 (인문계 능력): 국어, 영어, 한국사에 높은 기여
    
PCA: 전체 분산을 가장 잘 설명하는 축
FA:  공통 변동(요인)과 고유 변동(노이즈)을 분리
```

```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=3, random_state=42)
X_fa = fa.fit_transform(X_scaled)

# 요인 적재량 (Factor Loadings): 각 변수가 요인에 얼마나 기여하는가
loadings_fa = fa.components_.T   # shape: (n_features, n_components)
print("요인 적재량 (상위 변수 5개):")
for j in range(loadings_fa.shape[1]):
    top_vars = np.argsort(np.abs(loadings_fa[:, j]))[::-1][:3]
    print(f'  요인{j+1}: {[(i, f"{loadings_fa[i,j]:+.3f}") for i in top_vars]}')

# AIC/BIC로 최적 요인 수 결정
from sklearn.decomposition import FactorAnalysis
scores = []
for n in range(1, min(X.shape[1], 15)):
    fa_n = FactorAnalysis(n_components=n, random_state=42)
    fa_n.fit(X_scaled)
    scores.append(fa_n.score(X_scaled))   # 로그 우도
```

---

### ICA (Independent Component Analysis, 독립 성분 분석)

**칵테일 파티 문제**: 여러 사람이 동시에 말하는 방에서 각 마이크가 혼합된 소리를 녹음했을 때, 각 사람의 목소리를 분리하는 문제.

```
혼합 신호 X = A × S
  A: 혼합 행렬 (알 수 없음)
  S: 독립 원본 신호 (복원하고 싶은 것)

ICA 목표: X만으로 S를 복원
조건: 원본 신호들이 서로 통계적으로 독립이어야 함

PCA vs ICA:
  PCA: 성분이 서로 직교(uncorrelated)하면 됨 (상관 없음)
  ICA: 성분이 서로 통계적으로 독립(independent)해야 함 (훨씬 강한 조건)
  
  독립 = 상관 없음 + 고차 통계량도 관계 없음
```

```python
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt

# 칵테일 파티 시뮬레이션
np.random.seed(42)
n_samples = 2000
t = np.linspace(0, 8, n_samples)

# 원본 신호 3개 (서로 독립)
s1 = np.sin(2 * t)                         # 사인파
s2 = np.sign(np.sin(3 * t))                # 구형파
s3 = np.random.laplace(size=n_samples)     # 라플라스 노이즈

S = np.column_stack([s1, s2, s3])
S /= S.std(axis=0)   # 표준화

# 혼합 행렬 (알 수 없다고 가정)
A = np.array([[1, 1, 1],
              [0.5, 2, 1],
              [1.5, 1, 2]])

# 혼합 신호 (관측값)
X_mixed = S @ A.T

# ICA로 신호 분리
ica = FastICA(n_components=3, random_state=42, max_iter=500)
S_recovered = ica.fit_transform(X_mixed)

# 시각화
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
names = ['원본 신호', '혼합 신호 (관측)', 'ICA 복원 신호']
signals = [S.T, X_mixed.T, S_recovered.T]

for col, (name, sigs) in enumerate(zip(names, signals)):
    for row in range(3):
        axes[row, col].plot(t[:500], sigs[row, :500], linewidth=0.7)
        axes[row, col].set_title(f'{name} {row+1}')

plt.tight_layout()
plt.show()
```

---

### 비선형 차원 축소

PCA/FA/ICA는 선형 변환만 가능하다. 데이터가 비선형 구조를 가질 때는 다음 방법을 사용한다.

#### t-SNE (t-distributed Stochastic Neighbor Embedding)

고차원 데이터를 **2D/3D 시각화**에 특화된 방법이다.  
고차원에서 가까운 점들이 저차원에서도 가깝게 유지되도록 배치한다.

```
t-SNE의 특징:
  - 시각화 목적 (2D, 3D로만 축소)
  - 지역 구조 보존에 탁월 (클러스터가 뚜렷하게 분리됨)
  - 전역 구조 보존 미흡 (클러스터 간 거리는 의미 없음)
  - 비결정적 (실행마다 결과 다름, random_state 고정 필요)
  - 느림 (대량 데이터에 부적합)
  - 예측 불가: 새 데이터 변환 불가 (transform 없음)
```

```python
from sklearn.manifold import TSNE

# perplexity: 지역 이웃 수 (5~50, 데이터 크기에 따라 조정)
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            learning_rate='auto', init='pca', n_iter=1000)

# 먼저 PCA로 50차원으로 줄인 뒤 t-SNE (권장 방식, 속도 향상)
X_pca50 = PCA(n_components=min(50, X.shape[1])).fit_transform(X_scaled)
X_tsne = tsne.fit_transform(X_pca50)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7, s=5)
ax.set_title('t-SNE 2D 시각화')
plt.colorbar(scatter)
plt.show()
```

#### UMAP (Uniform Manifold Approximation and Projection)

t-SNE의 속도와 전역 구조 보존 문제를 개선한 방법이다. 실무에서 t-SNE를 빠르게 대체하고 있다.

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# UMAP은 transform() 가능 (새 데이터 변환 가능)
X_new_umap = reducer.transform(X_test_scaled)
```

#### Kernel PCA

표준 PCA에 커널 트릭을 적용해 **비선형 분리 가능한 데이터**를 다룰 수 있게 한 방법이다.

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_scaled)

# 다양한 커널
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    kpca_k = KernelPCA(n_components=2, kernel=kernel)
    X_k = kpca_k.fit_transform(X_scaled)
    print(f'{kernel:10s}: shape={X_k.shape}')
```

---

### 차원 축소 방법 종합 비교

|방법|유형|목적|레이블 사용|새 데이터 변환|속도|적합한 경우|
|---|---|---|---|---|---|---|
|**PCA**|선형|분산 최대화|아니오|가능|빠름|일반적 전처리, 노이즈 제거|
|**LDA**|선형|클래스 분리 최대화|예 (지도)|가능|빠름|분류 전처리|
|**FA**|선형|잠재 요인 발견|아니오|가능|빠름|잠재 구조 탐색|
|**ICA**|선형|독립 신호 분리|아니오|가능|보통|신호 분리, 노이즈 제거|
|**Kernel PCA**|비선형|비선형 구조 포착|아니오|가능|보통|비선형 데이터|
|**t-SNE**|비선형|지역 구조 시각화|아니오|불가|느림|2D/3D 시각화|
|**UMAP**|비선형|구조 보존 시각화|아니오|가능|빠름|시각화 + 전처리|

### 선택 가이드

```
데이터가 선형 구조인가?
   ├─ YES
   │   ├─ 분류 성능 향상이 목적? → LDA
   │   ├─ 잠재 요인 발견이 목적? → FA
   │   ├─ 신호 분리가 목적?      → ICA
   │   └─ 일반 차원 축소?        → PCA (표준)
   │
   └─ NO (비선형)
       ├─ 시각화만 필요?          → t-SNE 또는 UMAP
       ├─ 새 데이터 변환 필요?    → UMAP 또는 Kernel PCA
       └─ 속도 중요?              → UMAP (t-SNE보다 훨씬 빠름)
```

```python
# PCR (Principal Component Regression): PCA + 선형 회귀
# scikit-learn 2.0에서 강조하는 모델 결합의 대표 예시

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

pcr = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),   # 95% 분산 유지
    LinearRegression()
)

pcr.fit(X_train, y_train)
print(f'PCR R²: {pcr.score(X_test, y_test):.4f}')

# PCA 성분 수 최적화
from sklearn.model_selection import GridSearchCV
param_grid = {'pca__n_components': [0.80, 0.85, 0.90, 0.95, 0.99]}
grid = GridSearchCV(pcr, param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)
print(f'최적 분산 유지 비율: {grid.best_params_}')
```

### 특성 선택 (Feature Selection)

불필요한 변수를 제거해 모델 단순화 및 과적합 방지.

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2,    # 통계 기반 선택
    RFE,                              # 재귀적 특성 제거
    SelectFromModel,                  # 모델 기반 선택
    VarianceThreshold                 # 분산이 낮은 특성 제거
)
from sklearn.linear_model import Lasso

# 분산이 거의 없는 변수 제거 (상수에 가까운 변수)
sel = VarianceThreshold(threshold=0.01)
X_var = sel.fit_transform(X)

# 통계 검정으로 상위 k개 선택 (분류용)
sel_k = SelectKBest(score_func=f_classif, k=10)
X_best = sel_k.fit_transform(X, y)

# Lasso를 이용한 변수 선택 (계수가 0이 되는 변수 제거)
lasso = Lasso(alpha=0.01)
sel_lasso = SelectFromModel(lasso)
X_lasso = sel_lasso.fit_transform(X, y)

# RFE: 중요도가 낮은 특성을 반복적으로 제거
from sklearn.linear_model import LogisticRegression
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
print(f'선택된 특성: {rfe.support_}')
```

---

## 7. 정규화 (Regularization) — L1, L2

정규화는 **모델 복잡도에 패널티**를 부여해서 과적합을 방지하는 기법이다.

### L2 정규화 — Ridge

$$\min \sum(y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2$$

- 계수를 0에 가깝게 줄이지만 완전히 0으로 만들지 않는다
- **변수를 모두 유지**하면서 크기를 줄인다
- 다중공선성이 있을 때 효과적

### L1 정규화 — Lasso

$$\min \sum(y_i - \hat{y}_i)^2 + \alpha \sum |\beta_j|$$

- 일부 계수를 **정확히 0으로** 만든다
- **자동 변수 선택** 효과가 있다
- 변수가 많고 실제로 중요한 변수가 소수일 때 유용

### ElasticNet — L1 + L2 혼합

$$\min \sum(y_i - \hat{y}_i)^2 + \alpha \left[ \rho \sum|\beta_j| + (1-\rho) \sum \beta_j^2 \right]$$

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV

# alpha: 정규화 강도. 클수록 계수가 작아짐
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
enet  = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio: L1 비중

ridge.fit(X_train, y_train)
print(f'Ridge 계수 (0에 가깝지만 0은 아님): {ridge.coef_}')

lasso.fit(X_train, y_train)
print(f'Lasso 계수 (일부는 정확히 0): {lasso.coef_}')
print(f'선택된 변수 수: {(lasso.coef_ != 0).sum()}')

# 교차검증으로 최적 alpha 자동 선택
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(X_train, y_train)
print(f'최적 alpha: {ridge_cv.alpha_}')
```

---

## 8. 회귀 모델 (Regressors)

### LinearRegression

$$y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$$

정규화 없는 기본 선형 회귀다. OLS(최소제곱법)로 계수를 추정한다.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

print(f'절편: {model.intercept_:.4f}')
print(f'계수: {model.coef_}')

y_pred = model.predict(X_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}')
print(f'R²:   {r2_score(y_test, y_pred):.4f}')
```

### Lasso, Ridge, ElasticNet

위 정규화 섹션 참고. 변수가 많거나 다중공선성이 있을 때 선형 회귀보다 안정적이다.

### SGDRegressor (대량 데이터용)

**확률적 경사하강법**으로 회귀 계수를 학습한다.  
데이터가 메모리에 다 들어오지 않는 대량 데이터(Out-of-core learning)에 적합하다.

```python
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(
    loss='squared_error',      # 손실 함수
    penalty='l2',              # 정규화 종류 (l1, l2, elasticnet)
    alpha=0.0001,              # 정규화 강도
    learning_rate='invscaling', # 학습률 감소 방식
    max_iter=1000,
    random_state=42
)

# 미니배치 학습 (partial_fit으로 데이터를 나눠서 학습)
chunk_size = 1000
for i in range(0, len(X_train), chunk_size):
    X_chunk = X_train[i:i+chunk_size]
    y_chunk = y_train[i:i+chunk_size]
    sgd.partial_fit(X_chunk, y_chunk)
```

---

## 9. 분류 모델 (Classifiers)

### LogisticRegression

이름은 회귀지만 **이진 및 다중 클래스 분류**에 사용한다.  
시그모이드 함수로 출력을 확률(0~1)로 변환한다.

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots)}}$$

```python
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

lr = LogisticRegression(
    C=1.0,              # 정규화 강도의 역수 (C 작을수록 강한 정규화)
    penalty='l2',       # 정규화 종류
    solver='lbfgs',     # 최적화 방법 (lbfgs, saga, liblinear)
    max_iter=1000,
    multi_class='auto'  # 다중 분류 전략 자동 선택
)
lr.fit(X_train, y_train)

# 클래스별 확률 출력
proba = lr.predict_proba(X_test)   # shape: (n_samples, n_classes)
print(f'예측 확률 (상위 3개): {proba[:3]}')

# 교차검증으로 최적 C 자동 선택
lr_cv = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='lbfgs')
lr_cv.fit(X_train, y_train)
print(f'최적 C: {lr_cv.C_}')
```

### RidgeClassifier

Ridge 정규화를 적용한 분류기다. 다중 클래스 문제에서 빠르다.

```python
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

rc = RidgeClassifier(alpha=1.0)
rc.fit(X_train, y_train)

# 교차검증으로 최적 alpha 자동 선택
rc_cv = RidgeClassifierCV(alphas=[0.01, 0.1, 1, 10])
rc_cv.fit(X_train, y_train)
```

### SGDClassifier (대량 데이터용)

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(
    loss='hinge',      # SVM과 같은 손실 (log_loss → 로지스틱과 같음)
    penalty='l2',
    alpha=0.0001,
    random_state=42
)
sgd_clf.fit(X_train, y_train)
```

---

## 10. 모델 선택과 평가

### 훈련/테스트 분할

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 테스트 비율
    random_state=42,    # 재현성
    stratify=y          # 클래스 비율 유지 (분류 문제에서 중요)
)
```

### 교차검증 (Cross-Validation)

훈련 데이터를 여러 fold로 나눠서 **모델의 일반화 성능**을 더 신뢰성 있게 추정한다.

```
K-Fold 교차검증 (K=5):

Fold1: [검증] [훈련] [훈련] [훈련] [훈련]  → 성능 1
Fold2: [훈련] [검증] [훈련] [훈련] [훈련]  → 성능 2
Fold3: [훈련] [훈련] [검증] [훈련] [훈련]  → 성능 3
Fold4: [훈련] [훈련] [훈련] [검증] [훈련]  → 성능 4
Fold5: [훈련] [훈련] [훈련] [훈련] [검증]  → 성능 5

최종 성능 = 5개 성능의 평균 ± 표준편차
```

```python
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    cross_validate
)

model = LogisticRegression()

# 기본 교차검증
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'CV 정확도: {scores.mean():.3f} ± {scores.std():.3f}')

# 층화 K-Fold (클래스 비율 유지)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

# 여러 지표 동시에
results = cross_validate(model, X, y, cv=5,
                         scoring=['accuracy', 'f1_weighted', 'roc_auc'],
                         return_train_score=True)
for metric, vals in results.items():
    if 'test' in metric:
        print(f'{metric}: {vals.mean():.3f} ± {vals.std():.3f}')
```

### 하이퍼파라미터 튜닝

#### GridSearchCV

모든 하이퍼파라미터 조합을 탐색한다. 정확하지만 조합 수가 많으면 느리다.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'ridge__alpha': [0.01, 0.1, 1, 10, 100],
    'pca__n_components': [5, 10, 20]
}

pipe = make_pipeline(StandardScaler(), PCA(), Ridge())
grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,     # 모든 CPU 코어 사용
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f'최적 파라미터: {grid_search.best_params_}')
print(f'최적 CV 점수: {-grid_search.best_score_:.4f}')
print(f'테스트 성능: {grid_search.score(X_test, y_test):.4f}')
```

#### RandomizedSearchCV

무작위로 하이퍼파라미터 조합을 샘플링해서 탐색한다. GridSearch보다 빠르고 종종 더 좋은 결과를 낸다.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    'ridge__alpha': loguniform(1e-3, 1e3),   # 로그 균일 분포
    'pca__n_components': randint(5, 50)       # 정수 균일 분포
}

random_search = RandomizedSearchCV(
    pipe,
    param_dist,
    n_iter=50,       # 탐색 횟수
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
print(f'최적 파라미터: {random_search.best_params_}')
```

---

## 11. 분류 모델 평가 지표

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 양성 클래스 확률

print(f'정확도:  {accuracy_score(y_test, y_pred):.4f}')
print(f'정밀도:  {precision_score(y_test, y_pred, average="weighted"):.4f}')
print(f'재현율:  {recall_score(y_test, y_pred, average="weighted"):.4f}')
print(f'F1:      {f1_score(y_test, y_pred, average="weighted"):.4f}')
print(f'AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}')

# 전체 리포트 한 번에
print(classification_report(y_test, y_pred, target_names=['음성', '양성']))

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print(f'혼동 행렬:\n{cm}')
```

---

## 12. 군집화 (Clustering)

비지도학습의 대표적 방법이다. 레이블 없이 유사한 데이터끼리 묶는다.

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# K-Means: 미리 k를 지정해야 함
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
print(f'군집 중심:\n{kmeans.cluster_centers_}')

# 최적 k 찾기 (엘보우 방법)
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

# 실루엣 점수: 군집의 품질 (-1~1, 1에 가까울수록 좋음)
score = silhouette_score(X, labels)
print(f'실루엣 점수: {score:.3f}')

# DBSCAN: k를 지정하지 않아도 됨, 이상치 탐지 가능
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X)
# -1 레이블 = 이상치(noise)
print(f'이상치 수: {(labels_db == -1).sum()}')
```

---

## 13. 메타 추정기 (Meta-estimators)

다른 모델을 내부에서 사용하는 고수준 모델이다.

### 앙상블 방법

```python
from sklearn.ensemble import (
    RandomForestClassifier,    # 배깅 + 랜덤 특성 선택
    GradientBoostingClassifier, # 부스팅
    AdaBoostClassifier,         # 부스팅
    VotingClassifier,           # 다수결 투표
    StackingClassifier          # 스태킹 (메타 학습)
)

# 랜덤 포레스트
rf = RandomForestClassifier(
    n_estimators=100,          # 트리 수
    max_depth=None,            # 트리 최대 깊이 (None = 제한 없음)
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# 변수 중요도
importances = rf.feature_importances_
print(f'상위 5개 변수 중요도: {sorted(importances, reverse=True)[:5]}')

# 스태킹 (여러 모델 예측을 메타 모델이 학습)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

estimators = [
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('svc', SVC(probability=True))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
```

### XGBoost (eXtreme Gradient Boosting)

scikit-learn 외부 라이브러리지만 **scikit-learn 호환 인터페이스**를 제공해서 파이프라인과 GridSearchCV에 바로 쓸 수 있다.

**부스팅의 원리**:

```
부스팅 = 약한 모델(Weak Learner)을 순차적으로 쌓아 강한 모델을 만든다

1단계: 첫 번째 트리로 예측
        → 잔차(오차) 계산

2단계: 잔차를 학습하는 두 번째 트리 생성
        → 업데이트된 잔차 계산

3단계: 반복...
        최종 예측 = 모든 트리 예측의 합

랜덤 포레스트(병렬):  트리 100개를 독립적으로 학습 → 다수결
부스팅(순차):         트리 100개를 순서대로 학습 → 이전 오차를 다음이 보완
```

**XGBoost가 빠른 이유**:

|특징|설명|
|---|---|
|**2차 미분 활용**|손실 함수의 헤시안(2차 미분)으로 분할 기준을 더 정확하게 결정|
|**정규화 내장**|L1(α), L2(λ) 정규화로 과적합 방지 (Gradient Boosting에는 없음)|
|**병렬 처리**|트리 내 분할 기준 탐색을 병렬화|
|**희소 데이터 최적화**|결측값과 0이 많은 데이터 효율적 처리|
|**가지치기 전략**|최대 깊이까지 키운 뒤 불필요한 가지를 제거 (depth-first pruning)|

```python
# pip install xgboost
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── 기본 사용 (분류) ─────────────────────────────────────
xgb_clf = XGBClassifier(
    n_estimators=200,       # 트리 수 (부스팅 라운드 수)
    max_depth=6,            # 각 트리의 최대 깊이 (3~10)
    learning_rate=0.1,      # 각 트리의 기여도 축소 (작을수록 과적합 방지, 트리 수 늘려야)
    subsample=0.8,          # 각 트리 학습 시 사용할 데이터 비율 (배깅 효과)
    colsample_bytree=0.8,   # 각 트리에서 사용할 특성 비율
    reg_alpha=0.1,          # L1 정규화 (Lasso)
    reg_lambda=1.0,         # L2 정규화 (Ridge)
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(X_train, y_train)
print(f'XGBoost 정확도: {xgb_clf.score(X_test, y_test):.4f}')

# ── 조기 종료 (Early Stopping) ─────────────────────────
# 검증 성능이 더 이상 개선되지 않으면 학습 중단 → 최적 트리 수 자동 결정
xgb_es = XGBClassifier(n_estimators=1000, learning_rate=0.05,
                        random_state=42, eval_metric='logloss')
xgb_es.fit(X_train, y_train,
           eval_set=[(X_test, y_test)],
           early_stopping_rounds=20,   # 20라운드 연속 개선 없으면 중단
           verbose=False)
print(f'최적 트리 수: {xgb_es.best_iteration}')

# ── sklearn 파이프라인과 통합 ────────────────────────────
pipe_xgb = make_pipeline(
    StandardScaler(),
    XGBClassifier(n_estimators=100, random_state=42,
                  eval_metric='logloss', verbosity=0)
)
pipe_xgb.fit(X_train, y_train)

# ── GridSearchCV 하이퍼파라미터 튜닝 ──────────────────────
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.05, 0.1, 0.2]
}
grid = GridSearchCV(pipe_xgb, param_grid, cv=5,
                    scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print(f'최적 파라미터: {grid.best_params_}')
print(f'테스트 AUC: {grid.score(X_test, y_test):.4f}')

# ── 변수 중요도 ──────────────────────────────────────────
import matplotlib.pyplot as plt
from xgboost import plot_importance

plot_importance(xgb_clf, max_num_features=10, importance_type='gain')
plt.title('XGBoost 변수 중요도 (Gain 기준)')
plt.tight_layout()
plt.show()
```

**XGBoost 주요 파라미터 정리**

|파라미터|설명|권장 범위|
|---|---|---|
|`n_estimators`|트리 수 (부스팅 라운드)|100~1000, early_stopping으로 결정|
|`max_depth`|트리 최대 깊이|3~10 (깊을수록 과적합 위험)|
|`learning_rate`|각 트리의 기여도|0.01~0.3 (작을수록 트리 수 더 필요)|
|`subsample`|행 샘플링 비율|0.6~1.0|
|`colsample_bytree`|열 샘플링 비율|0.6~1.0|
|`reg_alpha`|L1 정규화|0~1|
|`reg_lambda`|L2 정규화|1 (기본값)|
|`gamma`|분할 최소 손실 감소량|0~5 (클수록 보수적)|

**GradientBoosting vs XGBoost 비교**

|구분|sklearn GradientBoosting|XGBoost|
|---|---|---|
|미분 차수|1차 미분만|1차 + 2차 미분 (헤시안)|
|정규화|없음|L1, L2 내장|
|병렬 처리|없음|있음|
|결측값 처리|불가 (전처리 필요)|자동 처리|
|속도|느림|빠름|
|메모리|많음|효율적|

---

## 14. ML 알고리즘 분류 체계

알고리즘을 **학습 방식의 수학적 기반**에 따라 분류하면 어떤 상황에 어떤 모델을 선택할지 이해하기 쉬워진다.

```
ML 알고리즘
   ├─ 확률 기반   : 확률 모델로 데이터를 설명
   ├─ 오차 기반   : 예측 오차를 최소화
   ├─ 정보 기반   : 정보 이론으로 데이터를 분할
   └─ 유사도 기반 : 데이터 간 거리/유사도로 판단
```

---

### 확률 기반 (Probabilistic Models)

데이터를 **확률 분포로 모델링**하고, 어떤 클래스일 확률이 가장 높은지를 기준으로 예측한다.

#### Logistic Regression (로지스틱 회귀)

$$P(y=1|x) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}$$

- 출력이 0~1 사이의 **확률값**이다
- 시그모이드(sigmoid) 함수로 선형 결합을 확률로 변환한다
- **결정 경계가 선형**이다 (데이터가 선형으로 분리 가능해야 잘 작동)
- MLE(최대우도법)로 파라미터를 추정한다

```
직관:
  "이 이메일이 스팸일 확률 = 87%"처럼 확률로 해석 가능
  임계값(threshold=0.5)으로 클래스를 결정
```

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)

# 확률 출력 (핵심!)
proba = lr.predict_proba(X_test)   # [[P(y=0), P(y=1)], ...]
print(f'클래스 확률 (상위 3개):\n{proba[:3]}')

# 임계값 조정 (기본 0.5 → 변경 가능)
threshold = 0.3   # 민감도 높이기 (암 진단처럼 FN 최소화)
y_pred_custom = (proba[:, 1] >= threshold).astype(int)
```

#### Naive Bayes (나이브 베이즈)

베이즈 정리를 이용해 **사후 확률**을 계산한다.

$$P(y|X) \propto P(y) \cdot \prod_{i=1}^{n}P(x_i|y)$$

"나이브(Naive)"한 이유: 특성들이 **서로 독립**이라는 단순한 가정을 한다.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# GaussianNB: 연속형 특성 (정규분포 가정)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# MultinomialNB: 텍스트 분류 (단어 빈도수)
# BernoulliNB: 이진 특성 (단어 있음/없음)
```

**장점**: 매우 빠름, 소량 데이터에서도 잘 작동, 텍스트 분류에 강함  
**단점**: 특성 독립 가정이 현실과 다를 때 성능 저하

#### Bayesian / Gaussian Process (GP, 가우시안 프로세스)

함수 자체를 **확률 분포로 모델링**한다. 예측값과 함께 **불확실성(신뢰구간)** 을 출력한다.

```
일반 회귀: "이 점에서 예측값 = 5.2"
GP 회귀:  "이 점에서 예측값 = 5.2 ± 0.8 (95% CI)"

데이터가 없는 구간에서 불확실성이 크게 표현됨 → 베이지안 최적화에 활용
```

```python
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

# 커널: 데이터 포인트 간 유사도를 정의하는 함수
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gpr.fit(X_train, y_train)

# 예측값 + 표준편차 (불확실성)
y_pred, y_std = gpr.predict(X_test, return_std=True)
print(f'예측값: {y_pred[:3]}')
print(f'불확실성: {y_std[:3]}')   # 클수록 예측이 불확실

# 가우시안 프로세스 분류
gpc = GaussianProcessClassifier(kernel=RBF())
gpc.fit(X_train, y_train)
proba_gp = gpc.predict_proba(X_test)
```

**장점**: 불확실성 정량화, 소량 데이터에서 강함, 하이퍼파라미터 자동 최적화  
**단점**: $O(n^3)$ 계산 복잡도 → 대량 데이터에 부적합

---

### 오차 기반 (Error-based Models)

**예측 오차(잔차)를 최소화**하는 방향으로 파라미터를 업데이트한다.

#### ANN (Artificial Neural Network, 인공 신경망)

뉴런을 층(layer)으로 쌓아 복잡한 함수를 근사한다.

```
입력층 → [은닉층 1] → [은닉층 2] → ... → 출력층

각 뉴런: z = w₁x₁ + w₂x₂ + ... + b
         output = activation(z)

활성화 함수 (비선형성 추가):
  ReLU:    max(0, z)           ← 딥러닝 기본
  Sigmoid: 1/(1+e^-z)          ← 이진 출력
  Tanh:    (e^z - e^-z)/(e^z + e^-z)
  Softmax: e^zᵢ / Σe^zⱼ        ← 다중 분류 출력
```

**역전파(Backpropagation)**:

```
순전파: 입력 → 예측값 계산
         ↓
손실 계산: Loss = f(y_pred, y_true)
         ↓
역전파: 출력층 → 은닉층 순서로 연쇄 미분법(Chain Rule)으로
        각 가중치에 대한 손실의 그래디언트 계산
         ↓
가중치 업데이트: w ← w - α × ∂Loss/∂w
```

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# MLP (Multi-Layer Perceptron) = 완전연결 신경망
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 은닉층 구조 (3층)
    activation='relu',                  # 활성화 함수
    solver='adam',                      # 최적화 방법 (역전파 + Adam)
    alpha=0.001,                        # L2 정규화
    learning_rate_init=0.001,           # 초기 학습률
    max_iter=500,
    early_stopping=True,                # 검증 손실 개선 없으면 조기 종료
    validation_fraction=0.1,            # 검증용 데이터 비율
    random_state=42
)
mlp.fit(X_train, y_train)
print(f'MLP 정확도: {mlp.score(X_test, y_test):.4f}')
print(f'학습 손실 곡선 (마지막 5개): {mlp.loss_curve_[-5:]}')

# 주의: sklearn MLP는 GPU 미지원 → 대규모 딥러닝은 PyTorch/TensorFlow 사용
```

**오차 기반 알고리즘 비교**

|모델|오차 함수|최적화|특징|
|---|---|---|---|
|선형 회귀|MSE|OLS / 경사하강법|해석 쉬움, 선형만|
|로지스틱 회귀|Cross-entropy|BFGS / L-BFGS|확률 출력|
|ANN/MLP|Cross-entropy / MSE|Adam / SGD + 역전파|비선형, 고차원|
|XGBoost|임의 손실 함수|2차 미분 그래디언트 부스팅|앙상블, 빠름|

---

### 정보 기반 (Information-based Models)

**정보 이론(Information Theory)** 을 기반으로 데이터를 분할한다.  
"이 기준으로 나누면 불순도(impurity)가 얼마나 줄어드는가"를 계산해서 최선의 분할을 찾는다.

#### Decision Tree (의사결정 트리)

분할 기준: **정보이득(Information Gain)** 또는 **지니 불순도(Gini Impurity)**

$$\text{엔트로피}: H = -\sum_k p_k \log_2 p_k$$ $$\text{정보이득}: IG = H(\text{부모}) - \sum_i \frac{|자식_i|}{|부모|} H(\text{자식}_i)$$ $$\text{지니}: Gini = 1 - \sum_k p_k^2$$

```
엔트로피: 불확실성의 정도
  완전 순수 (한 클래스만): H = 0    ← 정보이득 최대
  완전 혼합 (반반):        H = 1    ← 정보이득 0

정보이득이 가장 큰 특성과 기준값을 선택해서 분할
```

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(
    criterion='gini',        # 분할 기준: 'gini' 또는 'entropy'
    max_depth=5,             # 과적합 방지: 깊이 제한
    min_samples_split=10,    # 분할 최소 샘플 수
    min_samples_leaf=5,      # 리프 노드 최소 샘플 수
    random_state=42
)
dt.fit(X_train, y_train)

# 트리 규칙 텍스트로 출력 (화이트박스)
rules = export_text(dt, feature_names=feature_names)
print(rules[:500])

# 트리 시각화
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=feature_names, class_names=class_names,
          filled=True, ax=ax, max_depth=3)
plt.show()

# 변수 중요도 (정보이득 기여도 합산)
importances = dict(zip(feature_names, dt.feature_importances_))
top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
print(f'상위 5개 변수: {top5}')
```

#### Random Forest (랜덤 포레스트)

Decision Tree의 과적합 문제를 해결하기 위해 **배깅(Bagging) + 랜덤 특성 선택**을 결합한다.

```
배깅 (Bootstrap Aggregating):
  1. 원본 데이터에서 복원추출로 N개의 서로 다른 훈련셋 생성
  2. 각 훈련셋으로 독립적인 트리 학습
  3. 예측: 분류→다수결, 회귀→평균

랜덤 특성 선택:
  각 분할에서 전체 특성 중 √p개만 무작위로 선택해서 탐색
  → 트리 간 상관성 감소 → 앙상블 효과 극대화
```

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

rf = RandomForestClassifier(
    n_estimators=200,          # 트리 수 (많을수록 안정적, 수렴점 이후 효과 없음)
    max_features='sqrt',       # 각 분할에서 탐색할 특성 수 (분류: sqrt(p), 회귀: p/3)
    max_depth=None,            # 트리를 끝까지 키움 (배깅이 과적합 방지)
    bootstrap=True,            # 복원추출 (배깅)
    oob_score=True,            # Out-of-Bag 점수 계산 (별도 검증 셋 없이 평가)
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

print(f'OOB 점수: {rf.oob_score_:.4f}')   # 학습 안 한 샘플로 평가
print(f'테스트 점수: {rf.score(X_test, y_test):.4f}')

# 변수 중요도 시각화
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:15]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(indices)), importances[indices],
       yerr=std[indices], align='center', alpha=0.7)
ax.set_xticks(range(len(indices)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
ax.set_title('랜덤 포레스트 변수 중요도')
plt.tight_layout()
plt.show()
```

#### Gradient Boosting Tree 계열

```
GradientBoosting → XGBoost → LightGBM → CatBoost 발전 흐름

각각의 특징:
  GradientBoosting: sklearn 기본, 느리지만 안정적
  XGBoost:          2차 미분 활용, 병렬화, 빠름
  LightGBM:         리프 우선 분할(leaf-wise), 매우 빠름, 메모리 효율
  CatBoost:         범주형 변수 자동 처리, 데이터 누수 방지
```

```python
# LightGBM (pip install lightgbm)
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,        # 트리의 리프 수 (XGBoost의 max_depth와 다른 개념)
    random_state=42
)
lgbm.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         callbacks=[lgbm.early_stopping(50)])
```

---

### 유사도 기반 (Similarity-based Models)

**데이터 포인트 간 거리 또는 유사도**를 이용해서 분류/예측/군집화를 수행한다.

#### KNN (K-Nearest Neighbors, K-최근접 이웃)

새로운 데이터가 들어오면 **훈련 데이터에서 가장 가까운 K개 이웃**을 찾아서 다수결(분류) 또는 평균(회귀)으로 예측한다.

```
K=3 분류 예시:
  새 점 ★의 가장 가까운 3개 이웃: ●, ●, ○
  ● : 2개, ○ : 1개 → ★를 ● 클래스로 분류

K가 작을수록: 결정 경계가 복잡 (과적합 위험)
K가 클수록:  결정 경계가 단순 (과소적합 위험)

거리 측정:
  유클리드 거리: √Σ(xᵢ-yᵢ)²  (기본값, 연속형)
  맨해튼 거리:   Σ|xᵢ-yᵢ|    (이상치에 강함)
  코사인 유사도: cos(θ)         (텍스트, 방향 중심)
```

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

knn = KNeighborsClassifier(
    n_neighbors=5,         # K 값 (교차검증으로 결정)
    metric='minkowski',    # 거리 측정 방법 (p=2 → 유클리드)
    p=2,
    weights='distance',    # 거리 가중치 (가까울수록 더 큰 영향)
    n_jobs=-1
)
knn.fit(X_train, y_train)

# 최적 K 탐색 (elbow method)
from sklearn.model_selection import cross_val_score
k_scores = []
for k in range(1, 31):
    knn_k = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(knn_k, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

import matplotlib.pyplot as plt
plt.plot(range(1, 31), k_scores, 'b-o')
plt.xlabel('K 값')
plt.ylabel('CV 정확도')
plt.title('최적 K 탐색')
plt.axvline(x=k_scores.index(max(k_scores))+1, color='red', linestyle='--')
plt.show()

# KNN의 중요한 전처리 요구사항
# 반드시 스케일 정규화 필요! (거리가 스케일에 민감)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

knn_pipe = make_pipeline(
    StandardScaler(),   # 필수!
    KNeighborsClassifier(n_neighbors=5)
)
```

**주의**: KNN은 **게으른 학습(Lazy Learning)** 이다. 학습 단계에서 아무것도 하지 않고 데이터를 그대로 저장한다. 예측 시 전체 훈련 데이터와 거리를 계산한다 → 데이터가 많으면 예측이 느리다.

#### SVM (Support Vector Machine, 서포트 벡터 머신)

클래스 간 **마진(Margin)을 최대화**하는 결정 경계(초평면)를 찾는다.

```
마진 = 결정 경계와 가장 가까운 데이터 포인트들(서포트 벡터) 사이의 거리

마진을 최대화하면:
  → 새 데이터에 대한 일반화 성능이 높아짐
  → 분류 경계가 더 안정적

커널 트릭:
  선형으로 분리 불가능한 경우 → 고차원으로 변환해서 선형 분리
  RBF 커널:  exp(-γ||x-y||²)  → 가장 많이 사용
  Poly 커널: (γxᵀy + r)^d
```

```python
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 분류 (반드시 스케일 정규화 필요)
svc = make_pipeline(
    StandardScaler(),
    SVC(
        kernel='rbf',    # 커널: 'linear', 'rbf', 'poly', 'sigmoid'
        C=1.0,           # 오분류 허용도 (클수록 복잡한 경계, 과적합 위험)
        gamma='scale',   # RBF 커널의 영향 범위 (작을수록 넓은 범위)
        probability=True  # predict_proba() 사용 가능 (느려짐)
    )
)
svc.fit(X_train, y_train)
proba_svm = svc.predict_proba(X_test)

# 선형 SVM (대량 데이터에 빠름)
lsvc = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1.0, max_iter=5000)
)
```

---

### 알고리즘 분류 체계 종합

|기반|알고리즘|핵심 원리|해석 가능성|스케일 민감도|과적합 위험|
|---|---|---|---|---|---|
|**확률**|Logistic Regression|확률 모델링, MLE|높음|있음|낮음|
|**확률**|Naive Bayes|베이즈 정리, 독립 가정|높음|없음|매우 낮음|
|**확률**|Gaussian Process|함수의 확률 분포|중간|있음|낮음|
|**오차**|Linear Regression|MSE 최소화|높음|있음|낮음|
|**오차**|ANN/MLP|역전파, 경사하강법|낮음|있음|높음|
|**오차**|XGBoost|2차 미분 부스팅|중간|없음|중간|
|**정보**|Decision Tree|정보이득/지니|매우 높음|없음|높음|
|**정보**|Random Forest|배깅 + 랜덤 특성|중간|없음|낮음|
|**정보**|Gradient Boosting|순차적 잔차 학습|중간|없음|중간|
|**유사도**|KNN|이웃 다수결|높음|매우 있음|중간|
|**유사도**|SVM|최대 마진 초평면|낮음|매우 있음|낮음|

### 상황별 알고리즘 선택 가이드

```
데이터 크기 작음 (n < 1,000)?
   → Logistic Regression, SVM, KNN, Gaussian Process

데이터 크기 큼 (n > 10,000)?
   → Random Forest, XGBoost, SGD 계열, MLP

해석 가능성이 중요?
   → Logistic Regression (계수), Decision Tree (규칙)

비선형 관계?
   → Random Forest, XGBoost, MLP, SVM(RBF)

범주형 변수가 많음?
   → CatBoost, XGBoost (자동 처리)

불확실성 정량화 필요?
   → Gaussian Process, Bayesian 방법

빠른 예측이 필요?
   → Logistic Regression, Linear SVM (단순 행렬 연산)

표준화 없이 사용?
   → Tree 계열 (Decision Tree, Random Forest, XGBoost)
   → 나머지는 StandardScaler 필수
```

---

## 15. 전체 ML 파이프라인 코드 예시

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ── 1. 데이터 로드 ───────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
print(f'데이터 크기: {X.shape}')
print(f'클래스 분포: {np.bincount(y)}')

# ── 2. 훈련/테스트 분할 ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── 3. 파이프라인 구성 ──────────────────────────────────
pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),    # 95% 분산 유지
    LogisticRegression(max_iter=1000, random_state=42)
)

# ── 4. 교차검증 성능 확인 ───────────────────────────────
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
print(f'\nCV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

# ── 5. 하이퍼파라미터 튜닝 ─────────────────────────────
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'pca__n_components': [0.90, 0.95, 0.99]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print(f'최적 파라미터: {grid.best_params_}')
print(f'최적 CV AUC:  {grid.best_score_:.3f}')

# ── 6. 최종 평가 ────────────────────────────────────────
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(f'\n테스트 AUC: {roc_auc_score(y_test, y_prob):.3f}')
print('\n분류 리포트:')
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ── 7. 모델 저장 ────────────────────────────────────────
import joblib
joblib.dump(best_model, 'best_model.pkl')
loaded_model = joblib.load('best_model.pkl')
```

---

## 15. 전체 구조 정리

```
scikit-learn
   │
   ├─ 전처리 (Preprocessing)
   │   ├─ StandardScaler, MinMaxScaler, RobustScaler
   │   ├─ OneHotEncoder, LabelEncoder
   │   └─ SimpleImputer, KNNImputer
   │
   ├─ 데이터 변환 / 차원 축소
   │   ├─ PCA, FactorAnalysis, FastICA
   │   ├─ SelectKBest, RFE, SelectFromModel
   │   └─ PolynomialFeatures
   │
   ├─ 모델 (Estimators)
   │   ├─ 회귀: LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
   │   ├─ 분류: LogisticRegression, RidgeClassifier, SGDClassifier, SVC
   │   ├─ 앙상블: RandomForest, GradientBoosting, AdaBoost, Voting, Stacking
   │   └─ 군집화: KMeans, DBSCAN, AgglomerativeClustering
   │
   ├─ 파이프라인 (Pipeline)
   │   └─ make_pipeline, Pipeline
   │
   ├─ 모델 선택 / 평가
   │   ├─ train_test_split, cross_val_score, cross_validate
   │   ├─ GridSearchCV, RandomizedSearchCV
   │   └─ StratifiedKFold, KFold
   │
   └─ 평가 지표 (Metrics)
       ├─ 분류: accuracy, precision, recall, f1, roc_auc
       ├─ 회귀: MSE, RMSE, MAE, R²
       └─ 군집화: silhouette_score
```