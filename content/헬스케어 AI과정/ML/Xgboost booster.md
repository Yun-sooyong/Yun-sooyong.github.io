---
title: 038 Xgboost booster objective
tag:
  - 헬스케어 ai
  - ML
description: 260605 수업 내용 정리
---

# XGBoost: 경사하강법 — Booster & Objective

XGBoost는 Gradient Boosting을 기반으로 **뉴턴법(2차 미분)** 을 도입하고 여러 공학적 최적화를 더한 라이브러리다. 약한 학습기를 순차적으로 쌓아 손실 함수를 최소화하는 과정이 핵심이며, sklearn 없이도 고수준 API(`Booster`)와 sklearn 호환 래퍼(`XGBClassifier`) 두 가지 방식으로 사용할 수 있다.

```
약한 학습기 → 약한 학습기 → 약한 학습기 → ...
  ↓ 순차적으로 손실 감소 (뉴턴법으로 방향과 보폭 결정)
강한 학습기
```

---

## 1. XGBoost의 원리와 기능

### Boosting + 뉴턴법

일반 Gradient Boosting은 손실 함수의 **1차 미분(기울기)** 만 사용한다. XGBoost는 여기에 **2차 미분(헤시안)** 까지 활용해서 분할 기준을 더 정확하게 결정한다.

$$\text{분할 이득} = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}$$

- $G$: 1차 미분의 합 (기울기)
- $H$: 2차 미분의 합 (헤시안, 곡률)
- $\lambda$: L2 정규화 항

뉴턴법은 기울기(방향)뿐 아니라 곡률(보폭)까지 고려하므로 분할 지점을 더 빠르고 정확하게 찾는다. 다만 데이터가 매우 많으면 헤시안 계산도 그만큼 많아지므로 "대량에 악함"이라는 특성이 있다. 이를 완화하기 위해 서브샘플링, 캐시 최적화 등의 방법을 함께 사용한다.

---

### 트리 모델이라서 생기는 전처리 특징

XGBoost는 분기(Split) 기반으로 학습하기 때문에 **데이터의 절대적인 크기(스케일)가 아닌 순서(대소 관계)만** 보고 분할점을 결정한다. 따라서 다른 모델에서 필수인 정규화/표준화를 굳이 하지 않아도 결과가 달라지지 않는다. 이상치도 스케일 왜곡이 없어서 분기 방향 결정에 크게 영향을 미치지 않는다.

결측값도 별도 전처리 없이 XGBoost가 내부적으로 처리한다. 각 노드 분할 시 결측 샘플을 왼쪽과 오른쪽 중 **손실 감소가 더 큰 방향으로 자동 배정**한다.

```
필수 전처리:  없음 (결측값, 이상치, 스케일 모두 내부 처리)
권장 전처리:  범주형 변수 인코딩 (label encoding or ordinal encoding)
              → 원-핫 인코딩은 희소 행렬을 만들지만 DMatrix가 이를 효율적으로 처리
```

---

### XGBoost 기능 전체 정리

|기능|설명|
|---|---|
|**Boosting**|Gradient Boosting 기반. 순차적으로 잔차를 학습|
|**2차 미분 (뉴턴법)**|Gradient + Hessian으로 분할 이득 계산. 더 정확하지만 대량 데이터에 부담|
|**Regularization**|L1(alpha), L2(lambda) 정규화 내장. 과적합 방지|
|**Learning Rate**|각 트리의 기여도 축소. 낮을수록 과적합 방지, 트리 수 늘려야|
|**Subsampling**|`subsample`: 행 샘플링 / `colsample_bytree`: 열 샘플링|
|**Pruning**|트리를 최대 깊이까지 키운 뒤 손실 감소가 없는 가지를 제거|
|**Early Stopping**|검증 성능 개선이 없으면 학습 조기 종료 → 최적 트리 수 자동 결정|
|**Cross Validation**|`xgb.cv()`로 내장 교차검증 지원|
|**Missing Value**|결측값 자동 처리. 각 노드 분할 시 결측값을 최적 방향으로 배정|
|**Parallel Processing**|트리 내 후보 분할점 계산을 병렬화. 트리 간은 여전히 순차|
|**Cache Optimization**|CPU 캐시를 활용한 데이터 접근 최적화|
|**Sparse Data**|희소 행렬(0이 많은 데이터) 처리 최적화. 원-핫 인코딩 결과에 유리|
|**Out-of-Core learning**|메모리에 안 들어오는 대용량 데이터를 청크 단위로 읽어 학습|

---

## 2. Booster 종류

XGBoost는 내부적으로 사용하는 약한 학습기(Booster)를 선택할 수 있다.

### gbtree (기본값)

결정 트리를 약한 학습기로 사용한다. 비선형 관계를 포착할 수 있어 대부분의 경우 가장 좋은 성능을 낸다.

**gbtree 규제 파라미터**:

|파라미터|설명|권장 범위|
|---|---|---|
|`max_depth`|각 트리의 최대 깊이|3~10 (기본 6)|
|`min_child_weight`|리프 노드의 최소 헤시안 합|1~10 (클수록 보수적)|
|`gamma` (`min_split_loss`)|분할에 필요한 최소 손실 감소량|0~5 (클수록 분할 억제)|

`min_child_weight`는 scikit-learn의 `min_samples_leaf`와 비슷하지만 샘플 수 대신 **헤시안의 합**을 기준으로 한다. 분류에서는 $H = p(1-p)$이므로 클래스 불균형 데이터에서 소수 클래스의 노드 분할을 억제하는 효과가 있다.

### gblinear

선형 모델을 약한 학습기로 사용한다. 각 부스팅 라운드에서 선형 모델의 계수를 업데이트한다.

**gblinear 규제 파라미터**:

|파라미터|설명|
|---|---|
|`lambda` (L2)|계수의 크기를 0에 가깝게 축소|
|`alpha` (L1)|일부 계수를 정확히 0으로 만들어 변수 선택 효과|
|`gamma` (L0)|비제로 계수 수에 패널티 (희소성 강제)|

선형 데이터에서는 gbtree보다 빠르고 해석이 쉽지만, 비선형 관계를 포착하지 못한다.

### dart (Dropouts meet Multiple Additive Regression Trees)

딥러닝의 드롭아웃(Dropout) 아이디어를 트리 앙상블에 적용한 방법이다. 기존에 추가된 트리 중 일부를 무작위로 제외(드롭)하고 새 트리를 학습해서 특정 트리에 지나치게 의존하는 것을 방지한다.

```
일반 GB:  트리1 + 트리2 + 트리3 + 트리4 추가
DART:     트리1 + 트리2 + (트리3 제외) + 트리4  ← 일부 드롭
          → 남은 트리에 의존하지 않고 트리4가 독립적으로 학습
```

|특징|설명|
|---|---|
|과적합 방지|드롭아웃으로 앞쪽 트리 의존도 감소|
|느린 학습|드롭된 트리를 다시 스케일링하는 과정 필요|
|`early_stopping` 주의|DART는 조기 종료와 함께 사용할 때 성능이 불안정할 수 있음|

```python
import xgboost as xgb

# dart booster 사용
model_dart = xgb.XGBClassifier(
    booster='dart',
    rate_drop=0.1,      # 각 라운드에서 드롭할 트리 비율
    skip_drop=0.5,      # 드롭을 건너뛸 확률 (0이면 항상 드롭)
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
```

---

## 3. Objective (목적 함수)

목적 함수는 **모델이 무엇을 최소화할지**를 결정한다. 문제 유형에 맞는 목적 함수를 선택해야 한다.

### 회귀 (Regression)

|목적 함수|수식|특징|
|---|---|---|
|`reg:squarederror`|$\frac{1}{2}(y - \hat{y})^2$|기본값. MSE. 이상치에 민감|
|`reg:squaredlogerror`|$\frac{1}{2}(\log(y+1) - \log(\hat{y}+1))^2$|상대 오차. 값의 범위가 클 때|
|`reg:absoluteerror`|$|y - \hat{y}|
|`reg:tweedie`|Tweedie 분포|보험, 의료 등 0이 많은 데이터|

### 분류 (Classification)

|목적 함수|출력|사용 경우|
|---|---|---|
|`binary:logistic`|확률 (0~1)|이진 분류, 확률 출력 필요|
|`binary:hinge`|0 또는 1|이진 분류, SVM 스타일|
|`multi:softmax`|클래스 레이블|다중 분류|
|`multi:softprob`|클래스별 확률|다중 분류 + 확률 필요|

```python
from xgboost import XGBClassifier, XGBRegressor

# 이진 분류
clf = XGBClassifier(objective='binary:logistic', eval_metric='auc')

# 다중 분류
multi_clf = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss'
)

# 회귀
reg = XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
```

---

## 4. 과적합 방지 방법

XGBoost에서 과적합을 제어하는 방법은 크게 네 가지다.

### Cross-Validation

xgboost 내장 `cv()` 함수로 교차검증을 수행해 최적 파라미터를 찾는다.

```python
import xgboost as xgb
import pandas as pd

dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

# 내장 CV: 트리 수별 성능 추이 확인
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    nfold=5,
    stratified=True,
    early_stopping_rounds=20,
    seed=42,
    verbose_eval=False
)

print(cv_results.tail())
best_n = cv_results['test-auc-mean'].idxmax() + 1
print(f'최적 트리 수: {best_n}')
print(f'최고 CV AUC: {cv_results["test-auc-mean"].max():.4f}')
```

### Early Stopping

검증 데이터의 성능이 일정 라운드 동안 개선되지 않으면 학습을 자동으로 중단한다. 트리 수를 일일이 탐색하지 않아도 최적 지점을 자동으로 찾아준다.

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,
                                              test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=2000,      # 충분히 크게 설정 (early stopping이 알아서 자름)
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='auc'
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,  # 50라운드 연속 개선 없으면 중단
    verbose=100                 # 100라운드마다 출력
)

print(f'실제 사용된 트리 수: {model.best_iteration + 1}')
print(f'최고 검증 AUC: {model.best_score:.4f}')
```

### Pruning (가지치기)

XGBoost의 기본 전략은 **depth-first pruning**이다. 트리를 `max_depth`까지 끝까지 키운 뒤, 후진으로 올라가며 `gamma`(min_split_loss) 기준을 만족하지 못하는 가지를 잘라낸다.

```
일반 DT의 가지치기: 만들면서 즉시 분할 중단 (greedy)
XGBoost Pruning:   끝까지 키운 다음 뒤에서부터 정리 (depth-first)

→ 더 전역적인 관점에서 불필요한 분할을 제거할 수 있음
```

`gamma`가 0이면 이득이 조금이라도 있으면 분할, 클수록 더 큰 이득이 있어야만 분할한다.

### Regularization (정규화)

|파라미터|정규화 유형|효과|
|---|---|---|
|`alpha`|L1|일부 리프 가중치를 정확히 0으로 만들어 희소성 증가|
|`lambda`|L2|모든 리프 가중치를 0에 가깝게 축소 (기본값 1)|
|`gamma`|분할 최소 이득|이득이 gamma 이상일 때만 분할 허용|

```python
model = XGBClassifier(
    reg_alpha=0.1,    # L1: 작은 영향은 0으로
    reg_lambda=2.0,   # L2: 전체적으로 축소 (기본 1)
    gamma=0.1,        # 분할 이득 최소값
    max_depth=5,
    random_state=42
)
```

---

## 5. XGBoost의 두 가지 API

XGBoost는 사용 방식이 두 가지로 나뉜다.

### Booster (low-level API)

XGBoost 고유의 저수준 API다. `DMatrix`라는 XGBoost 전용 데이터 구조를 사용한다. scikit-learn 파이프라인과 직접 연결은 안 되지만 더 세밀한 제어가 가능하고 처음부터 고유 기능(내장 CV, Booster.cv 등)에 접근하기 좋다.

```python
import xgboost as xgb
import numpy as np

# DMatrix: XGBoost 내부 최적화 데이터 구조
# - 희소 행렬 자동 처리
# - 캐시 효율 최적화
# - 메모리 레이아웃 최적화
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_names)

params = {
    'booster':      'gbtree',
    'objective':    'binary:logistic',
    'eval_metric':  'auc',
    'max_depth':    6,
    'learning_rate': 0.1,
    'subsample':    0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':    0.1,
    'reg_lambda':   1.0,
    'seed':         42
}

evals = [(dtrain, 'train'), (dtest, 'eval')]
booster = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=50
)

# 예측 (확률값 반환)
y_prob = booster.predict(dtest)
y_pred = (y_prob >= 0.5).astype(int)

# 모델 저장/불러오기
booster.save_model('xgb_model.json')
loaded = xgb.Booster()
loaded.load_model('xgb_model.json')
```

### XGBClassifier / XGBRegressor (sklearn API)

Booster를 sklearn 인터페이스로 감싼 래퍼(wrapper)다. `fit()`, `predict()`, `score()` 같은 sklearn 표준 메서드를 지원해서 **Pipeline, GridSearchCV, cross_val_score** 등과 바로 연결된다.

---

### GPU 및 분산 처리 연동

XGBoost는 단일 머신을 넘어 GPU와 분산 처리 환경에서도 사용할 수 있다.

**GPU 사용 (CUDA)**  
`device='cuda'`를 설정하면 GPU의 CUDA 코어를 활용해 학습 속도를 크게 높일 수 있다. 특히 대량 데이터에서 CPU 대비 수십 배 빠른 경우도 있다.

```python
# GPU 사용 (CUDA가 설치된 환경)
model_gpu = XGBClassifier(
    device='cuda',    # GPU 사용 (구버전은 tree_method='gpu_hist')
    n_estimators=500,
    random_state=42
)

# CPU 병렬 처리
model_cpu = XGBClassifier(
    n_jobs=-1,        # 전체 CPU 코어 사용
    n_estimators=500,
    random_state=42
)
```

**분산 처리 프레임워크 연동**  
대용량 데이터를 단일 머신에서 처리하기 어려울 때 XGBoost는 분산 처리 프레임워크와 연동된다.

|프레임워크|특징|XGBoost 연동|
|---|---|---|
|**Hadoop (HDFS)**|파일 기반 분산 저장. 한 번 쓰면 수정/삭제 불가|분산 학습 데이터 저장소로 활용|
|**Spark**|인메모리 기반 고속 분산 처리|`xgboost.spark` 모듈로 Spark DataFrame 직접 학습|
|**Ray**|AI/ML 워크로드를 여러 하드웨어에서 병렬 제어|`ray.train`으로 분산 XGBoost 학습|

```python
# Spark와 XGBoost 연동 예시
from xgboost.spark import SparkXGBClassifier

spark_clf = SparkXGBClassifier(
    features_col='features',
    label_col='label',
    num_workers=4       # Spark 워커 수
)
model = spark_clf.fit(train_df)   # Spark DataFrame 입력
```

Spark는 HDFS에 저장된 대용량 데이터를 인메모리로 읽어 빠르게 처리하고, XGBoost는 그 위에서 분산 학습을 수행한다. Ray는 머신 간 분산뿐 아니라 하이퍼파라미터 탐색도 병렬화할 수 있어 ML 파이프라인 전체를 분산 처리하는 데 유리하다.

```python
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score

# sklearn 파이프라인과 통합
pipe = make_pipeline(
    StandardScaler(),
    XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
)

# GridSearchCV로 하이퍼파라미터 탐색
param_grid = {
    'xgbclassifier__max_depth':    [3, 5, 7],
    'xgbclassifier__learning_rate': [0.05, 0.1, 0.2],
    'xgbclassifier__n_estimators': [100, 200]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print(f'최적 파라미터: {grid.best_params_}')
print(f'CV AUC: {grid.best_score_:.4f}')
print(f'테스트 AUC: {grid.score(X_test, y_test):.4f}')
```

---

## 6. GridSearch + Pipeline 통합 구조

슬라이드의 구조를 코드로 표현하면 이렇다.

```
GridSearch
  └─ Pipeline
       ├─ 전처리 단계
       ├─ DMatrix (XGBoost 내부 변환)
       └─ Booster.cv / XGBClassifier
```

실제로는 sklearn API를 쓸 때 DMatrix 변환은 내부적으로 자동 처리된다.

```python
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

# 완전한 파이프라인
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # 결측값 처리
    ('scaler',  StandardScaler()),                   # 정규화 (XGBoost에 필수는 아니나 권장)
    ('model',   XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        verbosity=0,
        random_state=42
    ))
])

# RandomizedSearch: GridSearch보다 효율적
param_dist = {
    'model__n_estimators':      randint(100, 500),
    'model__max_depth':         randint(3, 10),
    'model__learning_rate':     loguniform(0.01, 0.3),
    'model__subsample':         loguniform(0.5, 1.0),
    'model__colsample_bytree':  loguniform(0.5, 1.0),
    'model__reg_alpha':         loguniform(1e-3, 10),
    'model__reg_lambda':        loguniform(1e-3, 10),
    'model__gamma':             loguniform(1e-3, 5)
}

search = RandomizedSearchCV(
    pipe, param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)
print(f'최적 파라미터: {search.best_params_}')
```

---

## 7. 주요 하이퍼파라미터 전체 정리

```python
XGBClassifier(
    # ── 트리 구조 ──────────────────────────────
    n_estimators=100,         # 트리 수 (early_stopping으로 자동 결정 권장)
    max_depth=6,              # 트리 최대 깊이 (3~10)
    min_child_weight=1,       # 리프의 최소 헤시안 합 (클수록 보수적)

    # ── 학습 속도 ──────────────────────────────
    learning_rate=0.1,        # 각 트리의 기여도 (낮을수록 n_estimators 늘려야)

    # ── 샘플링 (과적합 방지) ────────────────────
    subsample=0.8,            # 행 샘플링 비율 (0.5~1.0)
    colsample_bytree=0.8,     # 트리 단위 열 샘플링
    colsample_bylevel=1.0,    # 깊이 단위 열 샘플링
    colsample_bynode=1.0,     # 노드 단위 열 샘플링

    # ── 정규화 ─────────────────────────────────
    reg_alpha=0.0,            # L1 정규화
    reg_lambda=1.0,           # L2 정규화 (기본값)
    gamma=0.0,                # 분할 최소 이득 (0이면 이득만 있으면 분할)

    # ── 목적 함수 ──────────────────────────────
    objective='binary:logistic',  # 목적 함수
    eval_metric='auc',            # 평가 지표

    # ── 실행 설정 ──────────────────────────────
    booster='gbtree',         # 'gbtree', 'gblinear', 'dart'
    n_jobs=-1,                # 전체 코어 사용
    random_state=42,
    verbosity=0               # 출력 안 함
)
```

---

## 8. LightGBM — XGBoost의 속도 문제를 해결한 후속

XGBoost가 느린 이유는 각 분할에서 모든 후보 분할점을 탐색하기 때문이다. LightGBM은 이를 해결하는 두 가지 핵심 기법을 도입했다.

**Leaf-wise 분할 (XGBoost는 Level-wise)**:

```
XGBoost (Level-wise):   같은 깊이의 모든 노드를 분할
LightGBM (Leaf-wise):   손실 감소가 가장 큰 리프 하나만 분할

→ 같은 리프 수라도 LightGBM이 더 불균형한 트리를 만들지만
  더 중요한 곳에 집중하므로 성능이 높음
```

**GOSS (Gradient-based One-Side Sampling)**:  
기울기가 큰 샘플(아직 잘 못 맞춘 샘플)은 전부 유지하고, 기울기가 작은 샘플(이미 잘 맞춘 샘플)은 일부만 사용한다.

```python
# pip install lightgbm
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,             # 리프 수 직접 지정 (XGBoost와 다른 방식)
    max_depth=-1,              # -1이면 제한 없음
    min_child_samples=20,      # 리프 최소 샘플 수
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42
)
lgbm.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(100)])
```

|구분|XGBoost|LightGBM|
|---|---|---|
|분할 방식|Level-wise|Leaf-wise|
|속도|보통|매우 빠름|
|메모리|많음|적음|
|범주형 처리|전처리 필요|자동 처리|
|소량 데이터|안정적|과적합 위험|

---

## 9. 실전 파이프라인: 데이터 로드부터 제출까지

```python
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# ── 1. 데이터 준비 ──────────────────────────────────────
# (예시: sklearn 내장 유방암 데이터)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ── 2. 파이프라인 구성 ──────────────────────────────────
pipe = make_pipeline(
    StandardScaler(),
    XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        gamma=0.05,
        n_jobs=-1,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )
)

# ── 3. 교차검증 성능 확인 ──────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
print(f'CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# ── 4. 최종 학습 및 평가 ──────────────────────────────
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = pipe.predict(X_test)

print(f'\n테스트 AUC: {roc_auc_score(y_test, y_prob):.4f}')
print('\n분류 리포트:')
print(classification_report(y_test, y_pred))

# ── 5. 모델 저장 ────────────────────────────────────────
joblib.dump(pipe, 'xgb_pipeline.pkl')
```