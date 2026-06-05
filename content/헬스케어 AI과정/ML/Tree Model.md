---
title: 037 tree-model
tag:
  - 헬스케어 ai
  - ML
description: 260604 수업 내용 정리
---

# 의사결정 트리(DT)와 앙상블 학습

---

## 1. 의사결정 트리 (Decision Tree)

### 구조와 용어

트리는 데이터를 **조건에 따라 반복적으로 분기**해서 최종 예측에 도달하는 구조다.

```
                [Root Node]              ← 전체 데이터, 첫 번째 분할 기준
               꽃잎 길이 <= 2.45
              /              \
         True                 False
           ↓                    ↓
    [Leaf: setosa]         [Internal Node]    ← parent
      gini=0.0              꽃잎 너비 <= 1.75
      samples=50            /              \
                       True                False
                         ↓                  ↓
                  [Leaf: versicolor]  [Leaf: virginica]  ← child
                    gini=0.168          gini=0.043
                    samples=54          samples=46
```

|용어|설명|
|---|---|
|**Root Node**|트리의 시작점. 가장 정보이득이 큰 특성으로 첫 분할|
|**Internal Node**|중간 분기 노드. 조건에 따라 데이터를 두 방향으로 나눔|
|**Leaf Node**|끝 노드. 더 이상 분할하지 않고 최종 예측 결정|
|**Depth**|루트에서 해당 노드까지의 깊이. 클수록 복잡한 모델|

각 노드에 표시되는 정보:

- **gini**: 불순도 (0에 가까울수록 한 클래스만 존재하는 순수한 상태)
- **samples**: 해당 노드에 해당하는 데이터 수
- **value**: 클래스별 데이터 수 `[setosa, versicolor, virginica]`
- **class**: value에서 가장 많은 클래스 (다수결)

---

### Leaf Node가 학습하는 것 — 빈도에서 확률로

Leaf Node는 **불순도가 가장 낮은 노드**다. 트리가 실제로 학습하는 것은 각 Leaf에 도달한 데이터의 **빈도(Frequency)** 다. 이 빈도를 단순히 비율로 바꾼 것이 트리의 확률 출력이다.

```
Leaf Node: value = [0, 49, 5]  (총 54개)

빈도 → 의사 확률(Pseudo-Probability):
  P(versicolor) = 49/54 ≈ 0.907
  P(virginica)  =  5/54 ≈ 0.093
```

이 확률은 훈련 데이터의 빈도를 그대로 반영한 "의사 확률"이다. 실제 세계에서도 이 비율이 그대로 성립한다는 보장이 없기 때문에, 트리 모델의 확률 출력은 **과신(overconfident)** 하는 경향이 있다.

이를 실제 확률에 가깝게 교정하는 과정이 **Calibration(보정)** 이다. Calibration Curve가 대각선에 가까울수록 잘 보정된 모델이다.

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Platt Scaling: 시그모이드 함수로 보정 (데이터 적을 때)
cal_sigmoid  = CalibratedClassifierCV(dt, method='sigmoid', cv=5)
# Isotonic: 단조 증가 함수로 보정 (데이터 많을 때 더 정확)
cal_isotonic = CalibratedClassifierCV(dt, method='isotonic', cv=5)

cal_sigmoid.fit(X_train, y_train)
cal_isotonic.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(8, 6))
for clf, name in [(dt, 'DT (보정 전)'),
                  (cal_sigmoid, 'DT + Sigmoid'),
                  (cal_isotonic, 'DT + Isotonic')]:
    CalibrationDisplay.from_estimator(clf, X_test, y_test, ax=ax, name=name)
plt.title('Calibration Curve (대각선에 가까울수록 신뢰할 수 있는 확률)')
plt.show()
```

---

### 불순도와 분할 기준

트리는 각 분기에서 **정보이득(Information Gain)이 가장 큰 분할**을 선택한다.

$$IG = \text{불순도}(\text{부모}) - \sum_i \frac{n_i}{n} \cdot \text{불순도}(\text{자식}_i)$$

불순도를 측정하는 방법은 두 가지가 있다.

#### 지니 불순도 (Gini Impurity)

$$Gini = 1 - \sum_{k} p_k^2$$

계산이 빠르고 scikit-learn의 기본값이다. 값이 0이면 한 클래스만 존재하는 완전히 순수한 상태, 0.5가 이진 분류에서 최대 불순도다.

```
value=[50, 0, 0]: Gini = 1 - (1.0² + 0² + 0²) = 0.0  ← 완전 순수
value=[0, 50, 50]: Gini = 1 - (0² + 0.5² + 0.5²) = 0.5  ← 최대 혼합
```

#### 엔트로피 (Entropy)

$$H = -\sum_{k} p_k \log_2 p_k$$

정보 이론 기반의 측정 방식이다. 지니보다 계산이 느리지만 더 균형 잡힌 분할을 선호한다.

---

### DT의 특징

DT는 결과를 if-else 규칙으로 해석할 수 있는 **화이트박스 모델**이다. 스케일 정규화가 필요 없고 범주형·연속형 변수를 모두 처리할 수 있다.

그러나 두 가지 주의할 점이 있다. 첫째, **입력 순서에 따라 결과가 달라질 수 있다**. 같은 데이터라도 특성 순서가 바뀌거나 동률이 발생하면 다른 분할을 선택할 수 있어 `random_state`를 고정해서 재현성을 확보해야 한다. 둘째, 제한을 두지 않으면 **과적합이 매우 심하게 발생한다**. 깊이 제한이 없는 트리는 훈련 데이터를 완벽하게 암기해버린다.

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(
    criterion='gini',         # 'gini', 'entropy', 'log_loss'
    max_depth=5,              # 과적합 방지: 깊이 제한
    min_samples_split=10,     # 이 수 미만이면 분할 안 함
    min_samples_leaf=5,       # 리프가 이 수 미만이면 분할 안 함
    max_leaf_nodes=None,      # 리프 노드 최대 개수
    max_features=None,        # 각 분할에서 고려할 특성 수
    random_state=42
)
dt.fit(X_train, y_train)

# 화이트박스: 학습된 규칙 출력
print(export_text(dt, feature_names=feature_names))

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, ax=ax, max_depth=3)
plt.show()
```

---

## 2. Random Forest (RF)

### 핵심 아이디어

DT 여러 개를 각각 다른 데이터로 학습시키고 결과를 합쳐서 판단한다. 분류 문제에서는 각 트리의 예측을 **투표(다수결)**로, 회귀 문제에서는 **평균**으로 최종값을 결정한다.

```
분류:  트리1→A, 트리2→A, 트리3→B, 트리4→A  →  A (3:1)
회귀:  트리1→5.2, 트리2→4.8, 트리3→5.5      →  5.17 (평균)
```

단순히 DT를 여러 개 쓰는 것만으로는 의미가 없다. 모든 트리가 같은 데이터로 학습하면 거의 같은 결과를 내기 때문에 앙상블 효과가 없다. 핵심은 **서로 다르게 만드는 것**이고, 이를 위해 두 가지 무작위성을 도입한다.

---

### RF가 "Random"인 이유

RF는 두 가지 무작위성 때문에 이름에 "Random"이 붙는다.

**첫 번째: 복원추출 샘플링 (Bootstrapping)**  
원본 데이터에서 복원추출로 각 트리마다 다른 훈련셋을 만든다. 어떤 샘플은 여러 번 뽑히고, 어떤 샘플은 한 번도 뽑히지 않는다. 이 때문에 각 트리가 서로 다른 패턴을 학습한다.

**두 번째: 열(Feature) 무작위 선택**  
각 노드를 분할할 때 전체 특성 중 일부만 무작위로 선택해서 후보로 사용한다. 분류에서는 보통 $\sqrt{p}$개, 회귀에서는 $p/3$개를 선택한다. 특정 강한 특성이 모든 트리를 지배하는 것을 막아서 트리 간 상관성을 낮추고 앙상블 효과를 극대화한다.

---

### RF와 일반 Bagging의 차이

RF와 Bagging 모두 복원추출을 사용하지만, **열 무작위 선택 유무**가 결정적 차이다.

|구분|Bagging|Random Forest|
|---|---|---|
|복원추출 샘플링|✅|✅|
|열 무작위 선택|❌ (전체 특성 사용)|✅ ($\sqrt{p}$개 선택)|
|기본 모델|어떤 모델이든 가능|반드시 Decision Tree|
|트리 간 상관성|높음|낮음|

Bagging은 각 분할에서 전체 특성을 후보로 쓰기 때문에 강한 특성이 있으면 모든 트리가 비슷해진다. RF의 열 무작위 선택이 이 문제를 해결한다.

---

### RF의 편향-분산 특성

RF는 **분산(Variance)을 줄이는 데 강하지만 Bias는 줄이지 못한다**.

여러 트리 예측을 평균하면 분산이 줄어드는 것은 수학적으로 보장된다. 독립적인 예측기 $k$개를 평균한 분산은 $\sigma^2/k$로 줄어든다. 하지만 각 트리가 체계적으로 같은 방향으로 틀리고 있다면(Bias가 있다면), 아무리 평균을 내도 그 편향은 사라지지 않는다.

이 때문에 Bagging/RF는 **과적합된 모델을 안정시키는 데** 효과적이고, Boosting은 **정확도를 높이는 데** 효과적이다. 서로 해결하는 방향이 다르다.

---

### Out-of-Bag (OOB) Score

복원추출 시 특정 샘플이 한 번도 선택되지 않을 확률은 $(1-1/n)^n \to e^{-1} \approx 36.8%$ 다. 즉, 각 트리마다 약 36.8%의 데이터는 학습에 참여하지 않는다. 이 데이터를 Out-of-Bag 샘플이라 하며, **별도의 검증 세트 없이 자동으로 성능을 평가**하는 데 사용할 수 있다.

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np, matplotlib.pyplot as plt

rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',   # 각 분할에서 √p개 특성 후보
    bootstrap=True,         # 복원추출 (기본값)
    oob_score=True,         # OOB 점수 활성화
    n_jobs=-1,              # 전체 코어 사용
    random_state=42
)
rf.fit(X_train, y_train)

print(f'OOB 점수:    {rf.oob_score_:.4f}')  # 별도 검증셋 없이 평가
print(f'테스트 점수: {rf.score(X_test, y_test):.4f}')

# 변수 중요도 (각 특성이 분할 불순도 감소에 기여한 평균량)
importances = rf.feature_importances_
std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
idx = np.argsort(importances)[::-1][:15]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(idx)), importances[idx], yerr=std[idx], alpha=0.7)
ax.set_xticks(range(len(idx)))
ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right')
ax.set_title('Random Forest 변수 중요도')
plt.tight_layout()
plt.show()
```

---

## 3. AdaBoost (Adaptive Boosting)

### 핵심 아이디어

잘못 분류된 샘플에 **가중치를 높여** 다음 트리가 그 샘플에 더 집중하도록 만드는 방식이다. 깊이 1짜리 결정 스텀프처럼 약한 학습기(Weak Learner)를 순차적으로 쌓아 강한 학습기를 만든다.

```
라운드 1: 모든 샘플 동등 가중치 → 트리1 학습 → 틀린 샘플 가중치 증가
라운드 2: 가중치 높은 샘플에 집중 → 트리2 학습 → 틀린 샘플 가중치 증가
라운드 3: ...
최종 예측 = 각 트리 예측의 가중합 (성능 좋은 트리일수록 더 높은 가중치)
```

틀린 샘플에 집중하기 때문에 Bagging/RF와 반대로 **편향(Bias)을 줄이는** 방향으로 작동한다.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 깊이 1 결정 스텀프
    n_estimators=200,
    learning_rate=0.5,   # 작을수록 각 트리의 기여도가 낮아짐 → n_estimators 늘려야
    algorithm='SAMME',
    random_state=42
)
ada.fit(X_train, y_train)

# 잘 맞추는 트리일수록 가중치가 높게 부여됨
print(f'트리 가중치 (앞 5개): {ada.estimator_weights_[:5]}')
print(f'트리 오차  (앞 5개): {ada.estimator_errors_[:5]}')
```

---

## 4. Gradient Boosting (GB)

### 핵심 아이디어

AdaBoost가 샘플 가중치를 조절하는 방식이라면, GB는 **경사하강법을 잔차에 직접 적용**한다. 예측값과 실제값의 차이인 잔차를 다음 트리의 학습 목표로 삼아, 잔차가 발생한 구간을 보완하는 트리를 계속 추가해 나간다.

$$F_0(x) = \bar{y}$$ $$r_m = y - F_{m-1}(x) \quad \text{(잔차)}$$ $$F_m(x) = F_{m-1}(x) + \alpha \cdot h_m(x) \quad \text{(잔차 학습 트리 추가)}$$

$\alpha$(learning rate)가 작을수록 각 트리의 기여가 줄어 과적합을 방지할 수 있지만, 그만큼 트리 수를 늘려야 한다.

경사하강법과 연결하면, 파라미터 공간이 아닌 **함수 공간에서 손실 함수를 줄이는 방향**으로 새 트리를 추가하는 것이 GB의 수학적 본질이다. 덕분에 MSE 외에도 어떤 미분 가능한 손실 함수든 적용할 수 있다.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbt = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,       # 각 트리에 사용할 샘플 비율 (확률적 GBT, 과적합 방지)
    random_state=42
)
gbt.fit(X_train, y_train)

# 트리 수별 테스트 성능 확인 → 최적 트리 수 파악
staged = [accuracy_score(y_test, p) for p in gbt.staged_predict(X_test)]
best_n = staged.index(max(staged)) + 1
print(f'최적 트리 수: {best_n}, 최고 정확도: {max(staged):.4f}')
```

---

## 5. XGBoost (eXtreme Gradient Boosting)

scikit-learn과 별도 라이브러리지만 **scikit-learn 호환 인터페이스**를 제공해서 파이프라인과 GridSearchCV에 그대로 사용할 수 있다.

GB를 여러 측면에서 개선했다. 손실 함수를 1차 미분만 쓰는 GB와 달리 **2차 미분(헤시안)** 까지 활용해 분할 기준을 더 정확하게 결정한다. L1·L2 정규화가 내장되어 있고, 트리 내 분할 기준 탐색을 병렬화해 속도가 빠르며, 결측값도 자동으로 처리한다.

`n_jobs`는 사용할 CPU 코어 수를 지정한다. `-1`로 설정하면 전체 코어를 사용해 속도를 최대화한다.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,  # 각 트리에서 사용할 특성 비율
    reg_alpha=0.1,          # L1 정규화
    reg_lambda=1.0,         # L2 정규화
    n_jobs=-1,              # 전체 코어 사용
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,  # 20라운드 이상 개선 없으면 중단
        verbose=False)

print(f'최적 트리 수: {xgb.best_iteration}')
print(f'테스트 정확도: {xgb.score(X_test, y_test):.4f}')
```

|구분|GBT|XGBoost|
|---|---|---|
|미분 차수|1차|1차 + 2차 (헤시안)|
|정규화|없음|L1, L2 내장|
|병렬 처리|없음|있음|
|결측값 처리|전처리 필요|자동|

---

## 6. 앙상블 전략: Voting / Stacking / Blending

### Voting — 모델 결과를 직접 결합

여러 모델의 예측을 모아 최종 결정을 내리는 가장 단순한 방식이다.

**Hard Voting**은 각 모델이 예측한 클래스 레이블을 다수결로 결정한다.

**Soft Voting**은 각 모델의 확률값을 평균해서 가장 높은 확률의 클래스를 선택한다. 확률의 크기(신뢰도)를 반영하기 때문에 일반적으로 Hard Voting보다 성능이 좋다.

```
Hard: 모델1→A, 모델2→A, 모델3→B  →  A (2:1 다수결)

Soft: 모델1: P(A)=0.9, P(B)=0.1
      모델2: P(A)=0.8, P(B)=0.2
      모델3: P(A)=0.3, P(B)=0.7
              평균 → P(A)=0.67, P(B)=0.33  →  A

모델3이 "강하게 B"인지 "약하게 B"인지를 Soft는 구분할 수 있지만
Hard는 구분하지 못함
```

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

hard = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC())
    ],
    voting='hard', n_jobs=-1
)

soft = VotingClassifier(
    estimators=[
        ('lr',  LogisticRegression(max_iter=1000)),
        ('rf',  RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True))  # Soft는 predict_proba 필요
    ],
    voting='soft',
    weights=[1, 2, 1],  # 성능 좋은 모델에 더 높은 가중치
    n_jobs=-1
)

hard.fit(X_train, y_train)
soft.fit(X_train, y_train)
print(f'Hard Voting: {hard.score(X_test, y_test):.4f}')
print(f'Soft Voting: {soft.score(X_test, y_test):.4f}')
```

---

### Stacking — CV 기반 메타 모델 학습

여러 기본 모델(Level-0)의 예측 결과를 새로운 특성으로 삼아 **메타 모델(Level-1)이 다시 학습**하는 방식이다. 교차검증(CV)을 통해 Out-of-Fold 예측을 생성하기 때문에 데이터 누수 없이 메타 특성을 만들 수 있다.

```
K-Fold Stacking (K=5):

각 Fold에서 나머지 4개로 학습 → 해당 Fold 예측
→ 전체 훈련 데이터에 대한 OOF 예측 완성

이 OOF 예측값들을 새 특성으로 메타 모델 학습
→ 테스트는 Level-0 모델(전체 학습) → 메타 모델 순서로 통과
```

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

stacking = StackingClassifier(
    estimators=[
        ('lr',  LogisticRegression(max_iter=1000)),
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gbt', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ],
    final_estimator=LogisticRegression(),  # 메타 모델
    cv=5,                                   # CV로 OOF 예측 생성
    stack_method='predict_proba',
    n_jobs=-1
)
stacking.fit(X_train, y_train)
print(f'Stacking 정확도: {stacking.score(X_test, y_test):.4f}')
```

---

### Blending — CV 없이 홀드아웃만 사용

Stacking에서 CV를 제거하고 **고정된 홀드아웃 검증 세트**로만 메타 특성을 만드는 단순화 버전이다.

CV를 쓰면 모든 훈련 데이터를 메타 학습에 활용할 수 있지만(Stacking), 홀드아웃을 쓰면 검증 데이터가 학습에 쓰이지 않아 데이터 효율이 떨어진다(Blending). 대신 더 빠르고 구현이 단순하다.

```
Stacking:  CV(K-Fold) → 전체 훈련 데이터 활용 → 데이터 효율적
Blending:  홀드아웃   → 검증셋은 학습 못 함   → 더 단순·빠름
```

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 3분할
X_tr, X_temp, y_tr, y_temp = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
X_val, X_te, y_val, y_te   = train_test_split(X_temp,  y_temp,  test_size=0.5, random_state=42)

models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42)
]
for m in models:
    m.fit(X_tr, y_tr)

# 검증셋 예측값 → 메타 특성
val_meta  = np.column_stack([m.predict_proba(X_val)[:, 1] for m in models])
test_meta = np.column_stack([m.predict_proba(X_te)[:, 1]  for m in models])

meta = LogisticRegression()
meta.fit(val_meta, y_val)
print(f'Blending 정확도: {meta.score(test_meta, y_te):.4f}')
```

---

## 7. 전체 비교와 선택 가이드

### DT → RF → Ada → GB → XGBoost 발전 흐름

```
DT
  단일 트리, 해석 가능, 하지만 과적합 심함
     ↓ 여러 개 독립적으로 만들어 평균 (분산 감소)
RF
  복원추출 + 열 무작위 선택으로 다양한 트리 생성
  분산 감소에 탁월하지만 Bias는 줄이지 못함
     ↓ 순차적으로 오차를 보정 (편향 감소)
AdaBoost
  틀린 샘플 가중치 증가 → 약한 학습기 → 강한 학습기
     ↓ 경사하강법으로 수학적 일반화
GB
  잔차에 트리 추가, 어떤 손실 함수든 적용 가능
     ↓ 속도·정규화·결측값 처리 개선
XGBoost
  2차 미분 + 정규화 내장 + 병렬 처리 + n_jobs
```

### 앙상블 방법 비교

|방법|학습 순서|오차 감소|과적합 강인성|
|---|---|---|---|
|Bagging|병렬|분산 감소|강함|
|Random Forest|병렬|분산 감소|매우 강함|
|AdaBoost|순차|편향 감소|보통|
|GBT|순차|편향 감소|보통|
|XGBoost|순차|편향 감소|강함 (정규화)|
|Hard Voting|병렬|둘 다|모델 의존|
|Soft Voting|병렬|둘 다|모델 의존|
|Blending|병렬+순차|편향 감소|보통|
|Stacking|병렬+순차|편향 감소|강함|

### 상황별 선택

```
과적합이 문제 (High Variance)?     → Bagging, Random Forest
정확도를 높이고 싶다 (High Bias)?  → AdaBoost, GB, XGBoost
속도가 중요하다?                   → RF (n_jobs=-1 병렬)
최고 성능이 필요하다?              → XGBoost / LightGBM + Stacking
모델 결과를 단순하게 결합?         → Voting
서로 다른 모델을 계층적으로 결합?  → Stacking (정확) / Blending (빠름)
```