---
title: 026 의사결정 트리
tag:
  - 국비교육
  - R
description: 260506 수업 내용 정리
---

# 의사결정 트리 (Decision Tree)와 분류 모델 평가

---

## 1. 의사결정 트리란

데이터를 **규칙(Rule) 기반으로 반복해서 쪼개면서** 분류하는 모델이다.  
"꽃잎 길이가 2.45cm 이하인가?" 같은 조건을 순서대로 적용해 최종 분류에 도달한다.

회귀 분석과 달리 선형 관계를 가정하지 않고, 데이터를 있는 그대로 분할하는 방식이라 **직관적으로 해석하기 쉽다.**

---

## 2. 트리 구조 용어

```
              [Root Node]          ← 전체 데이터, 첫 번째 분할 기준
             /           \
        [Node]           [Node]    ← 중간 노드 (parent → child)
        /    \           /    \
    [Leaf]  [Leaf]   [Leaf]  [Leaf] ← 최종 분류 결과
```

|용어|설명|
|---|---|
|Root Node|트리의 시작점. 가장 중요한 변수로 첫 분할|
|Node|조건에 따라 데이터를 분기하는 지점|
|Leaf (Terminal Node)|더 이상 분할하지 않는 끝 노드. 최종 예측값|
|Depth|루트에서 해당 노드까지의 깊이|
|Parent / Child|분기 전 노드 / 분기 후 노드|

### Iris 데이터 예시 (이미지 기준)

```
꽃잎 길이 <= 2.45?
├─ True  → setosa (gini=0.0, 순도 100%)
└─ False → 꽃잎 너비 <= 1.75?
           ├─ True  → versicolor (gini=0.168)
           └─ False → virginica  (gini=0.043)
```

각 노드에 표시되는 정보:

- **gini**: 해당 노드의 불순도 (낮을수록 잘 분류됨)
- **samples**: 해당 노드에 포함된 데이터 수
- **value**: 각 클래스의 데이터 수 `[setosa, versicolor, virginica]`
- **class**: 다수결로 결정된 예측 클래스

---

## 3. 의사결정 트리의 특징

### 장점

- **비모수 검정**: 선형성, 정규성, 등분산성 가정이 필요 없다
- **Whitebox 모델**: 규칙이 명확히 보여서 결과를 설명하기 쉽다 (rule-based)
- **정규화 불필요**: 변수 스케일에 영향을 받지 않는다
- **계산 비용이 낮다**: 대규모 데이터에서도 비교적 빠르게 학습된다
- **범주형/연속형 모두 처리 가능**

### 단점

- **유의 수준 판단 기준 없음**: 통계적 유의성을 검증하는 p-value 같은 기준이 없어 추론(inference) 기능이 없다
- **과적합(Overfitting) 위험**: 트리가 깊어질수록 학습 데이터에 지나치게 맞춰진다. 새로운 데이터에 대한 성능이 떨어질 수 있다
- **불안정성**: 데이터가 조금만 바뀌어도 트리 구조가 크게 달라질 수 있다

---

## 4. 불순도와 분할 기준

트리는 각 분기에서 **불순도(Impurity)를 가장 많이 줄이는 변수와 기준값**을 선택한다.

### 지니 불순도 (Gini Impurity)

$$Gini = 1 - \sum_{k} p_k^2$$

- $p_k$: 노드에서 클래스 $k$의 비율
- 값이 0이면 완전히 순수한 노드 (한 클래스만 존재)
- 값이 클수록 여러 클래스가 섞여 있음

예시:

- [50, 0, 0] → Gini = 0.0 (완전 순수)
- [0, 50, 50] → Gini = 0.5 (두 클래스 반반)

### 엔트로피 (Entropy) / 정보이득 (Information Gain)

$$H(Q_m) = -\sum_{k} p_{mk} \log(p_{mk})$$

- 정보이론에서 온 개념으로, 불확실성의 정도를 나타낸다
- 엔트로피가 높을수록 불순도가 높다
- 분할 전후의 엔트로피 차이 = **정보이득(Information Gain)**
- 정보이득이 가장 큰 분할을 선택한다

|기준|계산 방식|특징|
|---|---|---|
|Gini|$1 - \sum p_k^2$|계산이 빠름. 기본값|
|Entropy|$-\sum p_k \log p_k$|계산 더 복잡. 균형 잡힌 분할 선호|

---

## 5. 과적합 방지: 하이퍼파라미터와 가지치기

트리가 너무 깊어지면 학습 데이터에만 과적합된다. 이를 제어하는 하이퍼파라미터들이다.

|파라미터|설명|
|---|---|
|`max_depth`|트리의 최대 깊이. 작을수록 단순한 모델|
|`max_leaf_nodes`|리프 노드의 최대 개수|
|`max_features`|각 노드에서 분할에 사용할 특성의 최대 수|
|`min_samples_split`|분할되기 위해 노드가 가져야 하는 최소 샘플 수|
|`min_samples_leaf`|리프 노드가 가져야 하는 최소 샘플 수|
|`min_weight_fraction_leaf`|`min_samples_leaf`와 같지만 가중치 기준 비율로 설정|

```r
library(rpart)
model <- rpart(Species ~ ., data = iris,
               control = rpart.control(
                 maxdepth = 3,
                 minsplit = 10,
                 minbucket = 5
               ))
```

### 가지치기 (Pruning)

트리를 먼저 끝까지 키운 다음, 불필요한 가지를 잘라내는 방법이다.  
교차검증(Cross-Validation)을 통해 최적의 트리 크기를 결정한다.

---

## 6. 랜덤 포레스트 (Random Forest)

의사결정 트리의 과적합 단점을 보완하기 위해 **여러 트리를 동시에 만들어 다수결로 예측**하는 앙상블 방법이다.

### 핵심 키워드

|개념|설명|
|---|---|
|**Bootstrap**|원본 데이터에서 복원추출로 여러 개의 서로 다른 학습 데이터셋을 만드는 방법|
|**Bagging**|Bootstrap + Aggregating. 여러 모델의 예측을 평균 또는 다수결로 합산|
|**Ensemble**|여러 모델을 결합해 단일 모델보다 성능을 높이는 전략|

### 랜덤 포레스트가 강한 이유

- 각 트리가 다른 Bootstrap 샘플로 학습 → 트리들이 서로 다른 패턴을 포착
- 각 분기에서 `max_features`개의 변수만 무작위로 사용 → 트리 간 상관성 감소
- 여러 트리의 결과를 합산 → 과적합 완화, 일반화 성능 향상

```r
library(randomForest)
model_rf <- randomForest(Species ~ ., data = iris, ntree = 500)
importance(model_rf)     # 변수 중요도
varImpPlot(model_rf)     # 시각화
```

---

## 7. 분류 모델 평가: 혼동 행렬 (Confusion Matrix)

분류 모델의 성능을 평가할 때는 단순 정확도만으로 부족한 경우가 많다.  
혼동 행렬은 예측과 실제의 모든 조합을 표로 정리한다.

||실제 Positive|실제 Negative|
|---|---|---|
|**예측 Positive**|TP (True Positive)|FP (False Positive)|
|**예측 Negative**|FN (False Negative)|TN (True Negative)|

- **TP**: 실제 양성 → 양성으로 올바르게 예측
- **TN**: 실제 음성 → 음성으로 올바르게 예측
- **FP**: 실제 음성 → 양성으로 잘못 예측 (오탐, 1종 오류)
- **FN**: 실제 양성 → 음성으로 잘못 예측 (미탐, 2종 오류)

---

## 8. 분류 평가 지표

### 정확도 (Accuracy)

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

전체 예측 중 맞은 비율이다. 직관적이지만 **클래스 불균형** 문제가 있을 때는 신뢰할 수 없다.  
예를 들어 암 환자가 1%인 데이터에서 "전부 정상"이라고 예측해도 정확도 99%가 나온다.

### 정밀도 (Precision)

$$Precision = \frac{TP}{TP + FP}$$

**양성으로 예측한 것 중 실제로 양성인 비율**이다.  
FP(오탐)을 줄이는 것이 중요할 때 사용한다.  
예: 스팸 필터 — 정상 메일을 스팸으로 잘못 분류하면 안 된다

### 재현율 (Recall) / 민감도 (Sensitivity)

$$Recall = \frac{TP}{TP + FN}$$

**실제 양성 중 양성으로 올바르게 예측한 비율**이다.  
FN(미탐)을 줄이는 것이 중요할 때 사용한다.  
예: 암 진단 — 실제 암 환자를 정상으로 놓치면 안 된다

### 특이도 (Specificity)

$$Specificity = \frac{TN}{TN + FP}$$

**실제 음성 중 음성으로 올바르게 예측한 비율**이다. 민감도의 반대 개념이다.

### 오분류율 (Error Rate)

$$Error\ Rate = 1 - Accuracy = \frac{FP + FN}{TP + TN + FP + FN}$$

### F1 Score

정밀도와 재현율의 **조화평균**이다. 둘의 균형을 하나의 수치로 나타낸다.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

정밀도와 재현율이 모두 높아야 F1 Score도 높아진다.

### 지표 선택 가이드

|상황|중요 지표|이유|
|---|---|---|
|암 진단, 사기 탐지|Recall|FN(미탐)이 치명적|
|스팸 필터, 추천 시스템|Precision|FP(오탐)이 불편|
|클래스 불균형이 심함|F1 Score|정확도가 왜곡됨|
|클래스 균형이 맞음|Accuracy|직관적으로 해석 가능|

```r
library(caret)
confusionMatrix(predicted, actual)
# Accuracy, Sensitivity, Specificity, Precision 등 한 번에 출력
```

---

## 9. 전체 흐름 정리

```
데이터 준비
   ↓
트리 학습 (불순도 기반 분할)
   ↓
과적합 확인
   ├─ 깊이/샘플 수 제한 (하이퍼파라미터)
   └─ 앙상블로 확장 (Random Forest)
   ↓
모델 평가 (혼동 행렬)
   ├─ Accuracy: 전체 정확도
   ├─ Precision: 오탐 관리
   ├─ Recall: 미탐 관리
   └─ F1: 균형 지표
   ↓
목적에 맞는 지표로 최종 모델 선택
```