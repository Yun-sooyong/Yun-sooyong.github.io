---
title: 043 SVM
tag:
  - 헬스케어 ai
  - ML
description: 260612 수업 내용 정리
---

# SVM (Support Vector Machine)

SVM은 **Perceptron의 선형 분류기를 수학적으로 정교하게 발전시킨 모델**이다.

### Perceptron에서 SVM으로의 발전

퍼셉트론은 데이터를 선형으로 분리하는 결정 경계를 찾지만, 데이터를 분리하는 직선은 무한히 많을 수 있다. 이 중 어떤 것이 가장 좋은지는 고려하지 않는다.

```
퍼셉트론의 한계:
  ●  ●  /  ○  ○       ●  ●  \  ○  ○       ●  ●  |  ○  ○
  ●    /      ○       ●    \      ○       ●       |    ○
  ●   /   ○  ○        ●    \  ○  ○        ●       | ○  ○

  모두 유효한 결정 경계 → 퍼셉트론은 수렴하면 아무거나 선택
  → 어떤 경계가 새 데이터에 가장 잘 일반화할지 알 수 없음
```

SVM은 이 중에서 **마진이 가장 넓은 경계 하나**를 수학적으로 유일하게 결정한다. "가장 넓은 여유 공간을 확보하는 경계"를 정의하고, 이를 최적화 문제로 풀어 해를 구한다. 이것이 **마진 최대화(Margin Optimization)** 원리이며 SVM이 강한 일반화 성능을 가지는 이유다.

---

## 1. SVM의 핵심 개념

### 마진 최대화 (Margin Optimization)

결정 경계를 기준으로 양쪽 클래스까지의 거리를 **마진(Margin)** 이라 한다. SVM은 이 마진이 최대가 되는 결정 경계를 찾는다.

```
마진이 좁은 경계:          마진이 넓은 경계 (SVM 선택):

  ●  ●  /  ○  ○          ●  ●  |     |  ○  ○
  ●    /      ○          ●     | Gap |      ○
  ●   /   ○  ○           ●     |     |  ○  ○

마진이 좁으면:             마진이 넓으면:
새 데이터가 조금만          새 데이터가 어느 정도
경계 근처에 있어도          경계에서 떨어져 있어도
잘못 분류할 수 있음        올바르게 분류할 수 있음
→ 일반화 약함              → 일반화 강함
```

마진이 크다는 것은 결정 경계와 가장 가까운 각 클래스의 데이터 사이에 넉넉한 공간이 있다는 뜻이다. 이 공간이 커질수록 훈련 데이터에 없던 새로운 데이터가 와도 여유 있게 분류할 수 있다.

통계학적으로 보면 마진을 최대화하는 것이 **VC 차원(Vapnik-Chervonenkis dimension)** 기반의 일반화 오차 상한을 최소화하는 것과 연결된다. 이것이 SVM의 이론적 우수성의 근거다.

### 최적 초평면 (Optimal Hyperplane)

결정 경계를 수식으로 표현한다.

$$\mathbf{w}^T \mathbf{x} + b = 0$$

- $\mathbf{w}$: 결정 경계의 법선 벡터 (경계면이 어느 방향을 향하는지)
- $b$: 편향 (경계면이 원점에서 얼마나 떨어져 있는지)

2차원에서는 직선, 3차원에서는 평면, $n$차원에서는 $(n-1)$차원의 **초평면(Hyperplane)** 이다.

마진의 양쪽 경계를 수식으로 쓰면:

$$\mathbf{w}^T \mathbf{x} + b = +1 \quad \text{(양성 클래스의 마진 경계선)}$$ $$\mathbf{w}^T \mathbf{x} + b = -1 \quad \text{(음성 클래스의 마진 경계선)}$$

이 두 경계 사이의 거리가 마진 너비이고, 수식으로 계산하면:

$$\text{마진 너비} = \frac{2}{|\mathbf{w}|}$$

따라서 마진을 최대화하는 것은 $|\mathbf{w}|$를 최소화하는 것과 동일하다.

**SVM의 학습 = 이차계획법(Quadratic Programming) 최적화 문제**:

$$\text{최소화}: \quad \frac{1}{2}|\mathbf{w}|^2$$ $$\text{조건}: \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \quad \forall i$$

조건은 "모든 데이터 포인트가 자신의 마진 경계 바깥에 있어야 한다"는 것이다.

### 쌍대 문제 (Dual Problem)와 라그랑주 승수

원래의 최적화 문제(주 문제)를 라그랑주 승수법으로 **쌍대 문제(Dual Problem)** 로 변환하면 두 가지 이점이 생긴다.

첫째, 데이터의 내적 형태로 문제가 표현되어 커널 트릭을 적용할 수 있게 된다. 둘째, 라그랑주 승수 $\alpha_i$가 0이 아닌 점들이 서포트 벡터임이 자동으로 결정된다.

$$\text{쌍대 문제: 최대화} \quad \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_i y_j \langle\mathbf{x}_i, \mathbf{x}_j\rangle$$ $$\text{조건: } \sum_i \alpha_i y_i = 0, \quad \alpha_i \geq 0$$

- $\alpha_i = 0$: 마진 바깥의 점 → 결정 경계에 영향 없음
- $\alpha_i > 0$: 서포트 벡터 → 결정 경계를 결정

**SMO(Sequential Minimal Optimization)**: sklearn SVC 내부 알고리즘. 한 번에 두 개의 $\alpha_i$를 골라서 반복 최적화한다. 한 번에 하나씩 업데이트하는 좌표 하강법의 SVM 특화 버전이다.

### 서포트 벡터 (Support Vectors)

마진 경계선 위에 정확히 놓이는 데이터 포인트들이다. 이 점들이 $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$ 조건을 등호로 만족하는 점들이다.

서포트 벡터가 SVM의 핵심인 이유는 다음과 같다. SVM의 학습 결과로 얻는 **라그랑주 승수** $\alpha_i$가 서포트 벡터에만 0이 아닌 값을 가진다. 나머지 모든 데이터 포인트는 $\alpha_i = 0$이다. 그래서 예측 수식은 서포트 벡터들의 합으로만 이루어진다.

$$f(x) = \sum_{i \in \text{서포트벡터}} \alpha_i y_i \langle \mathbf{x}_i, \mathbf{x} \rangle + b$$

```
직관적 의미:
  서포트 벡터가 아닌 점들은 결정 경계 결정에 아무 역할도 하지 않음
  → 그 점들을 제거해도 같은 결정 경계가 나옴
  → SVM은 가장 "어려운" 점들(경계에 가장 가까운 점들)만 기억하는 모델

장점: 메모리 효율적, 대부분의 학습 데이터를 버려도 됨
```

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# SVM은 반드시 스케일링 필요 (거리 기반이므로)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_train)

svc = SVC(kernel='linear', C=1.0)
svc.fit(X_sc, y_train)

# 서포트 벡터 확인
print(f'서포트 벡터 수: {len(svc.support_vectors_)}개 / 전체 {len(X_train)}개')
print(f'클래스별 서포트 벡터 수: {svc.n_support_}')
print(f'w (법선 벡터): {svc.coef_}')
print(f'b (편향): {svc.intercept_}')
```

---

## 2. 소프트 마진 SVM (Soft Margin SVM)

### 하드 마진의 한계

위에서 설명한 마진 최대화는 **하드 마진(Hard Margin)** SVM이다. 이것은 모든 데이터가 마진 밖에 있어야 한다는 엄격한 조건을 요구한다.

현실 데이터에서는 두 가지 이유로 하드 마진이 적용되지 않는 경우가 많다.

```
1. 선형 분리 불가능한 데이터:
   두 클래스가 뒤섞여 있어서 어떤 직선도 완벽히 분리 불가

2. 이상치(Outlier):
   양성 클래스 영역 안에 음성 클래스 하나가 섞여 있는 경우
   이 하나 때문에 마진이 극도로 좁아지거나 수렴 불가
```

### 소프트 마진의 핵심 아이디어

**일부 데이터가 마진을 위반하는 것을 허용**하되, 위반한 만큼 손실(패널티)을 부과한다.

**슬랙 변수 $\xi_i$**: 각 데이터 포인트가 마진을 얼마나 위반했는지 나타낸다.

```
ξᵢ = 0:        마진 바깥에 올바르게 있음 (이상적인 상태)
0 < ξᵢ < 1:   마진 안에 있지만 올바른 쪽에 있음 (마진 위반)
ξᵢ = 1:        결정 경계 위에 정확히 있음
ξᵢ > 1:        결정 경계를 넘어 잘못된 쪽에 있음 (오분류)
```

소프트 마진 SVM의 최적화 문제:

$$\text{최소화}: \quad \frac{1}{2}|\mathbf{w}|^2 + C \sum_{i=1}^n \xi_i$$

$$\text{조건}: \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

왼쪽 항은 마진 최대화, 오른쪽 항은 오분류 패널티다. **C가 두 항의 균형을 조절한다.**

### C 파라미터 — 마진 너비와 오차 허용의 균형

$$\text{전체 손실} = \underbrace{\frac{1}{2}|\mathbf{w}|^2}_{\text{마진 최대화}} + \underbrace{C \sum \xi_i}_{\text{오분류 패널티}}$$

```
C가 매우 크면:
  오분류 패널티가 커서 가급적 모든 점을 올바르게 분류하려 함
  → 마진이 좁아짐 → 결정 경계가 복잡해짐 → 과적합 위험
  
C가 매우 작으면:
  오분류를 많이 허용 → 마진이 넓어짐 → 결정 경계가 단순해짐 → 과소적합 위험

C=1 (기본값): 균형점

직관: C는 "얼마나 엄격하게 훈련 데이터를 맞출 것인가"를 결정
      C 크면 정확도 우선, C 작으면 마진(일반화) 우선
```

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
C_values = [0.01, 1.0, 1000]

for ax, C in zip(axes, C_values):
    pipe = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C))
    pipe.fit(X_train, y_train)
    
    svc = pipe.named_steps['svc']
    n_sv = len(svc.support_vectors_)
    acc  = pipe.score(X_test, y_test)
    
    ax.set_title(f'C={C}\n서포트벡터={n_sv}개, 정확도={acc:.2f}')

plt.tight_layout()
plt.show()
# C가 작을수록 서포트벡터 많음 (마진 안에 많은 점 허용)
# C가 클수록 서포트벡터 적음 (마진 안에 점 없애려 함)
```

---

## 3. 결정 함수 (Decision Function)

새 데이터 $x$가 들어왔을 때 클래스를 결정하는 값이다.

$$d(x) = \frac{\mathbf{w}^T x + b}{|\mathbf{w}|}$$

이 값의 **부호**가 클래스를 결정하고, **크기**가 결정 경계까지의 실제 거리(확신도)를 나타낸다.

```
d(x) = +2.5  → 클래스 +1, 경계에서 멀리 있음 (확신 높음)
d(x) = +0.1  → 클래스 +1, 경계 바로 옆 (확신 낮음, 경계에 민감)
d(x) =  0.0  → 결정 경계 위 (분류 불확실)
d(x) = -1.8  → 클래스 -1, 확신 높음

마진 안:  |d(x)| < 1  ← 마진 위반 구간
마진 밖:  |d(x)| ≥ 1  ← 완전히 분류됨
```

이 결정 함수값은 그 자체로는 확률이 아니다. 값이 +5라고 해서 95% 확률이라는 뜻이 아니다. 이것을 확률로 변환하려면 Platt Scaling이 필요하다.

---

## 4. Hinge Loss — SVM의 손실 함수

$$L_{\text{hinge}} = \max(0, 1 - y \cdot f(x))$$

- $y$: 실제 레이블 (+1 또는 -1)
- $f(x)$: 결정 함수값

```
Hinge Loss 계산 예시:

  샘플 A: y=+1, f(x)=+2.0 → L = max(0, 1-1×2)  = max(0,-1) = 0   ← 마진 밖, 손실 없음
  샘플 B: y=+1, f(x)=+0.5 → L = max(0, 1-1×0.5) = max(0,0.5)= 0.5 ← 마진 안, 손실 발생
  샘플 C: y=+1, f(x)=-1.0 → L = max(0, 1-1×(-1)) = max(0,2) = 2   ← 오분류, 큰 손실
  샘플 D: y=-1, f(x)=+0.5 → L = max(0, 1-(-1)×0.5) = max(0,1.5)= 1.5 ← 오분류

"올바르게 분류했어도 마진 안에 있으면 손실이 발생한다"
→ SVM은 단순히 맞추는 것을 넘어 마진 밖으로 밀어내도록 학습
```

Hinge Loss의 의미: 결정 함수값이 $y \cdot f(x) \geq 1$이면 (마진 밖에서 올바르게 분류) 손실 = 0이다. 그렇지 않으면 1에서 멀어질수록 선형으로 손실이 커진다.

**Hinge Loss vs 다른 손실 함수 비교**:

|손실 함수|수식|특징|
|---|---|---|
|**Hinge Loss** (SVM)|$\max(0, 1-yf)$|마진 밖이면 0. 희소한 서포트벡터|
|**Cross-Entropy** (로지스틱)|$-\log P(y\|x)$|부드럽고 미분 가능. 확률 출력 자연스러움|
|**Squared Hinge**|$\max(0, 1-yf)^2$|마진 위반에 더 강한 패널티|
|**0-1 Loss**|$I[y \neq \hat{y}]$|맞으면 0, 틀리면 1. 최적화 불가|

---

## 5. Platt Scaling — SVM의 확률 출력

SVM은 기본적으로 클래스 레이블과 결정 함수값만 출력한다. 결정 함수값이 클래스 확률을 직접 나타내지 않기 때문이다.

**Platt Scaling(1999)** 은 결정 함수값 $f(x)$를 로지스틱 함수에 통과시켜 확률로 변환한다. 보정 파라미터 $A$, $B$를 교차검증으로 학습한다.

$$P(y=1|x) = \frac{1}{1 + \exp(Af(x) + B)}$$

```
왜 직접 시그모이드만 적용하지 않는가?
  SVM 결정 함수값의 분포가 로지스틱 회귀처럼 균일하지 않음
  A, B를 추가로 학습해서 분포를 보정해야 정확한 확률이 나옴

단점:
  probability=True로 설정하면 내부적으로 5-Fold CV를 추가로 수행
  → 학습 시간이 크게 늘어남
  → 작은 데이터셋에서는 오히려 확률 추정이 불안정할 수 있음
```

```python
from sklearn.svm import SVC
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

# probability=True: Platt Scaling 활성화
svc_platt = SVC(kernel='rbf', probability=True)
svc_platt.fit(X_train, y_train)

proba = svc_platt.predict_proba(X_test)
dec   = svc_platt.decision_function(X_test)

# Calibration 확인 (Platt Scaling이 얼마나 잘 보정됐는지)
fig, ax = plt.subplots(figsize=(6, 5))
CalibrationDisplay.from_estimator(svc_platt, X_test, y_test, ax=ax, name='SVM+Platt')
ax.set_title('Calibration Curve (대각선에 가까울수록 좋음)')
plt.show()
```

---

## 6. 커널 함수 (Kernel Function) — 비선형 SVM

### 선형 분리 불가능한 데이터의 처리

현실 데이터 중 많은 것들이 직선 하나로 분리되지 않는다.

```
XOR 패턴 (선형 분리 불가):        고차원으로 변환 후 분리 가능:

  ○ ●                               ↑
  ● ○       →   φ(x) 변환   →     |  ●
                                   |      ○
"이 평면에서는              "3D에서는 평면으로 분리 가능"
 직선 불가"
```

고차원으로 변환하면 선형 분리가 가능해지지만, 차원이 늘어날수록 계산 비용이 폭발적으로 증가한다.

### 커널 트릭 (Kernel Trick)

SVM의 결정 함수에는 데이터 포인트 간의 **내적(inner product)** 만 등장한다.

$$f(x) = \sum_i \alpha_i y_i \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}) \rangle + b$$

커널 함수 $K(\mathbf{x}_i, \mathbf{x}) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}) \rangle$를 사용하면, **고차원 변환 $\phi$를 명시적으로 계산하지 않고** 커널값만 계산해도 된다.

```
직접 변환 방식:
  x → φ(x) (고차원 변환) → 내적 계산
  → 100만 차원이면 계산 불가

커널 트릭:
  K(xᵢ, x) 함수로 고차원 내적을 원래 차원에서 직접 계산
  → 어떤 차원이든 커널 함수 한 번 계산으로 끝
```

### 슬라이드의 2D → 3D 변환 예시

$$\phi\begin{pmatrix}x_1 \ x_2\end{pmatrix} = \begin{pmatrix}x_1^2 \ \sqrt{2}x_1 x_2 \ x_2^2\end{pmatrix}$$

2D 공간에서 $(0,0), (1,0), (0,1), (1,1)$의 4점은 XOR처럼 배치되면 선형 분리가 불가능하다. $\phi$ 변환으로 3D로 올리면 이 점들이 초평면으로 분리 가능한 위치로 배치된다.

이때의 커널 함수(2차 다항 커널):

$$K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2 = \langle\phi(\mathbf{x}), \phi(\mathbf{z})\rangle$$

직접 3D 변환 후 내적을 계산하지 않아도, 원래 2D 벡터의 내적을 제곱하면 같은 결과가 나온다.

---

## 7. 커널 종류 4가지

### 선형 커널 (Linear Kernel)

$$K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$$

변환 없이 원래 공간에서 선형 분류한다. 특성 수($p$)가 샘플 수($n$)보다 많거나 데이터가 이미 선형 분리 가능한 경우에 적합하다. 텍스트 분류(TF-IDF 특성이 수만 개)에서 잘 작동한다.

### 다항식 커널 (Polynomial Kernel)

$$K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$$

- $d$: 다항식 차수 (곡선의 복잡도)
- $r$: 상수항 (독립항 유무 조절)
- $\gamma$: 스케일

```
d=1: 선형 커널과 같음
d=2: 2차 곡선으로 분리 (포물선, 타원 등)
d=3: 3차 곡선으로 분리 (더 복잡한 경계)

d가 커질수록 → 더 복잡한 경계 → 과적합 위험
```

### RBF 커널 (Radial Basis Function, 가우시안 커널)

$$K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma |\mathbf{x} - \mathbf{z}|^2\right)$$

가장 많이 사용되는 커널이다. 두 점 사이의 거리가 가까울수록 커널값이 1에 가깝고, 멀수록 0에 가까워진다.

```
RBF 커널의 의미:
  "두 점이 가까울수록 서로 유사하다"는 직관을 수식으로 표현
  
  무한 차원으로의 변환에 해당 (테일러 급수 전개시)
  → 어떤 복잡한 경계도 표현 가능
  → 대부분의 상황에서 좋은 첫 번째 선택
```

**gamma(γ) 파라미터 — 가우시안 커널 폭의 역수**:

```
γ = 1/σ²  (σ: 가우시안의 표준편차)

γ가 크면  (σ 작음):
  커널의 영향 범위가 좁음 → 가까운 점만 영향
  → 각 서포트벡터 주변의 "거품"이 작음
  → 결정 경계가 구불구불 → 과적합
  
γ가 작으면 (σ 큼):
  커널의 영향 범위가 넓음 → 멀리 있어도 영향
  → 거품이 커서 넓게 영향
  → 결정 경계가 부드러움 → 과소적합

sklearn gamma='scale': 1 / (n_features × X.var())  ← 권장
sklearn gamma='auto':  1 / n_features
```

### 시그모이드 커널 (Sigmoid Kernel)

$$K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$$

신경망의 하이퍼볼릭 탄젠트(tanh) 활성화 함수와 유사한 구조다. 특정 파라미터에서는 2층 신경망과 동일한 결정 경계를 만든다. 실제로는 RBF보다 잘 쓰이지 않는다.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np, matplotlib.pyplot as plt

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

kernels = [('linear', {}), ('poly', {'degree': 3}),
           ('rbf', {'gamma': 'scale'}), ('sigmoid', {})]

for ax, (kernel, params) in zip(axes.ravel(), kernels):
    pipe = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, **params))
    pipe.fit(X_tr, y_tr)
    acc = pipe.score(X_te, y_te)

    # 결정 경계 시각화
    sc = pipe.named_steps['svc']
    h = 0.02
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    scaler = pipe.named_steps['standardscaler']
    Z = sc.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=20, edgecolors='k', linewidths=0.3)

    sv_orig = scaler.inverse_transform(sc.support_vectors_)
    ax.scatter(sv_orig[:,0], sv_orig[:,1], s=80, facecolors='none',
               edgecolors='black', linewidths=2, label='서포트벡터')
    ax.set_title(f'kernel={kernel}, 정확도={acc:.2f}')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
```

---

## 8. 다중 분류 전략 — OvO와 OvR

SVM의 수식은 $y_i \in {+1, -1}$인 이진 분류를 위한 것이다. 3개 이상의 클래스를 처리할 때는 이진 분류기를 조합하는 전략을 사용한다.

### OvO (One-vs-One)

모든 클래스 **쌍**마다 하나의 이진 분류기를 훈련한다.

$$\text{분류기 수} = \binom{K}{2} = \frac{K(K-1)}{2}$$

- K=3: 3개 분류기 (A vs B, A vs C, B vs C)
- K=10: 45개 분류기

새 데이터를 분류할 때는 45개 분류기가 모두 투표하고, **가장 많이 선택된 클래스**를 최종 결정으로 한다.

```
장점:
  각 분류기는 두 클래스만 집중 학습 → 개별 정확도 높음
  전체 데이터가 아닌 두 클래스 데이터만 써서 빠름

단점:
  클래스 수 많으면 분류기 수가 폭발 (K=100이면 4950개)
  투표 결과가 동점이 될 수 있음
  
sklearn SVC(decision_function_shape='ovo')의 기본값
```

### OvR (One-vs-Rest)

각 클래스를 **"이 클래스 vs 나머지 전체"** 로 K개의 이진 분류기를 훈련한다.

- K=3: 3개 분류기 (A vs [B+C], B vs [A+C], C vs [A+B])

새 데이터를 K개 분류기에 통과시켜서 **결정 함수값이 가장 높은 클래스**를 선택한다.

```
장점:
  분류기 수 = K (OvO보다 훨씬 적음)
  대규모 다중 분류에서 효율적

단점:
  각 분류기가 불균형 데이터로 학습 (한 클래스 vs 나머지 전체)
  클래스가 많을수록 나머지가 많아져 불균형 심화
  
sklearn LinearSVC의 기본값
```

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# OvO (기본값)
svc_ovo = SVC(kernel='rbf', decision_function_shape='ovo')
svc_ovo.fit(X_train, y_train)
print(f'OvO 결정함수 shape: {svc_ovo.decision_function(X_test).shape}')
# (n_samples, K*(K-1)/2) — 각 쌍의 결정값

# OvR
svc_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
print(f'OvR 결정함수 shape: {svc_ovr.fit(X_train, y_train).decision_function(X_test).shape}')
# (n_samples, K) — 각 클래스의 결정값
```

---

## 9. SVM 모델 종류

sklearn은 목적에 따라 다양한 SVM 클래스를 제공한다.

|클래스|목적|특징|
|---|---|---|
|**SVC**|분류|커널 SVM. 소~중규모 데이터|
|**LinearSVC**|선형 분류|선형 커널 특화, 대용량에 빠름|
|**NuSVC**|분류|C 대신 Nu 파라미터|
|**SVR**|연속형 값 예측|ε-tube 기반 회귀|
|**LinearSVR**|선형 회귀|대용량 선형 회귀|
|**NuSVR**|회귀|Nu 파라미터 기반 회귀|
|**OneClassSVM**|이상치 탐지|정상 데이터만으로 학습|

### LinearSVC vs SVC(kernel='linear')

같은 선형 커널처럼 보이지만 내부 알고리즘과 특성이 다르다.

|구분|LinearSVC|SVC(kernel='linear')|
|---|---|---|
|내부 알고리즘|Liblinear (좌표 하강법)|Libsvm (SMO)|
|시간 복잡도|$O(n \times p)$|$O(n^2)$~$O(n^3)$|
|대용량 적합성|매우 적합|부적합|
|다중 분류 기본|OvR|OvO|
|확률 출력|불가|probability=True 가능|
|수렴 안정성|높음|높음|

---

## 10. Nu 파라미터 — C의 직관적 대안

C 파라미터는 패널티의 절대적인 강도를 나타내서 데이터 스케일에 민감하다. Nu($\nu$)는 **비율**로 표현되어 더 직관적이다.

$$0 < \nu \leq 1$$

Nu의 두 가지 의미:

- **이상치(오분류)를 허용하는 비율의 상한**: $\nu=0.1$이면 훈련 데이터의 최대 10%만 오분류 허용
- **서포트 벡터 비율의 하한**: 서포트 벡터는 훈련 데이터의 최소 $\nu$ 비율 이상

```
C와 Nu의 차이:
  C=100: "패널티가 100" → 데이터 스케일이 바뀌면 의미가 달라짐
  ν=0.1: "10% 이하만 오분류 허용" → 스케일 무관하게 항상 같은 의미

실용적으로:
  Nu가 허용 오류율을 직접 조절하므로 
  "이 데이터에서 최대 몇 %까지 오류를 허용할 것인가"를 
  사전에 결정할 수 있을 때 NuSVC가 유리
```

```python
from sklearn.svm import NuSVC, NuSVR

nusvc = NuSVC(
    nu=0.1,            # 이상치 허용 비율 (0~1). 클수록 느슨한 경계
    kernel='rbf',
    gamma='scale',
    probability=True
)
nusvc.fit(X_train, y_train)

print(f'실제 서포트벡터 비율: {len(nusvc.support_vectors_) / len(X_train):.2%}')
# ≥ nu로 보장됨
```

---

## 11. OneClassSVM — 이상치 탐지 (Novelty Detection)

정상 데이터만으로 학습해서 새 데이터가 정상 분포 안에 있는지를 판단한다. 이상 데이터 샘플이 없거나 극히 희소한 경우에 사용한다.

```
언제 사용하는가:
  제조업 불량 탐지:  정상 제품은 많지만 불량 예시가 없을 때
  사이버 보안:       정상 트래픽만 있고 공격 패턴을 미리 알 수 없을 때
  의료 이상 탐지:    건강한 환자 데이터만으로 이상 징후 탐지

이 경우 이진 분류가 불가능한 이유:
  이상 데이터가 없거나 매우 적어서 레이블 부여가 어려움
  이상 데이터의 분포를 미리 알 수 없음
```

**작동 원리**: 정상 데이터가 분포하는 영역(초구, hypersphere)을 학습한다. 원점과 정상 데이터 사이에 초평면을 그어서 정상 영역의 **경계(Frontier)** 를 정의한다. 새 데이터가 경계 안쪽에 있으면 정상(+1), 바깥쪽이면 이상치(-1)로 분류한다.

**Nu 파라미터** (OneClassSVM에서):

- 훈련 데이터 중 이상치로 처리할 비율의 상한
- 동시에 서포트 벡터 비율의 하한
- Nu가 작을수록 엄격한 경계 (더 좁은 정상 영역), 클수록 느슨한 경계

```python
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 정상 데이터만으로 학습
X_normal  = np.random.randn(200, 2)            # 정규 분포 (정상)
X_outlier = np.random.uniform(-5, 5, (20, 2))  # 균일 분포 (이상치)

ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_normal)

# 예측: +1 = 정상, -1 = 이상치
pred_normal  = ocsvm.predict(X_normal)   # 대부분 +1
pred_outlier = ocsvm.predict(X_outlier)  # 대부분 -1

print(f'정상 데이터 정확히 분류: {(pred_normal == 1).mean():.2%}')
print(f'이상치 탐지율:           {(pred_outlier == -1).mean():.2%}')

# 결정 경계 시각화
xx, yy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 7))
ax.contourf(xx, yy, Z, levels=[-999, 0], colors='salmon', alpha=0.3)
ax.contourf(xx, yy, Z, levels=[0, 999], colors='lightblue', alpha=0.3)
ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
ax.scatter(X_normal[:,0], X_normal[:,1], c='steelblue', s=15, label='정상')
ax.scatter(X_outlier[:,0], X_outlier[:,1], c='red', marker='x', s=80,
           linewidths=2, label='이상치')
ax.set_title(f'OneClassSVM (nu={ocsvm.nu})\n파란 영역=정상, 빨간 영역=이상')
ax.legend()
plt.show()
```

---

## 12. SVR (Support Vector Regression)

SVM의 마진 개념을 **회귀**로 확장했다. 예측값이 실제값에서 $\varepsilon$ 이내이면 손실이 0이고, 그보다 멀어질수록 선형으로 손실이 증가한다.

$$L_{\varepsilon}(y, f(x)) = \max(0, |y - f(x)| - \varepsilon)$$

이 손실 함수를 **ε-insensitive loss** 라고 한다.

```
ε-tube 시각화:

실제값 y
    │   ε-tube 상단  ─────────────
    │        ↑
    │        │ε      ← tube 안: 손실 = 0
    │        ↓
    │   ε-tube 하단  ─────────────
    │
    └──────────────────────── x
    
   tube 밖의 점들만 서포트벡터
   → tube 안이면 "충분히 맞은 것"으로 처리
   → 노이즈에 강함
```

```
SVM 회귀의 특징:
  일반 회귀(OLS): 모든 점의 오차를 최소화
  SVR: ε 이내의 오차는 무시하고 그 이상만 패널티

  → 노이즈나 이상치가 있어도 튜브 밖 점만 영향을 미침
  → 더 강건(Robust)한 회귀
```

```python
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.1   # 노이즈 포함 sin 함수

svr_rbf = SVR(kernel='rbf', C=100, epsilon=0.1, gamma=0.1)
svr_lin = SVR(kernel='linear', C=100, epsilon=0.1)

svr_rbf.fit(X, y)
svr_lin.fit(X, y)

X_test = np.arange(0, 5, 0.01)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, svr, title in zip(axes, [svr_rbf, svr_lin], ['SVR (RBF)', 'SVR (Linear)']):
    y_pred = svr.predict(X_test)
    ax.scatter(X, y, c='navy', s=20, alpha=0.6, label='훈련 데이터')
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label='예측')
    # ε-tube 표시
    ax.fill_between(X_test.ravel(), y_pred - 0.1, y_pred + 0.1,
                    alpha=0.2, color='red', label='ε-tube')
    # 서포트벡터 표시
    ax.scatter(X[svr.support_], y[svr.support_], c='red', s=60,
               zorder=5, label=f'서포트벡터 ({len(svr.support_)}개)')
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

---

## 13. 파라미터 정리 및 튜닝 가이드

### 파라미터 전체 정리

|파라미터|적용 모델|의미|과적합 방향|
|---|---|---|---|
|**C**|SVC, SVR, LinearSVC|마진 위반 패널티 강도|C 증가 → 복잡한 경계|
|**kernel**|SVC, NuSVC, SVR|커널 종류|rbf, poly가 linear보다 복잡|
|**gamma**|rbf, poly, sigmoid|커널 영향 범위의 역수|γ 증가 → 좁은 영역만 영향|
|**degree**|poly|다항식 차수|degree 증가|
|**nu**|NuSVC, NuSVR, OneClassSVM|이상치 허용 비율 (0~1)|nu 증가 → 느슨한 경계|
|**epsilon**|SVR, LinearSVR|ε-tube 너비|epsilon 증가 → 더 많은 오차 허용|
|**probability**|SVC, NuSVC|Platt Scaling 활성화|—|

### 실용적 튜닝 순서

```
1. 커널 선택:
   선형 분리 가능한가? → linear
   아니면? → rbf (거의 항상 좋은 첫 번째 선택)
   텍스트 데이터? → linear
   이미지 데이터? → rbf

2. C 탐색 (로그 스케일로):
   [0.001, 0.01, 0.1, 1, 10, 100, 1000]
   
3. gamma 탐색 (rbf 커널일 때):
   ['scale', 'auto', 0.001, 0.01, 0.1, 1]
   
4. 교차검증으로 최적 조합 결정
```

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# SVM 최적화 파이프라인
pipe = make_pipeline(StandardScaler(), SVC(probability=True))

param_dist = {
    'svc__C':      loguniform(1e-3, 1e3),          # 로그 균일 분포
    'svc__kernel': ['rbf', 'poly', 'linear'],
    'svc__gamma':  ['scale', 'auto'] + list(loguniform(1e-4, 1e1).rvs(8))
}

search = RandomizedSearchCV(
    pipe, param_dist,
    n_iter=50,              # 50개 조합 탐색
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

print(f'최적 파라미터: {search.best_params_}')
print(f'CV AUC:       {search.best_score_:.4f}')
print(f'테스트 AUC:   {search.score(X_test, y_test):.4f}')

# 최적 모델의 서포트벡터 정보
best_svc = search.best_estimator_.named_steps['svc']
print(f'서포트벡터 수: {len(best_svc.support_vectors_)}개')
```

---

## 14. SVM vs 로지스틱 회귀 — 언제 무엇을 쓸까

두 모델 모두 선형 분류기지만 손실 함수와 접근 방식이 다르다.

|구분|SVM|로지스틱 회귀|
|---|---|---|
|**목표**|마진 최대화|로그우도 최대화|
|**손실 함수**|Hinge Loss|Cross-Entropy (Log Loss)|
|**확률 출력**|Platt Scaling 필요|자연스럽게 출력|
|**이상치 영향**|서포트벡터만 영향 → 상대적으로 강건|모든 데이터 영향 → 이상치에 민감|
|**결정 경계**|마진 경계를 기준으로 결정|확률 0.5 지점이 결정 경계|
|**고차원**|매우 강함|강함|
|**대용량**|느림 ($O(n^2)$~$O(n^3)$)|빠름 ($O(n \times p)$)|
|**해석**|어려움 (특히 커널)|계수 해석 가능 (Odds Ratio)|
|**비선형**|커널 트릭으로 처리|특성 변환 후 처리|

```
결정 경계 근처에만 집중 vs 전체 데이터 활용:

SVM:          로지스틱회귀:
  ●  ●            ●  ●
  ●  ↑SV          ●    
  ● [결정경계]    ● [결정경계] → 모든 점의 영향
  ○ ↓SV          ○   
  ○  ○            ○  ○

SVM은 경계 근처 점들(서포트벡터)만 중요
로지스틱은 멀리 있는 점들도 경계 위치에 영향
```

```python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=42)

models = {
    'SVM (RBF)':    make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True)),
    'SVM (Linear)': make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True)),
    'LogisticReg':  make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='roc_auc')
    model.fit(X_tr, y_tr)
    test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f'{name:20s}: CV AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}, '
          f'테스트 AUC={test_auc:.4f}')
```

---

## 15. SVM의 장단점 및 사용 가이드

### 장점

- **고차원에서 강력함**: 특성 수($p$)가 샘플 수($n$)보다 많아도 잘 동작 (텍스트, 유전자 데이터)
- **강건한 이론적 기반**: VC 차원 이론에 근거한 일반화 오차 상한 보장
- **마진 최대화로 과적합 방지**: 결정 경계가 수학적으로 최적임이 증명됨
- **커널 트릭으로 비선형 처리**: 명시적 고차원 변환 없이 비선형 경계 표현
- **메모리 효율**: 서포트벡터만 저장 → 예측 시 메모리 적게 사용
- **이상치에 상대적으로 강건**: 서포트벡터 외 데이터는 경계에 영향 없음

### 단점

- **대용량 데이터에 느림**: 학습 시간 $O(n^2)$~$O(n^3)$ → 수십만 이상이면 비실용적
- **스케일에 매우 민감**: 반드시 `StandardScaler` 적용 필수
- **파라미터 튜닝 필수**: C, gamma 조합 탐색 없이는 성능 보장 어려움
- **확률 출력 비용**: Platt Scaling으로 학습 시간 크게 증가
- **비선형 SVM의 해석 불가**: 커널 공간의 결정 경계를 시각적으로 이해하기 어려움
- **결측값 처리 불가**: 전처리로 결측값 제거/대체 필요

### 언제 SVM을 선택하는가

```
SVM이 강한 경우:
  ✅ 데이터 크기 < 10,000개 (소~중규모)
  ✅ 특성 수가 매우 많음 (텍스트 TF-IDF, 유전자 발현)
  ✅ 클래스 경계가 명확한 데이터
  ✅ 훈련 데이터 적고 일반화 성능이 중요한 경우
  ✅ 비선형 경계가 필요하지만 딥러닝은 쓰기 싫은 경우

다른 모델이 나은 경우:
  ❌ 데이터 > 100,000개 → SGDClassifier(loss='hinge'), XGBoost
  ❌ 확률 보정이 중요 → LogisticRegression (자연스러운 확률)
  ❌ 해석 가능성 필요 → LogisticRegression, DecisionTree
  ❌ 빠른 학습/예측 → LinearSVC, LogisticRegression
  ❌ 이미지/음성 → 딥러닝(CNN, RNN)
```

---

## 16. 시험 대비 핵심 요약

빅데이터 분석기사 시험에 자주 출제되는 개념들이다.

```
❶ SVM의 목적: 마진(Margin)을 최대화하는 초평면 탐색

❷ 마진 = 2/‖w‖  →  마진 최대화 = ‖w‖ 최소화

❸ 서포트 벡터: 마진 경계선 위의 점들. 결정 경계 결정에만 관여

❹ 하드 마진 SVM: 모든 점이 마진 바깥에 있어야 함 (선형 분리 가능 시)
   소프트 마진 SVM: 슬랙 변수 ξ로 일부 오류 허용 (C로 조절)

❺ C 파라미터:
   - 클수록: 엄격한 분류, 좁은 마진, 과적합 위험
   - 작을수록: 넓은 마진, 과소적합 위험

❻ 커널 트릭:
   - 선형 분리 불가 → 고차원으로 변환 → 선형 분리 가능
   - 변환 없이 K(x,z)만 계산 (계산 효율)

❼ 주요 커널:
   - Linear: 선형 경계, 고차원 데이터에 유리
   - RBF(가우시안): 가장 범용적, gamma로 폭 조절
   - Polynomial: 곡선 경계, degree로 복잡도 조절
   - Sigmoid: tanh 기반

❽ gamma(RBF):
   - 클수록: 좁은 영향 범위, 구불구불한 경계, 과적합
   - 작을수록: 넓은 영향 범위, 부드러운 경계, 과소적합

❾ 다중 분류:
   - OvO (One vs One): K(K-1)/2개 분류기, 투표
   - OvR (One vs Rest): K개 분류기, 최대 점수

❿ Hinge Loss = max(0, 1 - y·f(x))
   마진 밖이면 0, 마진 안이나 오분류면 양수

⓫ Platt Scaling: SVM 결정값 → 확률 변환
   P(y=1|x) = 1 / (1 + exp(Af(x) + B))

⓬ OneClassSVM: 정상 데이터만으로 학습 → 이상치 탐지
   Nu = 이상치 허용 비율 상한 = 서포트벡터 비율 하한

⓭ SVR: ε-insensitive loss, ε 이내 오차 무시
   ε 크면 더 많은 오차 허용, C 크면 엄격한 피팅
```