---
title: 045 Multi Layer Perceptron
tag:
  - 헬스케어 ai
  - ML
description: 260616 수업 내용 정리
---

# 신경망 — 다층 퍼셉트론 (Multi-Layer Perceptron, MLP)

MLP는 인간 뇌의 뉴런 연결 구조를 모방한 모델이다. 단층 퍼셉트론이 선형 분리만 가능하다는 한계를 극복하기 위해 **여러 층(Layer)을 쌓고 비선형 활성화 함수를 도입**해서 복잡한 패턴도 학습할 수 있게 만든 것이다.

뇌의 뉴런은 여러 입력 신호를 받아서 그 합이 일정 임계값을 넘으면 다음 뉴런으로 신호를 전달한다. MLP의 인공 뉴런도 같은 원리다. 각 입력에 **가중치(중요도)를 곱해서 합산**하고, **활성화 함수(임계값 역할)** 를 통과시켜 다음 층으로 신호를 전달한다.

```
생물학적 뉴런              인공 뉴런(퍼셉트론)

수상돌기 (입력)            x₁, x₂, x₃ (입력)
시냅스 강도 (연결 강도)    w₁, w₂, w₃ (가중치)
세포체 (신호 합산)         z = w₁x₁ + w₂x₂ + w₃x₃ + b
축삭 (출력)                a = f(z) (활성화 함수 통과 후 출력)
```

---

## 1. 퍼셉트론에서 MLP로

### 단층 퍼셉트론의 한계

1969년 Minsky와 Papert가 증명한 XOR 문제가 대표적이다. 단층 퍼셉트론은 직선 하나로 분리할 수 없는 패턴을 학습하지 못한다.

퍼셉트론의 수식을 보면 이유가 명확하다.

$$\hat{y} = \text{step}(w_1 x_1 + w_2 x_2 + b)$$

$w_1 x_1 + w_2 x_2 + b = 0$이 바로 결정 경계 직선이다. 아무리 $w_1, w_2, b$를 바꿔도 **직선 하나**밖에 만들 수 없다.

```
XOR 문제:
  입력 (0,0) → 출력 0
  입력 (0,1) → 출력 1
  입력 (1,0) → 출력 1
  입력 (1,1) → 출력 0

  x₂
  1│  ○(0,1)    ●(1,1)
   │
  0│  ●(0,0)    ○(1,0)
   └──────────────── x₁
        0           1
  
  ● 클래스와 ○ 클래스를 직선 하나로 분리할 방법이 없음
  → 어떤 직선을 그어도 반드시 하나의 점이 잘못 분류됨
```

### MLP의 해결책

은닉층과 비선형 활성화 함수를 추가하면 **여러 직선의 조합**으로 어떤 복잡한 경계도 표현할 수 있다.

```
직관:
  직선 1: x₁ + x₂ = 0.5  (왼쪽 아래 ●과 나머지를 구분)
  직선 2: x₁ + x₂ = 1.5  (오른쪽 위 ●과 나머지를 구분)
  
  두 직선의 사이 = "x₁+x₂가 0.5~1.5 사이" = ○ 클래스
  이 구간을 표현하려면 직선 2개의 조합이 필요
  → 은닉층이 이 직선들을 생성하고 출력층이 조합

은닉층이 많을수록:
  1개 은닉층: 볼록 다각형 모양 경계
  2개 은닉층: 어떤 복잡한 모양도 가능
```

**Universal Approximation Theorem**: 충분한 은닉 유닛이 있으면 1개 은닉층의 MLP만으로도 어떤 연속 함수도 임의의 정밀도로 근사할 수 있다. 이것이 MLP가 강력한 이론적 근거다.

---

## 2. MLP의 구조

### 층(Layer)의 종류

```
입력층 (Input Layer):
  데이터의 특성을 그대로 받아들임
  뉴런 수 = 입력 특성 수 (x1, x2, x3)
  가중치 학습 없음

은닉층 (Hidden Layer):
  입력을 변환하는 중간 처리 단계
  층 수와 각 층의 뉴런 수는 하이퍼파라미터
  비선형 활성화 함수 적용

출력층 (Output Layer):
  최종 예측값 출력
  분류: 클래스 수만큼 뉴런 (Softmax or Sigmoid)
  회귀: 뉴런 1개 (선형 or Identity)
```

슬라이드 그림:

```
Input    Hidden     Output
 x1 ────→ ○ ─┐
  ↘    ×  ○  ├──→ ○ ──→ Activation → y1
 x2 ────→ ○  |         ↗            → y2
  ↘    ×  ○ ─┘  ──→ ○
 x3 ────→ ○

모든 이전 층의 뉴런이 다음 층의 모든 뉴런과 연결
= 완전연결(Fully Connected) = Dense Layer
```

### 행렬 연산으로 이해하기

슬라이드의 행렬 곱셈이 바로 한 층의 연산이다.

$$\mathbf{Z} = \mathbf{A}^{(prev)} \times \mathbf{W} + \mathbf{b}$$

```
입력층(4개) × 가중치행렬(4×4) = 출력행렬(4×4)

[a₁₁ a₁₂ a₁₃ a₁₄]   [b₁₂ b₁₃ b₁₄]   [c₁₁ c₁₂ c₁₃ c₁₄]
[a₂₁ a₂₂ a₂₃ a₂₄] × [b₂₂ b₂₃ b₂₄] = [c₂₁ c₂₂ c₂₃ c₂₄]
[a₃₁ a₃₂ a₃₃ a₃₄]   [b₃₂ b₃₃ b₃₄]   [c₃₁ c₃₂ c₃₃ c₃₄]
[a₄₁ a₄₂ a₄₃ a₄₄]   [b₄₂ b₄₃ b₄₄]   [c₄₁ c₄₂ c₄₃ c₄₄]

A: 입력(이전 층 출력)
W: 가중치 행렬 (빨간 점선 — 학습되는 파라미터)
C: 이 층의 선형 결합 결과 (활성화 함수 적용 전)
```

이것을 수식으로 풀어보면, 출력의 첫 번째 원소 $c_{11}$은 다음과 같다.

$$c_{11} = a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31} + a_{14}b_{41}$$

즉 입력 벡터의 모든 값이 각 가중치와 곱해져 합산된다. **행렬 곱셈 하나가 그 층의 모든 뉴런 계산을 동시에 처리**하는 것이다.

**파라미터 수 계산**: 입력 $n$개 → 출력 $m$개인 층의 파라미터 수는 $n \times m + m$이다 (가중치 + 편향).

```
예: hidden_layer_sizes=(128, 64) 이고 입력 10개, 출력 2개인 MLP

  입력층(10) → 은닉층1(128):  10×128 + 128 = 1,408개
  은닉층1(128) → 은닉층2(64): 128×64  + 64  = 8,256개
  은닉층2(64)  → 출력층(2):    64×2   + 2   =   130개
  ──────────────────────────────────────────────────
  합계:                                       9,794개

→ 비교적 단순한 MLP도 수천~수만 개의 파라미터를 가짐
→ 딥러닝은 수억 개 이상
```

이후 활성화 함수를 통과시켜 비선형성을 부여한다:

$$\mathbf{A} = f(\mathbf{Z})$$

---

## 3. 활성화 함수 (Activation Function)

### 왜 활성화 함수가 필요한가

활성화 함수 없이 층을 쌓으면 아무리 깊어도 결국 하나의 선형 변환과 같다.

$$W_3 \cdot W_2 \cdot W_1 \cdot x = (W_3 W_2 W_1) \cdot x = W_{합산} \cdot x$$

활성화 함수가 **비선형성**을 추가해서 각 층이 의미 있는 변환을 하게 만든다.

### 활성화 함수 4종

#### ReLU (Rectified Linear Unit) — 딥러닝의 기본

$$f(x) = \max(0, x)$$

```
음수: 0으로 처리
양수: 그대로 통과

그래프:
        │  /
        │ /
────────┼──────── x
        │
```

ReLU가 딥러닝의 표준이 된 이유는 **기울기 소실 문제를 크게 완화**하기 때문이다. 양수 구간에서 기울기가 정확히 1이라서 역전파 시 기울기가 줄어들지 않는다. 또한 계산이 단순해서(비교 연산 하나) 학습 속도가 빠르다.

**Dying ReLU 문제**: 음수 입력에서 출력이 항상 0이고 기울기도 0이라서, 한번 음수 영역에 갇힌 뉴런은 다시는 활성화되지 않는다. 이를 막기 위해 Leaky ReLU($f(x) = \max(0.01x, x)$)나 ELU를 사용하기도 한다.

- **사용**: 은닉층의 기본값

#### Identity (항등 함수)

$$f(x) = x$$

입력을 그대로 출력한다. 선형 변환만 하는 것이므로 활성화 함수가 없는 것과 같다. 회귀 문제의 출력층에서 사용한다. "예측값이 어떤 범위의 연속값이어야 한다"는 제약이 없을 때 적합하다.

#### Logistic (Sigmoid)

$$f(x) = \frac{1}{1+e^{-x}}$$

출력을 0~1 사이로 변환한다. 이 때문에 자연스럽게 **확률로 해석**할 수 있다.

```
x = -10: f(x) ≈ 0.00005  (거의 0)
x =   0: f(x) = 0.5       (중립)
x =  10: f(x) ≈ 0.99995  (거의 1)
```

**기울기 소실 문제**: 입력이 크거나 작으면 기울기가 0에 가까워진다. Sigmoid의 최대 기울기는 0.25(x=0에서)다. 이 값이 층을 거칠수록 곱해지면 금세 0에 수렴한다. 10층 이상의 깊은 네트워크에서는 쓰기 어렵다.

- **사용**: 이진 분류 출력층, 로지스틱 회귀와 연결

#### Tanh (하이퍼볼릭 탄젠트)

$$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

출력 범위 -1~1이다. Sigmoid의 "출력이 항상 양수"라는 문제를 해결했다. 출력이 양수와 음수를 오가므로 평균이 0에 가까워 학습이 더 안정적이다. 하지만 역시 기울기 소실 문제는 남아 있다. RNN에서는 여전히 tanh가 많이 쓰인다.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 300)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
funcs = [
    ('ReLU',     lambda x: np.maximum(0, x)),
    ('Identity', lambda x: x),
    ('Sigmoid',  lambda x: 1 / (1 + np.exp(-x))),
    ('Tanh',     lambda x: np.tanh(x))
]

for ax, (name, f) in zip(axes.ravel(), funcs):
    y = f(x)
    ax.plot(x, y, 'b-', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(name, fontsize=12)
    ax.set_xlim(-5, 5)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 활성화 함수 선택 가이드

|층 위치|문제 유형|권장 함수|
|---|---|---|
|은닉층|모든 경우|ReLU (기본값)|
|출력층|이진 분류|Sigmoid (확률 0~1)|
|출력층|다중 분류|Softmax (클래스 확률 합=1)|
|출력층|회귀|Identity (값을 그대로)|
|은닉층|RNN 등|Tanh|

---

## 4. 순전파 (Forward Propagation)

입력에서 출력까지 **한 방향으로 계산이 흐르는 과정**이다. 학습된 가중치로 예측값을 만드는 과정이기도 하다.

```
슬라이드의 순전파 흐름:
  [x1, x2, x3]
    ↓ (가중치 W₁ 행렬 곱 + 편향 b₁)
  선형 결합 z₁ = W₁x + b₁
    ↓ (활성화 함수 ReLU)
  은닉층 출력 a₁ = max(0, z₁)
    ↓ (가중치 W₂ 행렬 곱 + 편향 b₂)
  선형 결합 z₂ = W₂a₁ + b₂
    ↓ (출력층 활성화 함수)
  출력 ŷ = [y1, y2]
    ↓
  손실 계산: logloss 또는 (ŷ - y)²
```

### 단계별 수치 계산 예시

실제로 숫자를 넣어서 따라가 보면 각 층에서 무슨 일이 벌어지는지 명확해진다.

```
네트워크 구조: 입력 2개 → 은닉층 3개 → 출력 1개
입력: x = [1, 2]

─── 은닉층 계산 ───────────────────────────────────

가중치 W₁ (2×3):         편향 b₁ (1×3):
  [0.5  -0.3  0.8]         [0.1  0.0  -0.2]
  [0.2   0.7 -0.1]

z₁ = x · W₁ + b₁:
  z₁[0] = 1×0.5 + 2×0.2 + 0.1 = 1.0
  z₁[1] = 1×(-0.3) + 2×0.7 + 0.0 = 1.1
  z₁[2] = 1×0.8 + 2×(-0.1) + (-0.2) = 0.4

  z₁ = [1.0, 1.1, 0.4]

ReLU 적용 (a₁ = max(0, z₁)):
  a₁ = [1.0, 1.1, 0.4]   ← 모두 양수라 그대로

─── 출력층 계산 ───────────────────────────────────

가중치 W₂ (3×1):         편향 b₂ (1×1):
  [0.6]                     [0.1]
  [-0.4]
  [0.3]

z₂ = a₁ · W₂ + b₂:
  z₂ = 1.0×0.6 + 1.1×(-0.4) + 0.4×0.3 + 0.1
      = 0.6 - 0.44 + 0.12 + 0.1 = 0.38

Sigmoid 적용 (이진 분류):
  ŷ = 1/(1 + e^(-0.38)) ≈ 0.594

─── 결론 ────────────────────────────────────────

입력 [1, 2]에 대해 클래스 1일 확률 = 59.4%
```

```python
import numpy as np

def relu(x):       return np.maximum(0, x)
def sigmoid(x):    return 1 / (1 + np.exp(-x))

# 네트워크 파라미터
X  = np.array([[1, 2]])               # 입력
W1 = np.array([[0.5, -0.3, 0.8],
               [0.2,  0.7,-0.1]])     # 입력→은닉 가중치
b1 = np.array([[0.1, 0.0, -0.2]])
W2 = np.array([[0.6], [-0.4], [0.3]])  # 은닉→출력 가중치
b2 = np.array([[0.1]])

# 순전파
z1 = X  @ W1 + b1;  a1 = relu(z1)
z2 = a1 @ W2 + b2;  y_hat = sigmoid(z2)

print(f'z1: {z1}')
print(f'a1 (ReLU): {a1}')
print(f'z2: {z2}')
print(f'ŷ (예측 확률): {y_hat}')
```

```python
import numpy as np

# 간단한 신경망 (입력 3 → 은닉 4 → 출력 2)
np.random.seed(42)
X = np.array([[1.0, 2.0, 3.0]])   # 입력 (1샘플 × 3특성)

# 가중치 초기화
W1 = np.random.randn(3, 4) * 0.1  # 입력→은닉
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 2) * 0.1  # 은닉→출력
b2 = np.zeros((1, 2))

# 순전파
z1 = X @ W1 + b1           # 선형 결합 (1×4)
a1 = np.maximum(0, z1)     # ReLU 활성화 (1×4)
z2 = a1 @ W2 + b2          # 선형 결합 (1×2)

# 출력층: Softmax (다중 분류)
def softmax(z):
    exp_z = np.exp(z - z.max())  # 수치 안정성
    return exp_z / exp_z.sum()

y_pred = softmax(z2[0])
print(f'z1 (선형 결합): {z1}')
print(f'a1 (ReLU 후):  {a1}')
print(f'z2 (출력 전):  {z2}')
print(f'예측 확률:     {y_pred}')  # 두 클래스의 확률 합=1
```

---

## 5. 손실 함수 (Cost/Loss Function)

예측값($\hat{y}$)과 실제값($y$)의 차이를 수치로 표현한다. 이 값이 학습의 목표가 된다.

### MSE (Mean Squared Error) — 회귀

슬라이드의 수식: $\text{Cost} = (\hat{y} - y)^2$

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

```
예측이 맞으면: (ŷ - y)² = 0 → 손실 = 0
예측이 틀리면: 오차 제곱 → 큰 오차에 더 큰 패널티

실제값 y=5, 예측값 ŷ=7:  손실 = (7-5)² = 4
실제값 y=5, 예측값 ŷ=3:  손실 = (3-5)² = 4  (방향 무관, 크기만)
```

### Log Loss (Cross-Entropy) — 분류

슬라이드에서 언급된 `logloss`:

$$\text{Log Loss} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

```
y=1 (양성)이고 ŷ=0.9 (90% 확률로 양성 예측): -log(0.9) ≈ 0.10 → 작은 손실
y=1 (양성)이고 ŷ=0.1 (10% 확률로 양성 예측): -log(0.1) ≈ 2.30 → 큰 손실

"확률 0에 가까운 잘못된 예측에 로그 함수로 매우 큰 패널티"
```

### 손실 함수 선택

|문제|손실 함수|sklearn 파라미터|
|---|---|---|
|회귀|MSE|`loss='squared_error'`|
|이진 분류|Binary Cross-Entropy|`activation='logistic'`|
|다중 분류|Categorical Cross-Entropy|`activation='softmax'` 내부 처리|

---

## 6. 역전파 (Backpropagation)

### 핵심 아이디어

순전파로 예측을 하고 나면 "어느 가중치를 얼마나 바꿔야 손실이 줄어드는가"를 알아야 한다. 가중치가 수천~수만 개인데 하나씩 바꿔보며 확인할 수는 없다. 역전파는 이 기울기를 **효율적으로 한 번에 계산**하는 알고리즘이다.

```
직관:
  출력에서 손실이 발생했다 → 이 손실의 책임이 어느 가중치에 얼마나 있는가?
  
  마지막 층 가중치: 출력에 바로 연결 → 손실에 직접 기여 → 기울기 계산 쉬움
  첫 번째 층 가중치: 여러 층을 거쳐서 출력에 영향 → 기여도 추적이 어려움
  
  역전파: 출력층 → 은닉층 → 입력층 방향으로
          체인 룰을 이용해 각 층의 기울기를 순서대로 계산
```

### 체인 룰 (Chain Rule, 연쇄 법칙)

슬라이드의 핵심 수식:

$$\frac{\partial E}{\partial w} = \frac{\partial E}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial w}$$

이것은 합성 함수의 미분 공식이다. $E$가 $\hat{y}$의 함수이고, $\hat{y}$가 $w$의 함수일 때, $E$를 $w$로 미분하려면 두 미분의 곱으로 계산한다.

3층 신경망 전체의 체인 룰을 풀어 쓰면:

$$\frac{\partial E}{\partial w_1} = \underbrace{\frac{\partial E}{\partial \hat{y}}}_{\text{손실→출력}} \times \underbrace{\frac{\partial \hat{y}}{\partial a_2}}_{\text{출력→은닉2}} \times \underbrace{\frac{\partial a_2}{\partial z_2}}_{\text{활성화 미분}} \times \underbrace{\frac{\partial z_2}{\partial a_1}}_{\text{은닉2→은닉1}} \times \underbrace{\frac{\partial a_1}{\partial z_1}}_{\text{활성화 미분}} \times \underbrace{\frac{\partial z_1}{\partial w_1}}_{\text{은닉1→가중치}}$$

```
각 항의 의미:
  ∂E/∂ŷ:      손실 함수를 예측값으로 미분 (MSE면 2(ŷ-y), CE면 ŷ-y)
  ∂ŷ/∂z:      활성화 함수의 미분 (sigmoid면 σ(1-σ), relu면 0 or 1)
  ∂z/∂w:      선형 결합의 가중치 미분 = 이전 층의 출력값 a

핵심: 이미 계산된 값들을 재사용 → 효율적
```

### 수치 예시 — 역전파 한 스텝

```
앞선 순전파 결과를 이용:
  입력 x=[1,2], 실제값 y=1, 예측값 ŷ=0.594

손실 (MSE):
  E = (ŷ - y)² = (0.594 - 1)² = 0.165

역전파 출력층:
  ∂E/∂ŷ     = 2(ŷ - y) = 2(0.594-1) = -0.812
  ∂ŷ/∂z₂   = ŷ(1-ŷ) = 0.594×0.406 = 0.241  (Sigmoid 미분)
  ∂z₂/∂W₂  = a₁ = [1.0, 1.1, 0.4]          (z₂=a₁·W₂+b₂이므로)

  ∂E/∂W₂ = (∂E/∂ŷ) × (∂ŷ/∂z₂) × a₁ᵀ
           = (-0.812) × 0.241 × [1.0, 1.1, 0.4]ᵀ
           = -0.196 × [1.0, 1.1, 0.4]ᵀ
           = [-0.196, -0.216, -0.078]ᵀ

가중치 업데이트 (학습률 α=0.1):
  W₂ ← W₂ - α × ∂E/∂W₂
  W₂ = [0.6, -0.4, 0.3]ᵀ - 0.1 × [-0.196, -0.216, -0.078]ᵀ
     = [0.620, -0.378, 0.308]ᵀ

같은 방식으로 은닉층까지 역방향으로 계속 계산
```

### 가중치 업데이트

$$w \leftarrow w - \alpha \cdot \frac{\partial E}{\partial w}$$

- $\alpha$: 학습률 (Learning Rate). 한 번에 얼마나 이동할지
- 기울기의 **반대 방향**으로 이동 → 손실이 줄어드는 방향

```
학습률의 영향:

  α 너무 크면: 최솟값을 지나쳐 발산
  ──────────
         ↑ ↑  ← 너무 큰 보폭으로 왔다갔다
  ──────

  α 너무 작으면: 수렴이 매우 느림
  ↓ ↓ ↓ ↓ ↓ ← 조금씩 이동해서 오래 걸림

  α 적당하면: 안정적으로 수렴
  ↓ ↓ ↓ ← 적절한 보폭으로 내려감
    최솟값
```

```python
import numpy as np

def relu(x):              return np.maximum(0, x)
def relu_deriv(x):        return (x > 0).astype(float)
def sigmoid(x):           return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):     s = sigmoid(x); return s * (1 - s)

np.random.seed(42)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)   # XOR

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))
lr = 0.5

losses = []
for epoch in range(5000):
    # ── 순전파 ──────────────────────────────────
    z1 = X  @ W1 + b1
    a1 = relu(z1)                # 은닉층 활성화
    z2 = a1 @ W2 + b2
    ŷ  = sigmoid(z2)             # 출력층 활성화

    loss = np.mean((ŷ - y) ** 2)
    losses.append(loss)

    # ── 역전파 ──────────────────────────────────
    # 출력층
    dL_dŷ  = 2 * (ŷ - y) / len(X)          # ∂Loss/∂ŷ
    dŷ_dz2 = sigmoid_deriv(z2)              # ∂ŷ/∂z₂ (Sigmoid 미분)
    delta2  = dL_dŷ * dŷ_dz2               # 출력층 오차 신호

    dL_dW2 = a1.T @ delta2                  # ∂Loss/∂W₂
    dL_db2 = delta2.sum(axis=0, keepdims=True)

    # 은닉층 (체인 룰 적용)
    dL_da1  = delta2 @ W2.T                 # 출력층 오차를 은닉층으로 전달
    da1_dz1 = relu_deriv(z1)               # ∂a₁/∂z₁ (ReLU 미분)
    delta1  = dL_da1 * da1_dz1             # 은닉층 오차 신호

    dL_dW1 = X.T @ delta1
    dL_db1 = delta1.sum(axis=0, keepdims=True)

    # ── 가중치 업데이트 ──────────────────────────
    W2 -= lr * dL_dW2;  b2 -= lr * dL_db2
    W1 -= lr * dL_dW1;  b1 -= lr * dL_db1

    if epoch % 1000 == 0:
        print(f'Epoch {epoch:5d}: Loss = {loss:.4f}')

print(f'\nXOR 예측 결과:')
print(ŷ.round(3))  # [[0], [1], [1], [0]]에 가까워야 함
```

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 간단한 1층 역전파 예시
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력 (XOR 데이터)
y = np.array([[0], [1], [1], [0]])               # 레이블

np.random.seed(42)
W = np.random.randn(2, 1) * 0.1  # 가중치
b = np.zeros((1, 1))
lr = 0.1

losses = []
for epoch in range(1000):
    # 순전파
    z    = X @ W + b
    ŷ    = sigmoid(z)

    # 손실 (MSE)
    loss = np.mean((ŷ - y) ** 2)
    losses.append(loss)

    # 역전파 (체인 룰)
    dL_dŷ  = 2 * (ŷ - y) / len(X)        # ∂Loss/∂ŷ
    dŷ_dz  = sigmoid_derivative(z)         # ∂ŷ/∂z
    dz_dW  = X                             # ∂z/∂W

    dL_dW = dz_dW.T @ (dL_dŷ * dŷ_dz)   # 체인 룰
    dL_db = np.sum(dL_dŷ * dŷ_dz, axis=0)

    # 가중치 업데이트
    W -= lr * dL_dW
    b -= lr * dL_db

    if epoch % 200 == 0:
        print(f'Epoch {epoch}: Loss = {loss:.4f}')

print(f'최종 예측:\n{ŷ.round(3)}')
```

---

## 7. 기울기 소실 문제 (Vanishing Gradient)와 해결책

### 문제의 원인

역전파에서 체인 룰로 기울기를 곱해 나가다 보면, 시그모이드 함수의 기울기(최대 0.25)가 반복적으로 곱해진다.

```
기울기 소실 과정 (Sigmoid 10층 사용 시):

  출력층: 기울기 = 1.0
  9층:    기울기 × 0.25 = 0.25
  8층:    기울기 × 0.25 = 0.0625
  7층:    기울기 × 0.25 = 0.0156
  ...
  1층:    기울기 × 0.25¹⁰ = 0.000001

  → 입력에 가까운 층의 가중치가 거의 업데이트되지 않음
  → 깊은 층이 전혀 학습되지 않는 것과 같음

ReLU를 사용하면:
  양수 구간에서 기울기 = 1.0
  → 아무리 곱해도 기울기가 줄어들지 않음
  → 100층 이상도 학습 가능
```

### 해결책

```
1. ReLU 사용:
   양수 구간에서 기울기 = 1 → 아무리 곱해도 1
   기울기 소실 문제 크게 완화

2. 배치 정규화 (Batch Normalization):
   각 층의 출력을 평균 0, 분산 1로 정규화
   → 활성화 함수 입력이 포화 영역(기울기≈0)에 빠지지 않도록 예방
   → 학습 안정성 향상, 더 높은 학습률 사용 가능

3. Residual Connection (ResNet):
   입력을 몇 층을 건너뛰어 출력에 더함 (x + F(x))
   → 기울기가 스킵 연결을 통해 직접 흐름
   → 수백 층도 학습 가능

4. 가중치 초기화 방법:
   Xavier(Glorot): tanh/sigmoid에 적합 → 분산 = 2/(fan_in + fan_out)
   He:             ReLU에 적합         → 분산 = 2/fan_in
```

---

## 8. 역전파의 Solver (최적화 방법)

슬라이드에서 "역전파(solver)"라고 명시한 부분이다.

### Gradient Descent 종류

```
Batch GD:     전체 데이터로 기울기 계산 후 한 번 업데이트
              → 안정적이지만 느림

SGD:          샘플 1개로 기울기 계산 후 즉시 업데이트
              → 빠르지만 진동이 심함

Mini-batch GD: 일부(batch_size개)로 기울기 계산
              → 위 둘의 절충 (실무 표준)
```

### Momentum — 비틀거림 문제 해결

슬라이드에서 언급된 "비틀거림의 문제"를 해결하는 방법이다.

```
일반 경사하강법의 문제:
  골짜기에서 좌우로 진동하며 느리게 수렴
  → "비틀거림(zigzag)"

Momentum:
  이전 이동 방향을 관성으로 누적
  → 골짜기 방향으로 빠르게, 진동 방향으로 억제

슬라이드 그래프:
  gradient step:   지그재그 (진동)
  momentum step:   부드럽게 목표로 수렴
  actual step:     두 방향의 합
```

$$v_t = \gamma v_{t-1} + \alpha \nabla L(\theta)$$ $$\theta \leftarrow \theta - v_t$$

- $\gamma$: 모멘텀 계수 (0.9가 일반적)
- $v_t$: 이전 속도의 누적

### Adam (Adaptive Moment Estimation)

현재 딥러닝의 표준 옵티마이저다. Momentum + 적응형 학습률을 결합한 것으로, 두 가지 핵심 아이디어가 있다.

**1차 모멘트 $m_t$**: 기울기의 지수 이동 평균 → 어느 방향으로 이동하는 관성 **2차 모멘트 $v_t$**: 기울기 제곱의 지수 이동 평균 → 각 방향의 기울기 크기

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L \quad \text{(방향)}$$ $$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2 \quad \text{(크기²)}$$ $$\theta \leftarrow \theta - \frac{\alpha}{\sqrt{v_t}+\epsilon} m_t$$

```
직관:
  분자 m_t: 관성으로 안정적인 방향으로 이동
  분모 √v_t: 자주 큰 기울기가 나온 방향은 학습률을 줄임
             드물게 작은 기울기가 나온 방향은 학습률을 높임

결과:
  자주 업데이트되는 파라미터: 기울기가 큼 → √v_t 큼 → 효과적 학습률 낮음
  드물게 업데이트되는 파라미터: 기울기 작음 → √v_t 작음 → 효과적 학습률 높음
  
  → 각 파라미터가 최적의 속도로 학습됨
  → 학습률 튜닝 부담이 크게 줄어듦

β₁=0.9, β₂=0.999 (기본값):
  β₁: 이전 방향의 90%를 현재에 유지
  β₂: 이전 크기 정보의 99.9%를 유지
```

### sklearn의 solver 옵션

|Solver|방법|특징|적합한 경우|
|---|---|---|---|
|**adam**|Adam|적응형 학습률, 빠름|기본값, 대부분의 경우|
|**sgd**|확률적 경사하강법|Momentum 지원|대용량, 세밀한 조정 필요|
|**lbfgs**|준뉴턴법 (2차 미분)|소용량에 정확|소규모 데이터|
|**beta_1**|Adam의 m_t 감쇠율|(파라미터)|기본값 0.9|
|**beta_2**|Adam의 v_t 감쇠율|(파라미터)|기본값 0.999|

---

## 9. sklearn의 MLP 클래스

### MLPClassifier — 분류

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP는 반드시 스케일링 필요
mlp_clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # 은닉층 구조: 3층, 각 128/64/32개 뉴런
        activation='relu',                  # 은닉층 활성화 함수
        solver='adam',                      # 최적화 방법
        alpha=0.0001,                       # L2 정규화 강도
        batch_size='auto',                  # 배치 크기 (200 또는 샘플 수)
        learning_rate='constant',           # 학습률 방식 ('constant', 'invscaling', 'adaptive')
        learning_rate_init=0.001,           # 초기 학습률
        max_iter=200,                       # 최대 에포크 수
        early_stopping=True,                # 검증 성능 개선 없으면 조기 종료
        validation_fraction=0.1,            # 검증용 데이터 비율
        n_iter_no_change=10,                # 개선 없는 에포크 허용 수
        momentum=0.9,                       # SGD solver에서 사용
        nesterovs_momentum=True,            # Nesterov Momentum 사용
        warm_start=False,                   # True면 이전 학습에서 이어서 학습
        random_state=42,
        verbose=True                        # 학습 과정 출력
    )
)

mlp_clf.fit(X_tr, y_tr)
print(classification_report(y_te, mlp_clf.predict(X_te)))

# 학습 손실 곡선
mlp_estimator = mlp_clf.named_steps['mlpclassifier']
print(f'학습 에포크 수: {mlp_estimator.n_iter_}')
print(f'최종 손실: {mlp_estimator.loss_:.4f}')
```

### MLPRegressor — 회귀

```python
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mlp_reg = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',       # 은닉층은 relu
        solver='adam',
        alpha=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
)

# 사인 곡선 데이터 (비선형 회귀)
X_reg = np.linspace(0, 4*np.pi, 500).reshape(-1, 1)
y_reg = np.sin(X_reg.ravel()) + np.random.randn(500) * 0.1

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2)
mlp_reg.fit(X_tr_r, y_tr_r)

y_pred_r = mlp_reg.predict(X_te_r)
print(f'RMSE: {mean_squared_error(y_te_r, y_pred_r, squared=False):.4f}')
print(f'R²:   {r2_score(y_te_r, y_pred_r):.4f}')
```

### BernoulliRBM — 제한 볼츠만 머신

슬라이드에 언급된 BernoulliRBM은 일반 MLP와 다른 비지도 학습 모델이다.

```
RBM (Restricted Boltzmann Machine):
  - 에너지 기반 확률 모델
  - 입력 데이터의 잠재 표현을 학습
  - 주로 사전 학습(Pre-training)이나 특성 추출에 사용
  - BernoulliRBM: 이진(0/1) 입력에 특화

활용:
  RBM으로 특성 추출 → LogisticRegression 분류
```

```python
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

digits = load_digits()
X_d = minmax_scale(digits.data)   # [0,1] 범위로 정규화 (이진화 전처리)
y_d = digits.target

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

# RBM → LogisticRegression 파이프라인
rbm_pipe = Pipeline([
    ('rbm', BernoulliRBM(n_components=100, learning_rate=0.01,
                          n_iter=20, random_state=42, verbose=True)),
    ('lr',  LogisticRegression(max_iter=1000))
])

rbm_pipe.fit(X_tr_d, y_tr_d)
print(f'RBM+LR 정확도: {rbm_pipe.score(X_te_d, y_te_d):.4f}')
```

---

## 10. 주요 하이퍼파라미터와 의미

### 네트워크 구조 파라미터

**hidden_layer_sizes**: 은닉층의 구조를 결정한다.

```python
hidden_layer_sizes=(100,)          # 은닉층 1개, 뉴런 100개
hidden_layer_sizes=(128, 64)       # 은닉층 2개, 각 128/64개
hidden_layer_sizes=(256, 128, 64)  # 은닉층 3개, 점점 좁아지는 구조

"넓고 얕은 vs 좁고 깊은":
  넓고 얕음: 단순한 패턴, 빠른 학습
  좁고 깊음: 복잡한 패턴, 위계적 특성 학습
```

### 학습 관련 파라미터

**alpha (L2 정규화)**:

```
alpha가 크면: 강한 정규화 → 가중치 축소 → 과적합 방지, 단순한 모델
alpha가 작으면: 약한 정규화 → 복잡한 모델 가능 → 과적합 위험
기본값: 0.0001
```

**learning_rate_init**:

```
너무 크면: 최적점을 지나쳐 발산
너무 작으면: 수렴이 매우 느림
일반적으로 0.001 ~ 0.0001 사용
```

**batch_size='auto'**:

```
'auto': min(200, n_samples) 자동 선택
작은 배치: 빠른 업데이트, 노이즈 많음, 일반화 가능성 높음
큰 배치:  안정적이지만 느림, 메모리 많이 필요
```

**learning_rate 옵션**:

```
'constant':   학습률 고정 (기본값)
'invscaling': 에포크가 지남에 따라 서서히 감소
'adaptive':   검증 손실이 개선되지 않으면 자동으로 감소
```

**warm_start=True**: 이전 fit() 결과에서 이어서 학습한다. 점진적 학습(Incremental Learning)에 유용하다.

---

## 11. 과적합 방지 방법

### Dropout

학습 중 무작위로 일부 뉴런을 비활성화한다. sklearn MLP는 기본 지원하지 않지만 개념 이해가 중요하다.

```
Dropout:
  학습 시: 각 뉴런을 p 확률로 비활성화 → 다양한 서브네트워크 학습
  예측 시: 모든 뉴런 사용, 가중치에 (1-p) 곱

효과:
  - 특정 뉴런에 과도하게 의존하는 것 방지
  - 앙상블 효과 (많은 서브네트워크의 평균)
```

### 조기 종료 (Early Stopping)

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    early_stopping=True,         # 조기 종료 활성화
    validation_fraction=0.1,     # 검증 데이터 비율
    n_iter_no_change=10,         # 10 에포크 동안 개선 없으면 중단
    tol=1e-4                     # 최소 개선 임계값
)
```

### L2 정규화 (Weight Decay)

```
alpha 파라미터로 제어:
  손실 = 기존 손실 + alpha × Σw²
  
  큰 가중치에 패널티 → 모델 단순화 → 과적합 방지
```

---

## 12. 손실 곡선 분석 및 시각화

```python
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    )
)
mlp.fit(X_tr, y_tr)

clf = mlp.named_steps['mlpclassifier']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 손실 곡선
axes[0].plot(clf.loss_curve_, label='훈련 손실', color='blue')
if clf.validation_scores_ is not None:
    # validation_scores_: 검증 정확도
    ax2 = axes[0].twinx()
    ax2.plot(clf.validation_scores_, label='검증 정확도', color='orange', linestyle='--')
    ax2.set_ylabel('검증 정확도')
    ax2.legend(loc='lower right')
axes[0].set_xlabel('에포크')
axes[0].set_ylabel('훈련 손실')
axes[0].set_title('손실 및 검증 정확도 곡선')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# 가중치 분포 (첫 번째 층)
weights = clf.coefs_[0].ravel()
axes[1].hist(weights, bins=50, color='steelblue', alpha=0.7)
axes[1].set_xlabel('가중치 값')
axes[1].set_ylabel('빈도')
axes[1].set_title(f'첫 번째 층 가중치 분포\n(평균={weights.mean():.3f}, 표준편차={weights.std():.3f})')

plt.tight_layout()
plt.show()
```

---

## 13. 전체 학습 흐름 정리

```
모델 정의
   ↓ 입력층 크기 = 특성 수
   ↓ 은닉층 수, 각 층의 뉴런 수
   ↓ 출력층 크기 = 클래스 수(분류) or 1(회귀)
   ↓
가중치 초기화 (랜덤)
   ↓
순전파 (Forward Pass)
   ↓ z = Wx + b  (선형 결합)
   ↓ a = f(z)    (활성화 함수)
   ↓ 반복 (층 수만큼)
   ↓ 최종 출력 ŷ
   ↓
손실 계산
   ↓ 회귀: MSE = (ŷ - y)²
   ↓ 분류: Cross-Entropy = -y log ŷ
   ↓
역전파 (Backward Pass)
   ↓ 체인 룰: ∂E/∂w = ∂E/∂ŷ × ∂ŷ/∂w
   ↓ 각 가중치에 대한 기울기 계산
   ↓
가중치 업데이트 (자동)
   ↓ w ← w - α × ∂E/∂w
   ↓ Solver: Adam, SGD, lbfgs
   ↓
에포크 반복 → 수렴 또는 Early Stopping
```

### 파라미터 요약표

|파라미터|설명|기본값|튜닝 방향|
|---|---|---|---|
|`hidden_layer_sizes`|은닉층 구조|`(100,)`|복잡한 데이터면 더 깊게|
|`activation`|활성화 함수|`'relu'`|은닉층은 relu 권장|
|`solver`|최적화 방법|`'adam'`|대용량이면 sgd, 소용량이면 lbfgs|
|`alpha`|L2 정규화|`0.0001`|과적합이면 증가|
|`batch_size`|배치 크기|`'auto'`|메모리에 따라 조정|
|`learning_rate_init`|초기 학습률|`0.001`|발산하면 감소, 느리면 증가|
|`max_iter`|최대 에포크|`200`|early_stopping과 함께 크게 설정|
|`early_stopping`|조기 종료|`False`|항상 True 권장|
|`momentum`|모멘텀 계수|`0.9`|SGD 사용 시 적용|
|`warm_start`|이어서 학습|`False`|점진적 학습 시 True|