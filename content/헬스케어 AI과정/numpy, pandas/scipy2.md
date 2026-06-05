---
title: 035 scipy2
tag:
  - 헬스케어 ai
  - scipy
description: 2605027 수업 내용 정리
---

# SciPy와 모델: 수치해석의 실전 적용

SciPy의 각 기능은 독립적으로 존재하는 것이 아니라 서로 연결되어 실제 문제 해결에 쓰인다.  
이 문서는 **미분 → 최적화 → ML/DL** 흐름을 중심으로, 각 개념이 왜 필요하고 어떻게 연결되는지를 정리한다.

```
미분(differentials)       기울기를 계산하는 도구
        ↓
최적화(optimization)      기울기로 파라미터를 업데이트하는 방법
        ↓
ML/DL 모델 학습           최적화로 손실 함수를 최소화하는 과정
```

수치해석은 **수학적인 문제를 컴퓨터의 산술 연산으로 풀 수 있도록 공식화하는 기법**이다.  
$\sin(x)$, $e^x$ 같은 초월함수는 컴퓨터가 직접 계산할 수 없다. 테일러 급수처럼 다항식으로 근사해서 덧셈·곱셈만으로 계산한다. 이것이 수치해석의 본질이다.

---

## 1. 미분 (Differentials) — 변화를 수치화하는 언어

### 왜 미분이 중요한가

머신러닝 모델을 학습한다는 것은 결국 **손실 함수(Loss Function)를 최소화하는 파라미터를 찾는 것**이다.  
"어느 방향으로 파라미터를 바꿔야 손실이 줄어드는가"를 알려주는 것이 바로 **미분(기울기)** 이다.

미분이 없으면 파라미터를 어떻게 바꿔야 할지 방향을 알 수 없다. 무작위로 시도하는 것과 다를 바가 없어진다.

### 함수 → 평균변화율 → 순간변화율 → 도함수

변화를 이해하는 단계적 개념이다.

```
함수 f(x): x 값에 대응하는 결과

평균변화율: (f(x+h) - f(x)) / h
  ← 두 점 사이의 기울기 (어림)

순간변화율: lim(h→0) (f(x+h) - f(x)) / h
  ← 한 점에서의 기울기 (정확)

도함수 f'(x): 모든 점에서의 순간변화율 함수
  ← "어느 위치에서든 기울기를 계산할 수 있음"
```

### 차수별 본질과 ML에서의 역할

|차수|본질|수학적 의미|ML에서의 역할|
|---|---|---|---|
|**1차**|현재 변화|기울기(Gradient): 함수가 어느 방향으로, 얼마나 빠르게 변하는가|경사하강법의 이동 방향 결정|
|**2차**|변화의 구조|곡률(Curvature): 기울기 자체가 얼마나 빠르게 변하는가|뉴턴법의 보폭 자동 조절, 안장점 탐지|
|**3차**|구조 변화|변곡점: 곡률이 오목에서 볼록으로 바뀌는 지점|고급 최적화 분석, 수렴 속도 분석|

```
f(x) = x³ - 3x² + 2x 예시:

f(x)   = x³ - 3x² + 2x  ← 원함수
f'(x)  = 3x² - 6x + 2   ← 1차: 어디서 증가/감소하는지 (f'=0이 극값)
f''(x) = 6x - 6          ← 2차: 어디서 오목/볼록 바뀌는지 (f''=0이 변곡점)
f'''(x) = 6              ← 3차: 곡률 변화율 (이 함수에서는 상수)

f'(x) = 0 → 3x² - 6x + 2 = 0 → x ≈ 0.42, 1.58 (극값 위치)
f''(x) = 0 → 6x - 6 = 0 → x = 1 (변곡점: 여기서 볼록↔오목 전환)
f''(0.42) > 0 → 극솟값 / f''(1.58) < 0 → 극댓값
```

```python
from scipy.misc import derivative
from scipy.optimize import brentq
import numpy as np

f  = lambda x: x**3 - 3*x**2 + 2*x
f1 = lambda x: derivative(f, x, dx=1e-6, n=1)   # 1차 미분
f2 = lambda x: derivative(f, x, dx=1e-6, n=2)   # 2차 미분

# 극값 찾기: f'(x) = 0
x1 = brentq(f1, 0, 1)    # 구간 [0,1]에서 f'=0인 점
x2 = brentq(f1, 1, 2)    # 구간 [1,2]에서 f'=0인 점
print(f'극값: x={x1:.4f} (f\'\'={f2(x1):.2f} > 0 → 극솟값)')
print(f'극값: x={x2:.4f} (f\'\'={f2(x2):.2f} < 0 → 극댓값)')

# 변곡점 찾기: f''(x) = 0
x_infl = brentq(f2, 0, 2)
print(f'변곡점: x={x_infl:.4f}')
```

### 편미분과 그래디언트 (Gradient)

ML 모델은 파라미터가 수천~수억 개다. 이 경우 **각 파라미터에 대한 편미분을 동시에 계산**해야 한다.

편미분 $\frac{\partial f}{\partial x_i}$는 다른 변수는 고정하고 $x_i$만 변화시켰을 때의 기울기다.

이 편미분들을 벡터로 묶은 것이 **그래디언트**다.

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

**그래디언트는 함수값이 가장 빠르게 증가하는 방향을 가리킨다.**  
따라서 경사하강법은 그래디언트의 **반대 방향**($-\nabla f$)으로 이동해 최솟값을 찾는다.

```
2변수 함수 f(x₁, x₂) = x₁² + 2x₂² + x₁x₂ 에서:

∂f/∂x₁ = 2x₁ + x₂   ← x₁ 방향 기울기
∂f/∂x₂ = 4x₂ + x₁   ← x₂ 방향 기울기

(x₁, x₂) = (1, 2)에서:
∇f = (2×1 + 2, 4×2 + 1) = (4, 9)   ← 이 방향이 가장 가파른 오르막
경사하강법: 다음 위치 = (1, 2) - α×(4, 9)
```

```python
from scipy.optimize import approx_fprime
import numpy as np

def f_multi(x):
    return x[0]**2 + 2*x[1]**2 + x[0]*x[1]

x0 = np.array([1.0, 2.0])
grad = approx_fprime(x0, f_multi, epsilon=1e-6)
print(f'그래디언트: {grad}')   # ≈ [4.0, 9.0]

# 수동 경사하강법 (학습률 0.1)
x = x0.copy()
alpha = 0.1
for i in range(20):
    g = approx_fprime(x, f_multi, 1e-6)
    x = x - alpha * g
    if i % 5 == 0:
        print(f'iter {i:2d}: x={x}, f={f_multi(x):.6f}')
```

### 헤시안 행렬 (Hessian Matrix)

2차 편미분들을 모은 행렬이다. **함수의 곡률 구조(landscape)** 를 완전히 기술한다.

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} \end{bmatrix}$$

헤시안의 **고유값** 으로 현재 점의 성격을 판단한다.

|헤시안 고유값|현재 점의 성격|설명|
|---|---|---|
|모두 양수 (양의 정부호)|**극솟값 (Local Minimum)**|모든 방향으로 올라감 → 진짜 최솟값|
|모두 음수 (음의 정부호)|**극댓값 (Local Maximum)**|모든 방향으로 내려감|
|양수와 음수 혼재|**안장점 (Saddle Point)**|어떤 방향으로는 최솟값, 다른 방향으로는 최댓값|

```
안장점의 문제:
  기울기(gradient)가 0이다 → 경사하강법이 멈춤
  하지만 극솟값이 아니다 → 원하는 해가 아님

딥러닝에서 안장점:
  파라미터가 수억 개 → 헤시안의 모든 고유값이 양수일 확률이 극히 낮음
  → 실제로는 극솟값보다 안장점이 훨씬 더 많음
  → SGD의 노이즈가 오히려 안장점 탈출에 도움을 줌
```

```python
# 헤시안 수치 계산
def hessian(f, x, eps=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = ej[j] = eps
            H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*eps**2)
    return H

x0 = np.array([1.0, 2.0])
H = hessian(f_multi, x0)
eigenvalues = np.linalg.eigvals(H)
print(f'헤시안:\n{H}')
print(f'고유값: {eigenvalues}')
print(f'판단: {"극솟값" if all(eigenvalues > 0) else "안장점 또는 극댓값"}')
```

---

## 2. 최적화 (Optimization) — 파라미터를 찾는 방법들

최적화는 **손실 함수를 최소화하는 파라미터 값을 찾는 과정**이다.  
모든 ML/DL 학습의 핵심이다.

### 방법별 사용 정보 비교

|방법|사용하는 정보|보폭 결정|고차원 적합성|
|---|---|---|---|
|**경사하강법**|1차 미분 (Gradient)만|학습률 $\alpha$ 직접 설정|매우 적합 (수억 파라미터도 가능)|
|**뉴턴법**|1차 + 2차 미분 (Hessian)|자동으로 최적 보폭 계산|부적합 ($n^2$ 크기 Hessian 필요)|
|**준뉴턴법 (BFGS)**|Gradient만으로 Hessian 근사|근사된 Hessian으로 보폭 결정|중간 규모 적합|
|**L-BFGS**|최근 m번의 Gradient만 저장|메모리 효율적 Hessian 근사|대규모 적합|

### 지역해 vs 전역해 문제

#### 지역해 (Local Optimum)

대부분의 최적화 알고리즘은 초기값에서 출발해 **가장 가까운 최솟값(지역해)** 으로 수렴한다.

```
함수 풍경 (loss landscape) 예시:

Loss
  │        ___
  │       /   \     <- 지역 최댓값
  │  ____/     \____/‾‾‾‾
  │ /     ↑         ↑
  │/   지역 최솟값   전역 최솟값
  └─────────────────────────── 파라미터

초기값을 어디서 시작하느냐에 따라 다른 골짜기로 떨어짐
```

뉴턴법은 곡률 정보로 큰 보폭으로 이동하므로 지역해에 빠질 위험이 크다.  
SGD는 배치마다 노이즈가 있어서 오히려 지역해를 탈출하는 데 유리하다.

```python
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# 여러 극솟값이 있는 함수 (멀티모달)
# 이런 함수에서는 초기값에 따라 완전히 다른 결과가 나옴
def multimodal(x):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0]**2))

print("=== 초기값에 따른 수렴 비교 ===")
for x0 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    result = minimize(multimodal, [x0], method='BFGS')
    print(f'초기값={x0:+.1f} → 수렴점={result.x[0]:+.4f}, '
          f'Loss={result.fun:.4f}, 반복={result.nit}회')

# 시각화: 어떤 초기값이 어떤 해에 수렴하는지
x_range = np.linspace(-3, 3, 300)
y_vals  = [multimodal([xi]) for xi in x_range]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_range, y_vals, 'b-', linewidth=2, label='손실 함수')
colors = ['red', 'green', 'orange', 'purple', 'brown']
for x0, c in zip([-2, -1, 0, 1, 2], colors):
    res = minimize(multimodal, [x0], method='BFGS')
    ax.axvline(x=x0, color=c, linestyle=':', alpha=0.5, label=f'초기값 {x0}')
    ax.scatter([res.x[0]], [res.fun], color=c, s=100, zorder=5)
ax.legend(loc='upper right')
ax.set_title('초기값에 따른 다른 지역해 수렴')
plt.show()
```

#### 전역해 (Global Optimum) — brute, differential_evolution

모든 가능한 해를 탐색하거나 생물 진화를 모방해서 **전역 최솟값**을 찾는다.  
지역해에 갇히지 않지만 계산 비용이 크다.

```python
from scipy.optimize import brute, differential_evolution

# brute: 격자 탐색
# [-3, 3] 구간을 0.05 간격으로 모두 평가 → 가장 낮은 점 반환
ranges = (slice(-3, 3, 0.05),)
result_brute = brute(lambda x: multimodal([x[0]]), ranges, finish=None)
print(f'brute 전역해: x ≈ {result_brute[0]:.4f}')

# differential_evolution: 진화 알고리즘
# 무작위 후보군을 진화시켜 전역해 탐색. brute보다 훨씬 효율적
result_de = differential_evolution(
    lambda x: multimodal(x),
    bounds=[(-3, 3)],
    seed=42,
    maxiter=1000,
    tol=1e-8
)
print(f'진화 알고리즘 전역해: x ≈ {result_de.x[0]:.4f}, Loss={result_de.fun:.6f}')
print(f'함수 평가 횟수: {result_de.nfev}회')
```

#### 곡선 피팅 (Curve Fitting) — curve_fit

실험 데이터가 있고 **그 데이터를 설명하는 함수의 파라미터**를 찾을 때 사용한다.  
함수의 형태는 도메인 지식으로 가정하고, 파라미터만 데이터에 맞게 최적화한다.

회귀분석과 비슷하지만, 선형 모델에 한정되지 않고 **임의의 비선형 함수**에도 적용할 수 있다.

```python
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# 배경: 방사성 원소의 붕괴를 측정했다
# 이론: 지수 감쇠 모델 A×e^(-kt) 를 따를 것이라 예측
t_data = np.linspace(0, 4, 50)
y_true = 3 * np.exp(-0.5 * t_data)          # 실제 (A=3, k=0.5)
y_data = y_true + np.random.normal(0, 0.15, 50)  # 측정 노이즈 추가

# 피팅할 함수 형태를 정의 (파라미터 A, k를 추정)
def exp_decay(t, A, k):
    """지수 감쇠: 시간 t에서의 양 = A × e^(-kt)"""
    return A * np.exp(-k * t)

# curve_fit: 잔차 제곱합을 최소화하는 A, k를 찾음 (내부적으로 Levenberg-Marquardt)
# p0: 초기 추정값 (없으면 [1, 1, ...])
# pcov: 파라미터의 공분산 행렬 (대각 = 분산 → 제곱근이 표준편차)
popt, pcov = curve_fit(exp_decay, t_data, y_data, p0=[1.0, 0.1])
perr = np.sqrt(np.diag(pcov))   # 각 파라미터의 표준편차

print(f'추정: A = {popt[0]:.3f} ± {perr[0]:.3f}  (실제: 3.000)')
print(f'추정: k = {popt[1]:.3f} ± {perr[1]:.3f}  (실제: 0.500)')
# ±는 68% 신뢰구간 (1σ)

# 95% 신뢰구간 = ±1.96σ
print(f'A의 95% CI: [{popt[0]-1.96*perr[0]:.3f}, {popt[0]+1.96*perr[0]:.3f}]')

# 시각화
t_fine = np.linspace(0, 4, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(t_data, y_data, alpha=0.4, label='측정 데이터', zorder=3)
ax.plot(t_fine, exp_decay(t_fine, *popt), 'r-', linewidth=2,
        label=f'피팅: A={popt[0]:.2f}, k={popt[1]:.3f}')
ax.plot(t_fine, y_true, 'g--', linewidth=1.5, label='실제 함수')
ax.set_xlabel('시간 (t)')
ax.set_ylabel('측정값')
ax.legend()
ax.set_title('지수 감쇠 곡선 피팅')
plt.show()
```

#### 방정식의 근 찾기 (Root Finding)

$f(x) = 0$이 되는 $x$를 찾는다. 최솟값 탐색($f'(x)=0$)과 구분하는 것이 중요하다.

```
최적화: f(x)를 최소화하는 x 찾기 (f'(x) = 0)
근 찾기: f(x) = 0인 x 찾기

예: f(x) = x² - 2 = 0 → √2 를 수치적으로 구함
   f'(x) = 2x → 뉴턴법: x₁ = x₀ - f(x₀)/f'(x₀) = x₀ - (x₀²-2)/(2x₀)
```

|방법|알고리즘|장점|단점|
|---|---|---|---|
|`brentq`|이분법 기반|반드시 수렴, 안정적|구간 내 근이 존재해야 함|
|`newton`|뉴턴-랩슨법|빠른 수렴 (이차 수렴)|초기값에 민감, 발산 가능|
|`fsolve`|뉴턴법 기반 다변수|연립방정식 처리|수렴 보장 없음|

```python
from scipy.optimize import brentq, newton, fsolve
import numpy as np

# 단변수 방정식: x³ - 2x - 5 = 0 의 근
f  = lambda x: x**3 - 2*x - 5
df = lambda x: 3*x**2 - 2

# brentq: [2, 3] 구간에서 이분법으로 탐색
# f(2) = -1 < 0, f(3) = 16 > 0 → 구간 안에 근 존재
root_brent = brentq(f, 2, 3)
print(f'brentq:  {root_brent:.8f}')   # ≈ 2.09455148

# newton: 초기값 x₀=2.5에서 뉴턴-랩슨 반복
root_newton = newton(f, x0=2.5, fprime=df)
print(f'newton:  {root_newton:.8f}')

# 검증
print(f'f(root) = {f(root_brent):.2e}')  # ≈ 0 이어야 함

# 연립방정식: x² + y² = 4, x - y = 1
# 두 방정식이 동시에 0이 되는 (x, y) 찾기
def system(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 4   # x² + y² = 4 (원)
    eq2 = x - y - 1          # x - y = 1 (직선)
    return [eq1, eq2]

# 두 교점이 있으므로 초기값에 따라 다른 해
for x0 in [[1, 0], [-1, -2]]:
    sol = fsolve(system, x0)
    print(f'해: x={sol[0]:.4f}, y={sol[1]:.4f}  →  검증: {[f"{v:.2e}" for v in system(sol)]}')
```

---

## 3. ML/DL과 최적화 방법의 연결

### 왜 분야마다 다른 최적화 방법을 쓰는가

최적화 방법 선택의 핵심 기준은 두 가지다.

1. **파라미터 수 (n)**: 헤시안 행렬의 크기가 $n \times n$이므로 n이 크면 헤시안 사용 불가
2. **함수의 부드러운 정도**: 불연속적이거나 노이즈가 많으면 2차 미분이 의미 없음

```
파라미터 n이 증가할 때:

n=100:     헤시안 = 100×100 = 10,000개 원소 → 계산 가능
n=10,000:  헤시안 = 10,000×10,000 = 1억개 → 느리지만 가능
n=1,000만: 헤시안 = 10^14개 → 불가능 (딥러닝 파라미터 수준)

→ 딥러닝은 반드시 1차 미분(Gradient)만 사용해야 함
```

|분야|주 사용 방법|이유|
|---|---|---|
|**딥러닝**|SGD / Adam|파라미터 수 수억 개 → Hessian 계산 불가, 미니배치로 빠른 업데이트|
|**통계모델 (ARIMA 등)**|Newton / BFGS|파라미터 수 적음(10~100개) → 2차 미분 활용 가능, 정확한 수렴 필요|
|**Logistic Regression**|L-BFGS|중규모(수천~수만 특성) → 메모리 효율적 준뉴턴, sklearn 기본 solver|
|**SVM**|Quasi-Newton / SMO|커널 함수 최적화, 제약 조건 있는 이차계획법|
|**대규모 선형 모델**|L-BFGS|수천~수만 파라미터, Hessian 근사로 메모리 절약|
|**물리 시뮬레이션**|Newton|고정밀 수치 적분 필요, 소규모 방정식|

### SGD vs Adam — 딥러닝 옵티마이저 비교

```
SGD (확률적 경사하강법):
  w ← w - α × ∇L(w)
  단순하고 빠름, 하지만 학습률 설정이 어렵고 진동이 심함

Momentum SGD:
  v ← β×v + ∇L(w)    ← 이전 이동 방향을 관성으로 누적
  w ← w - α×v
  안정적으로 수렴, 안장점 탈출에 유리

Adam (Adaptive Moment Estimation):
  m ← β₁×m + (1-β₁)×∇L    ← 1차 모멘트 (방향)
  v ← β₂×v + (1-β₂)×(∇L)² ← 2차 모멘트 (크기)
  w ← w - α × m/√v
  각 파라미터마다 학습률을 자동 조절 → 현재 딥러닝 표준
```

```python
from scipy.optimize import minimize
import numpy as np

# 로지스틱 회귀 손실 함수를 다른 방법으로 최적화
np.random.seed(42)
n_samples, n_features = 200, 10
X = np.random.randn(n_samples, n_features)
true_w = np.random.randn(n_features)
y = (X @ true_w > 0).astype(float)

def sigmoid(z): return 1 / (1 + np.exp(-z))

def log_loss(w, X, y):
    """이진 교차 엔트로피 손실"""
    p = sigmoid(X @ w)
    return -np.mean(y * np.log(p + 1e-8) + (1-y) * np.log(1-p + 1e-8))

def log_loss_grad(w, X, y):
    """그래디언트 (해석적으로 계산)"""
    p = sigmoid(X @ w)
    return X.T @ (p - y) / len(y)

w0 = np.zeros(n_features)
print("방법별 수렴 비교:")
print(f"{'방법':15s} {'반복':>6s} {'최종 손실':>12s}")
print("-" * 40)

for method in ['CG', 'BFGS', 'L-BFGS-B', 'Newton-CG']:
    result = minimize(log_loss, w0,
                      args=(X, y),
                      method=method,
                      jac=log_loss_grad)
    print(f'{method:15s} {result.nit:6d} {result.fun:12.6f}')
```

### 준뉴턴법 (BFGS) — 뉴턴법과 경사하강법의 절충

```
뉴턴법:     H⁻¹ 계산 → n×n 행렬 필요 → 비용: O(n³)
경사하강법: H 무시 → 방향만 정확, 보폭은 학습률에 의존
BFGS:       이전 그래디언트들로 H를 점진적으로 근사
            → O(n²) 메모리, O(n²) 계산 → 중간 크기 문제에 최적

L-BFGS:     최근 m개의 그래디언트만 저장 (m ≈ 10~20)
            → O(m×n) 메모리 → 대규모 문제도 가능
```

---

## 4. 적분 (Integration)과 몬테카를로 샘플링

### 수치 적분의 통계적 의미

적분은 **함수 아래의 넓이**다. 확률 이론에서 확률밀도함수(PDF)를 특정 구간에서 적분하면 그 구간의 확률이 된다.

$$P(a \leq X \leq b) = \int_a^b f(x) dx$$

정규분포의 68-95-99.7 법칙도 사실 PDF를 $\mu \pm 1\sigma$, $\mu \pm 2\sigma$, $\mu \pm 3\sigma$ 구간에서 적분한 결과다.

```python
from scipy import integrate, stats
import numpy as np

# 정규분포 N(μ=0, σ=1)의 특정 구간 확률 계산
dist = stats.norm(0, 1)

# 방법 1: cdf 사용 (내부적으로 수치 적분)
prob_cdf = dist.cdf(1) - dist.cdf(-1)
print(f'cdf로 계산 (±1σ): {prob_cdf:.6f}')   # ≈ 0.6827

# 방법 2: quad로 직접 적분
prob_quad, error = integrate.quad(dist.pdf, -1, 1)
print(f'quad로 계산 (±1σ): {prob_quad:.6f} (오차: {error:.2e})')

# 68-95-99.7 법칙 확인
for sigma in [1, 2, 3]:
    prob, _ = integrate.quad(dist.pdf, -sigma, sigma)
    print(f'±{sigma}σ 확률: {prob*100:.2f}%')

# 꼬리 확률 계산 (p-value의 원리)
# "표준정규분포에서 z > 1.96일 확률"
tail_prob, _ = integrate.quad(dist.pdf, 1.96, np.inf)
print(f'P(Z > 1.96) = {tail_prob:.4f}')  # ≈ 0.025 (5% 양측검정의 한쪽)
```

### 몬테카를로 샘플링 (Monte Carlo Sampling)

고차원이거나 해석적으로 적분하기 어려울 때 **무작위 샘플링으로 적분을 근사**하는 방법이다.

**핵심 아이디어**:

$$\int_a^b f(x)dx \approx (b-a) \cdot \frac{1}{N}\sum_{i=1}^{N}f(x_i), \quad x_i \sim \text{Uniform}(a,b)$$

구간에 무작위로 점을 뿌리고, "그 점에서 함수값의 평균 × 구간 길이"로 적분을 추정한다.

```
직관 — 원의 넓이로 π 추정:

  정사각형 [-1,1]×[-1,1] 안에 무작위로 점을 던짐
  원(x²+y²≤1) 안에 들어가는 비율 ≈ π/4

  N=100개:    π ≈ 3.16 (오차 큼)
  N=10,000개: π ≈ 3.141 (오차 줄어듦)
  N=1,000,000개: π ≈ 3.14159 (오차 더 줄어듦)
  
  오차가 줄어드는 속도: 1/√N (샘플 4배 → 오차 절반)
```

```python
import numpy as np
import matplotlib.pyplot as plt

# ── π 추정 (기하학적 몬테카를로) ──────────────────────────
np.random.seed(42)
N = 100_000

x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)
inside = (x**2 + y**2) <= 1   # 단위원 내부 판별

pi_est = 4 * inside.mean()   # 비율 × 정사각형 넓이(4)
print(f'N={N:,}개로 π 추정: {pi_est:.5f} (오차: {abs(pi_est - np.pi):.5f})')

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 점들 시각화
n_vis = 3000
axes[0].scatter(x[:n_vis][inside[:n_vis]], y[:n_vis][inside[:n_vis]],
                c='steelblue', s=1, alpha=0.5, label='원 내부')
axes[0].scatter(x[:n_vis][~inside[:n_vis]], y[:n_vis][~inside[:n_vis]],
                c='salmon', s=1, alpha=0.5, label='원 외부')
theta = np.linspace(0, 2*np.pi, 100)
axes[0].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
axes[0].set_aspect('equal')
axes[0].set_title(f'몬테카를로 π 추정 (N={n_vis})')
axes[0].legend()

# 오른쪽: 수렴 시각화
ns = np.logspace(1, 5, 50).astype(int)
pi_estimates = []
for n in ns:
    x_s = np.random.uniform(-1, 1, n)
    y_s = np.random.uniform(-1, 1, n)
    pi_estimates.append(4 * (x_s**2 + y_s**2 <= 1).mean())

axes[1].semilogx(ns, pi_estimates, 'b-', alpha=0.7, label='MC 추정')
axes[1].axhline(np.pi, color='red', linestyle='--', label=f'실제 π={np.pi:.4f}')
axes[1].fill_between(ns,
                      np.pi - 2/np.sqrt(ns),
                      np.pi + 2/np.sqrt(ns),
                      alpha=0.2, color='red', label='±2/√N 오차 범위')
axes[1].set_xlabel('샘플 수 (N)')
axes[1].set_ylabel('π 추정값')
axes[1].set_title('샘플 수에 따른 수렴 (1/√N 속도)')
axes[1].legend()
plt.tight_layout()
plt.show()

# ── 함수 적분 ───────────────────────────────────────────
# ∫₀^π sin(x) dx = 2 (해석적 값)
N = 1_000_000
x_samples = np.random.uniform(0, np.pi, N)
integral_mc = np.pi * np.mean(np.sin(x_samples))
print(f'∫sin(x)dx ≈ {integral_mc:.6f} (정확값: 2.000000)')

# ── 고차원 적분 (몬테카를로의 진가) ──────────────────────
# 10차원 단위 구의 부피 (수치 적분으로는 10^n번 평가 필요)
dim = 10
N   = 5_000_000
samples = np.random.uniform(-1, 1, (N, dim))
inside_nd = (samples**2).sum(axis=1) <= 1
volume = (2**dim) * inside_nd.mean()   # 전체 박스 부피 × 내부 비율
exact  = np.pi**(dim/2) / np.math.gamma(dim/2 + 1)
print(f'{dim}차원 구 부피 추정: {volume:.3f} (정확값: {exact:.3f})')
```

### 몬테카를로 수렴 원리

$$\text{오차} \sim \frac{1}{\sqrt{N}}$$

|샘플 수|예상 오차|
|---|---|
|100|±0.1|
|10,000|±0.01|
|1,000,000|±0.001|

핵심: 차원이 높아져도 오차 수렴 속도가 **변하지 않는다**.  
수치 적분(격자법)은 차원 $d$가 늘어날수록 계산량이 $N^d$으로 폭발적으로 증가하지만(차원의 저주),  
몬테카를로는 **차원에 무관하게 $1/\sqrt{N}$** 으로 수렴한다.

|방법|수렴 속도|차원 증가 시|적합한 경우|
|---|---|---|---|
|수치 적분 (quad)|고정밀도|계산량 지수 증가|저차원 (1~3D), 부드러운 함수|
|몬테카를로|$1/\sqrt{N}$|변화 없음|고차원, 복잡한 영역, 확률 시뮬레이션|

---

## 5. 이미지 컨볼루션 — 슬라이드 예시 상세 설명

### 컨볼루션 연산의 직관

커널(필터)은 **"무엇을 감지할 것인가"를 정의하는 템플릿**이다.  
이미지 위를 슬라이딩하면서 "이 위치가 템플릿과 얼마나 비슷한가"를 수치로 계산한다.

```
슬라이드 예시 직접 계산:

Input (4×4):    Kernel (3×3):
 2  2  2  2      1  1  1
 5  5  6  6  *   1  2  1   =  47 (왼쪽 위 첫 번째 출력)
 6  7  7  7      1  1  1
 8  7  9  9

왼쪽 위 3×3 영역과 커널의 원소별 곱 후 합산:
  1×2 + 1×2 + 1×2   = 6   (1행)
+ 1×5 + 2×5 + 1×6   = 21  (2행: 중앙 가중치 2)
+ 1×6 + 1×7 + 1×7   = 20  (3행)
= 6 + 21 + 20 = 47  ✓
```

```python
from scipy import ndimage
import numpy as np

img = np.array([[2, 2, 2, 2],
                [5, 5, 6, 6],
                [6, 7, 7, 7],
                [8, 7, 9, 9]], dtype=float)

kernel = np.array([[1, 1, 1],
                   [1, 2, 1],
                   [1, 1, 1]], dtype=float)

# correlate: 딥러닝 방식 (커널 회전 없이 그대로 슬라이딩)
# mode='constant', cval=0: 경계 밖은 0으로 패딩
result = ndimage.correlate(img, kernel, mode='constant', cval=0)
print("필터링 결과:")
print(result.astype(int))
# 내부 값은 47이 나와야 함
# (경계 처리 방식에 따라 가장자리 값은 다를 수 있음)

# 경계 처리 방식 비교
print("\n경계 처리 방식 비교 (중앙 2×2만 동일, 가장자리가 다름):")
for mode in ['constant', 'reflect', 'nearest', 'wrap']:
    r = ndimage.correlate(img, kernel, mode=mode)
    print(f'{mode:10s}: {r[0, 0]:.0f} {r[0, 1]:.0f} ...')
```

---

## 6. 푸리에 변환과 주파수 공간

### 파동과 주기성 → 푸리에 변환 → 주파수 분석

모든 복잡한 신호는 **단순한 사인파와 코사인파의 합**으로 분해할 수 있다. (오일러 공식 $e^{i\theta} = \cos\theta + i\sin\theta$가 이 연결의 기반)

```
시간 영역: "언제 어떤 값이었나" (파형)
   ↓ 푸리에 변환
주파수 영역: "어떤 주파수가 얼마나 강한가" (스펙트럼)

예시:
  도레미를 동시에 연주한 소리 → FFT → 각 음의 주파수와 세기
  연간 기온 데이터             → FFT → 1년 주기, 1달 주기 등 탐지
  이미지                       → FFT → 공간 주파수 분석 (JPEG 압축)
```

```python
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

# 시계열 데이터에서 계절성 주기 탐지
t = np.arange(365 * 3)                              # 3년 일별 데이터
y = (3.0 * np.sin(2*np.pi*t / 365)                 # 1년 주기 (강함)
   + 1.0 * np.sin(2*np.pi*t / 30)                  # 30일 주기 (중간)
   + 0.5 * np.sin(2*np.pi*t / 7)                   # 7일 주기 (약함, 주간)
   + 0.3 * np.random.randn(len(t)))                 # 노이즈

# FFT
Y     = fft(y)
freqs = fftfreq(len(t), d=1)          # 단위: 1/일
amp   = np.abs(Y) / len(t)            # 진폭

# 양의 주파수만 (실수 신호는 대칭)
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]
amp_pos   = amp[pos_mask] * 2         # 대칭이므로 2배

# 상위 5개 주파수 탐지
top5_idx  = np.argsort(amp_pos)[::-1][:5]
top5_freq = freqs_pos[top5_idx]
top5_amp  = amp_pos[top5_idx]

print("주요 주파수 성분:")
for freq, amplitude in zip(top5_freq, top5_amp):
    period = 1 / freq
    print(f'  주기: {period:6.1f}일  진폭: {amplitude:.3f}')
# 출력: 365일, 30일, 7일 등이 탐지됨

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 시간 영역
axes[0].plot(t[:365], y[:365])
axes[0].set_title('시간 영역 신호 (1년 표시)')
axes[0].set_xlabel('일 (day)')
axes[0].set_ylabel('값')

# 주파수 영역
axes[1].semilogy(1/freqs_pos, amp_pos)  # 주기(일) vs 진폭
axes[1].set_title('주파수 영역 (스펙트럼)')
axes[1].set_xlabel('주기 (일)')
axes[1].set_ylabel('진폭 (log 스케일)')
for p in [365, 30, 7]:
    axes[1].axvline(x=p, color='red', linestyle='--', alpha=0.7, label=f'{p}일 주기')
axes[1].set_xlim(1, 400)
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## 7. SciPy와 수치해석 전체 연결 구조

```
문제 정의
   ↓
수학적 모델링
   ├─ 선형 관계 → linalg: 연립방정식, 행렬 분해
   │              Ax = b → x = A⁻¹b
   │
   ├─ 변화율 → differentials: 1차(기울기), 2차(곡률)
   │           f'(x): 어느 방향으로 변하는가
   │
   ├─ 최솟값 → optimize: 경사하강법, 뉴턴법, BFGS
   │           argmin f(x): 손실 함수를 최소화하는 파라미터
   │
   ├─ 넓이/확률 → integrate: quad, 몬테카를로
   │              ∫f(x)dx = P(a ≤ X ≤ b)
   │
   ├─ 주기성 → fft: 시간 → 주파수 변환
   │            파장, 계절성, 노이즈 분리
   │
   ├─ 이미지 → ndimage: 컨볼루션, 필터링
   │            커널 슬라이딩으로 특징 감지
   │
   └─ 확률 분포 → stats: PDF/CDF/ppf/rvs, 가설 검정
                  P(X ≤ x), 신뢰구간, t-test
   ↓
수치해법 선택
   파라미터 수 많음 → 경사하강법(SGD/Adam)
   파라미터 수 적음 → 뉴턴법/BFGS
   메모리 제한 있음 → L-BFGS
   미분 불가능      → 진화 알고리즘
   ↓
구현: NumPy(자료구조) + SciPy(알고리즘)
   ↓
ML/DL 모델 학습 및 예측
```