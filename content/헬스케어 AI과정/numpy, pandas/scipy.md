---
title: 034 scipy
tag:
  - 헬스케어 ai
  - scipy
description: 2605026 수업 내용 정리
---

# SciPy와 수치해석: 시각적 이해

수치해석은 **수학적인 문제를 컴퓨터의 산술 연산으로 풀 수 있도록 공식화하는 기법**이다.  
미분방정식, 최적화, 적분처럼 해석적(analytic) 풀이가 어렵거나 불가능한 문제를 **근삿값으로 수치적으로 해결**한다.

SciPy는 NumPy를 기반으로 수치해석 알고리즘을 제공하는 라이브러리다.

```
자료구조(NumPy) + 알고리즘(SciPy)
      ↓
문제 정의 → 수학적 모델 → 수치해법 → 구현
```

---

## 1. 수치해석의 기반: 테일러 급수와 오일러 공식

SciPy의 많은 알고리즘은 두 가지 수학적 도구 위에 서 있다.

### 테일러 급수 (Taylor Series)

임의의 함수를 **다항식의 합으로 근사**하는 방법이다.  
어떤 점 $a$ 근방에서 함수 $f(x)$를 다음처럼 전개한다.

$$p_n(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n = \sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k$$

**수치해석에서의 의미**:  
컴퓨터는 $\sin(x)$, $e^x$ 같은 초월함수를 직접 계산할 수 없다.  
테일러 급수를 통해 이를 덧셈·곱셈만으로 이루어진 다항식으로 바꿔 계산한다.

- 항을 많이 쓸수록 더 정확하지만 계산량이 늘어난다
- 수치미분, 오차 분석의 이론적 기초가 된다
- $a=0$에서 전개하면 **매클로린 급수**라고 한다

```python
import numpy as np
from scipy.special import factorial

# e^x를 테일러 급수로 근사 (x=1, 즉 e 계산)
x = 1.0
n_terms = 20
approx = sum(x**k / factorial(k) for k in range(n_terms))
print(f'근사값: {approx:.10f}')
print(f'실제값: {np.e:.10f}')
```

### 오일러 공식 (Euler's Formula)

$$e^{i\theta} = \cos\theta + i\sin\theta$$

복소수와 삼각함수, 지수함수를 하나로 연결하는 공식이다.  
$\theta = \pi$를 대입하면 $e^{i\pi} + 1 = 0$ (오일러 항등식)이 된다.

**수치해석에서의 의미**:

- **푸리에 변환(FFT)** 의 수학적 기반이다
- 회전 변환, 신호 분석, 양자역학 계산에 핵심적으로 사용된다
- 시간 영역의 신호를 주파수 영역으로 변환하는 원리

---

## 2. 미분과 적분 (Differentiation & Integration)

### 함수 → 평균변화율 → 순간변화율 → 도함수

함수의 변화를 이해하는 단계적 개념이다.

```
함수 f(x)
   ↓
평균변화율: Δy/Δx = (f(x+h) - f(x)) / h   ← 두 점 사이의 기울기
   ↓ h → 0
순간변화율: lim(h→0) (f(x+h) - f(x)) / h  ← 한 점에서의 기울기
   ↓
도함수: f'(x)                               ← 모든 점에서의 기울기 함수
```

### 수치 미분

해석적으로 도함수를 구하기 어려울 때, **아주 작은 $h$로 차분을 계산**해서 근사한다.

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} \quad \text{(중앙 차분, 더 정확)}$$

```python
from scipy.misc import derivative

# 수치 미분: f(x) = x^3의 x=2에서 도함수 (정확한 값: 12)
f = lambda x: x**3
df = derivative(f, x0=2.0, dx=1e-6)
print(f'수치 미분: {df:.6f}')  # ≈ 12.0

# 고차 도함수
d2f = derivative(f, x0=2.0, dx=1e-4, n=2)  # 2차 도함수 (정확한 값: 12)
print(f'2차 미분: {d2f:.6f}')
```

### 수치 적분

$$\int_a^b f(x)dx$$

함수 아래의 **넓이를 수치적으로 계산**한다. 해석적 적분이 어려운 경우에 사용한다.

```python
from scipy import integrate

# 기본 수치 적분 (quad: 가우스 구적법 기반, 고정밀도)
f = lambda x: x**2
result, error = integrate.quad(f, 0, 1)   # [0,1] 구간 적분
print(f'∫x²dx = {result:.6f} (오차: {error:.2e})')  # 정확한 값: 1/3

# 다중 적분
f2d = lambda y, x: x * y
result2d, _ = integrate.dblquad(f2d, 0, 1, 0, 1)   # 이중 적분
print(f'∬xy dxdy = {result2d:.6f}')

# 이미 계산된 데이터 포인트의 적분 (사다리꼴 법칙)
x = np.linspace(0, 1, 100)
y = x**2
result_trap = integrate.trapezoid(y, x)
print(f'사다리꼴 적분: {result_trap:.6f}')
```

---

## 3. scipy.linalg — 선형대수

NumPy의 `linalg`보다 더 많은 기능과 더 안정적인 알고리즘을 제공한다.  
행렬 변환, 분해, 연립방정식 풀이에 사용한다.

### 주요 행렬 변환 (슬라이드 이미지 기준)

|변환|행렬|의미|
|---|---|---|
|스케일링 (Scaling)|$\begin{bmatrix} k & 0 \ 0 & k \end{bmatrix}$|크기를 $k$배 균등 확대/축소|
|불균등 스케일링|$\begin{bmatrix} k_1 & 0 \ 0 & k_2 \end{bmatrix}$|x, y 방향을 다른 비율로 변환|
|회전 (Rotation)|$\begin{bmatrix} \cos\theta & -\sin\theta \ \sin\theta & \cos\theta \end{bmatrix}$|원점 기준 $\theta$ 회전|
|전단 (Horizontal Shear)|$\begin{bmatrix} 1 & k \ 0 & 1 \end{bmatrix}$|수평 방향으로 밀림|
|쌍곡 회전|$\begin{bmatrix} \cosh\varphi & \sinh\varphi \ \sinh\varphi & \cosh\varphi \end{bmatrix}$|쌍곡선 좌표 변환|

이 변환들은 모두 **행렬곱으로 표현**된다. 오일러 공식과 연결되어 회전이 복소수 곱과 동일한 구조를 가진다.

```python
from scipy import linalg
import numpy as np

A = np.array([[2, 1], [1, 3]], dtype=float)

# 역행렬
A_inv = linalg.inv(A)

# 행렬식 (det = 0이면 역행렬 없음)
det = linalg.det(A)
print(f'det(A) = {det:.2f}')

# 연립방정식 Ax = b 풀기
b = np.array([5, 10])
x = linalg.solve(A, b)
print(f'해: {x}')

# LU 분해: A = P·L·U (수치 안정성을 위한 분해)
P, L, U = linalg.lu(A)

# QR 분해: A = Q·R (Q: 직교행렬, R: 상삼각행렬)
Q, R = linalg.qr(A)

# SVD (특이값 분해): A = U·Σ·V^T
U, s, Vt = linalg.svd(A)
print(f'특이값: {s}')

# 고유값 분해
eigenvalues, eigenvectors = linalg.eig(A)
print(f'고유값: {eigenvalues}')
```

### QR 분해와 회귀분석 연결

회귀분석에서 최소제곱법을 계산할 때 QR 분해가 사용된다.  
$Ax = b$를 직접 푸는 것보다 수치적으로 안정적이다.

$$A = QR \Rightarrow x = R^{-1}Q^Tb$$

```python
# 최소제곱 회귀계수 계산 (QR 방식)
X = np.column_stack([np.ones(len(df)), df['x']])  # 절편 포함
coeffs, residuals, rank, sv = linalg.lstsq(X, df['y'])
print(f'회귀계수: {coeffs}')
```

---

### 행렬 분해 (Matrix Decomposition)

행렬 분해는 **복잡한 행렬을 단순한 구조의 행렬들의 곱으로 쪼개는 것**이다.  
연립방정식 풀기, 차원 축소, 데이터 압축, 추천 시스템 등에 광범위하게 활용된다.

```
행렬 A
   ↓ 분해
A = (구조적으로 단순한 행렬들의 곱)
   ↓
계산 효율 ↑  / 수치 안정성 ↑  / 구조 파악 가능
```

---

#### 고유값 분해 (Eigenvalue Decomposition)

**정방행렬** $A$를 다음 형태로 분해한다.

$$A = P \Lambda P^{-1}$$

- $P$: 고유벡터를 열로 가진 행렬 (새로운 축의 방향)
- $\Lambda$: 고유값을 대각 성분으로 가진 대각행렬 (각 축의 스케일)
- $P^{-1}$: $P$의 역행렬

**고유벡터와 고유값의 의미**:

$$A\mathbf{v} = \lambda\mathbf{v}$$

행렬 $A$가 벡터 $\mathbf{v}$에 작용할 때 **방향은 바뀌지 않고 크기만 $\lambda$배 변하는 벡터** $\mathbf{v}$가 고유벡터이고, 그 배수 $\lambda$가 고유값이다.

```
일반 벡터에 A를 곱하면: 방향과 크기 모두 바뀜 → 해석 어려움
고유벡터에 A를 곱하면: 크기만 λ배 변함 → 변환의 본질 파악 가능

예: 공분산 행렬의 고유벡터
    = 데이터가 가장 많이 퍼진 방향 (주성분)
    고유값 = 그 방향으로의 분산 크기
```

**조건**: 정방행렬이어야 하고, 고유값이 모두 다르면 분해 가능 (중복 고유값은 특별 처리 필요)

```python
from scipy import linalg
import numpy as np

A = np.array([[4, 2],
              [1, 3]], dtype=float)

# 고유값 분해
eigenvalues, eigenvectors = linalg.eig(A)
print(f'고유값:    {eigenvalues}')    # [5, 2]
print(f'고유벡터:\n{eigenvectors}')   # 열이 각 고유벡터

# 검증: A @ v = λ × v
for i in range(len(eigenvalues)):
    v  = eigenvectors[:, i]
    lv = eigenvalues[i]
    print(f'A@v = {A @ v},  λ×v = {lv * v}')  # 같아야 함

# 재조합: A = P @ Λ @ P^{-1}
P      = eigenvectors
Lambda = np.diag(eigenvalues)
P_inv  = linalg.inv(P)
A_reconstructed = P @ Lambda @ P_inv
print(f'재조합 오차: {np.max(np.abs(A - A_reconstructed)):.2e}')

# 대칭 행렬의 고유값 분해 (더 안정적, 고유값이 항상 실수)
# PCA에서 공분산 행렬에 이 방식 사용
A_sym = A @ A.T   # A×A^T = 대칭 행렬
eigenvalues_sym, eigenvectors_sym = linalg.eigh(A_sym)  # eigh: 대칭/에르미트 행렬 전용
```

**활용 분야**:

- **PCA (주성분 분석)**: 공분산 행렬의 고유벡터 = 주성분 방향
- **마르코프 연쇄**: 정상 분포를 고유벡터로 계산
- **연립 미분방정식**: 고유값으로 시스템 안정성 분석
- **그래프 이론**: 인접 행렬의 고유값으로 연결성 분석 (Spectral Clustering)

---

#### 특이값 분해 (SVD, Singular Value Decomposition)

**모든 행렬** (정방행렬이 아니어도)에 적용 가능한 일반화된 분해다.  
고유값 분해는 정방행렬에만 쓸 수 있지만, SVD는 어떤 크기의 행렬에도 적용된다.

$$A_{m \times n} = U_{m \times m} \cdot \Sigma_{m \times n} \cdot V^T_{n \times n}$$

- $U$: 좌 특이벡터 — 출력 공간(행 공간)의 직교 기저
- $\Sigma$: 특이값 대각행렬 — $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ (크기 순 정렬)
- $V^T$: 우 특이벡터의 전치 — 입력 공간(열 공간)의 직교 기저

```
직관적으로:
  행렬 A의 변환 = "입력 방향 회전(V^T)" → "각 방향 스케일링(Σ)" → "출력 방향 회전(U)"
  
  특이값(σ)이 클수록 해당 방향이 데이터에서 중요한 정보를 담고 있다
  특이값이 0에 가까우면 그 방향은 거의 정보가 없다
```

**고유값 분해 vs SVD 비교**:

|구분|고유값 분해|SVD|
|---|---|---|
|적용 행렬|정방행렬만|모든 행렬 ($m \times n$)|
|결과|고유값 $\lambda$, 고유벡터|특이값 $\sigma$, 좌/우 특이벡터|
|고유값/특이값|실수 or 복소수|항상 0 이상의 실수|
|관계|—|$\sigma_i = \sqrt{\lambda_i(A^T A)}$|

```python
# SVD
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)   # 2×3 비정방행렬

U, s, Vt = linalg.svd(A)
print(f'U shape:  {U.shape}')    # (2, 2)
print(f's:        {s}')          # 특이값 [2개]
print(f'Vt shape: {Vt.shape}')   # (3, 3)

# Σ 행렬 복원 (s는 대각 원소 벡터)
Sigma = np.zeros_like(A)
Sigma[:len(s), :len(s)] = np.diag(s)

# 재조합 검증: A = U @ Σ @ Vt
A_reconstructed = U @ Sigma @ Vt
print(f'재조합 오차: {np.max(np.abs(A - A_reconstructed)):.2e}')

# ── 저랭크 근사 (Low-rank Approximation) ──────────
# 상위 k개 특이값만 사용해서 행렬을 압축
k = 1
A_approx = s[0] * np.outer(U[:, 0], Vt[0, :])   # rank-1 근사
print(f'rank-1 근사:\n{A_approx}')

# k개 성분으로 근사
def low_rank_approx(U, s, Vt, k):
    return sum(s[i] * np.outer(U[:, i], Vt[i, :]) for i in range(k))

A_k = low_rank_approx(U, s, Vt, k=1)
```

**이미지 압축 예시 (SVD 저랭크 근사)**:

```python
from skimage import data
import matplotlib.pyplot as plt

img = data.camera().astype(float)   # 512×512

# SVD 분해
U, s, Vt = linalg.svd(img)

# 특이값의 누적 에너지 비율 확인 (몇 개로 얼마나 표현 가능한지)
cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
k_90 = np.searchsorted(cumulative_energy, 0.90) + 1  # 90% 에너지 포함하는 k
k_99 = np.searchsorted(cumulative_energy, 0.99) + 1  # 99% 에너지 포함하는 k
print(f'90% 에너지: k={k_90}개 / 전체 {len(s)}개')
print(f'99% 에너지: k={k_99}개 / 전체 {len(s)}개')

# 여러 k로 압축 비교
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, k in zip(axes, [1, 5, 20, 50, 512]):
    img_k = sum(s[i] * np.outer(U[:, i], Vt[i, :]) for i in range(min(k, len(s))))
    ax.imshow(img_k, cmap='gray')
    ratio = k * (512 + 512 + 1) / (512 * 512) * 100
    ax.set_title(f'k={k}\n({ratio:.1f}% 데이터)')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**활용 분야**:

- **PCA**: SVD로 주성분 분석 수행 (scikit-learn의 `PCA`가 내부적으로 SVD 사용)
- **추천 시스템**: 사용자-아이템 행렬을 SVD로 분해해 잠재 요인 추출 (Matrix Factorization)
- **이미지 압축**: 상위 k개 특이값만 유지
- **의사역행렬(Pseudoinverse)**: 역행렬이 없는 경우에도 최소제곱 해를 구할 때
- **LSA (잠재 의미 분석)**: 문서-단어 행렬을 분해해 문서 간 유사도 계산

```python
# 의사역행렬 (Moore-Penrose Pseudoinverse)
# 역행렬이 존재하지 않는 경우 (비정방 or 특이 행렬)에 사용
A_pinv = linalg.pinv(A)    # 내부적으로 SVD 사용
print(f'의사역행렬 shape: {A_pinv.shape}')

# 최소제곱 해: A @ x ≈ b
b = np.array([1, 2])
x_lstsq = A_pinv @ b   # A^+ @ b = 최소제곱 해
```

---

## 4. scipy.stats — 확률 분포와 통계

### 확률 분포의 4가지 함수

|함수|이름|의미|
|---|---|---|
|**pdf**|Probability Density Function (확률밀도함수)|특정 값에서의 확률 밀도|
|**cdf**|Cumulative Distribution Function (누적분포함수)|$P(X \leq x)$, 누적 확률|
|**ppf**|Percent Point Function (분위수함수, CDF의 역함수)|확률 $p$에 대응하는 $x$ 값|
|**rvs**|Random Variates (난수 생성)|해당 분포에서 무작위 표본 추출|

```
pdf: "x=100에서 밀도는 얼마인가?" → 0.025
cdf: "X≤120일 확률은?" → 0.84 (84%)
ppf: "상위 5%의 기준값은?" → 116.4  (cdf의 역함수)
rvs: "이 분포에서 100개 샘플을 뽑으면?" → [98, 103, 95, ...]
```

```python
from scipy import stats

# 정규분포 N(mean=100, std=10)
dist = stats.norm(loc=100, scale=10)

# pdf: x=100에서의 확률밀도
print(f'pdf(100): {dist.pdf(100):.4f}')    # 최댓값 (평균에서)

# cdf: P(X ≤ 120)
print(f'P(X≤120): {dist.cdf(120):.4f}')   # ≈ 0.9772

# ppf: 상위 5% 기준값 (cdf의 역함수)
print(f'상위 5% 기준: {dist.ppf(0.95):.2f}')  # ≈ 116.45

# rvs: 난수 1000개 생성
samples = dist.rvs(size=1000, random_state=42)

# 신뢰구간 계산
ci = dist.interval(0.95)   # 95% 신뢰구간 (정규분포 가정)
print(f'95% 구간: {ci}')   # (80.4, 119.6)
```

### 주요 확률 분포

```python
# 다양한 분포
stats.norm(loc=0, scale=1)     # 정규분포
stats.t(df=10)                 # t분포 (자유도 10)
stats.chi2(df=5)               # 카이제곱 분포
stats.f(dfn=2, dfd=10)         # F분포
stats.binom(n=10, p=0.5)       # 이항분포
stats.poisson(mu=3)            # 포아송 분포
stats.expon(scale=2)           # 지수분포
stats.uniform(loc=0, scale=1)  # 균일분포
```

### 통계 검정

```python
# t-검정
t_stat, p_val = stats.ttest_ind(group_a, group_b)   # 독립 표본
t_stat, p_val = stats.ttest_rel(before, after)       # 대응 표본
t_stat, p_val = stats.ttest_1samp(sample, popmean=0) # 단일 표본

# 정규성 검정 (Shapiro-Wilk)
stat, p = stats.shapiro(data)
print(f'정규성 p-value: {p:.4f}')  # p > 0.05 → 정규분포

# 카이제곱 적합도 검정
chi2, p = stats.chisquare(observed, expected)

# 상관계수
r, p = stats.pearsonr(x, y)
r, p = stats.spearmanr(x, y)
```

---

## 5. scipy.optimize — 최적화

최적화는 **함수의 최솟값(또는 최댓값)이 되는 입력값을 찾는 과정**이다.  
머신러닝의 학습 과정이 사실 이 최적화 문제를 반복해서 푸는 것이다.

```
목표: f(x)를 최소화하는 x를 찾아라
방법: 기울기를 따라 내려가거나 (경사하강법)
      여러 점에서 함수값을 비교하거나
      2차 미분 정보를 활용 (뉴턴법)
```

### 함수 최솟값 찾기

```python
from scipy.optimize import minimize, minimize_scalar

# 단변수 최적화 (scalar)
f = lambda x: (x - 3)**2 + 2   # 최솟값: x=3, f=2
result = minimize_scalar(f, bounds=(0, 5), method='bounded')
print(f'최적 x: {result.x:.4f}')    # ≈ 3.0
print(f'최솟값: {result.fun:.4f}')  # ≈ 2.0

# 다변수 최적화
def rosenbrock(x):
    """로젠브록 함수: 최솟값 (1,1)"""
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

result = minimize(rosenbrock,
                  x0=[0, 0],           # 초기값
                  method='BFGS',        # 최적화 방법
                  jac='2-point')        # 수치 기울기 사용
print(f'최적점: {result.x}')   # ≈ [1, 1]
```

### 방정식의 근 찾기 (Root Finding)

$f(x) = 0$이 되는 $x$를 찾는다. 연립방정식 풀이와 연결된다.

```python
from scipy.optimize import fsolve, brentq

# 단변수 방정식의 근: x^3 - x - 2 = 0
f = lambda x: x**3 - x - 2
root = brentq(f, 1, 2)   # 구간 [1,2]에서 근 탐색 (이분법)
print(f'근: {root:.6f}')  # ≈ 1.5214

# 연립방정식의 근
def equations(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 4    # x² + y² = 4
    eq2 = x - y - 1           # x - y = 1
    return [eq1, eq2]

solution = fsolve(equations, x0=[1, 0])
print(f'해: x={solution[0]:.4f}, y={solution[1]:.4f}')
```

### 뉴턴법 (Newton's Method)

**2차 미분(곡률) 정보까지 활용**해서 최솟값을 빠르게 찾는 방법이다.  
현재 위치에서 함수를 2차 함수로 근사한 뒤, 그 2차 함수의 최솟값으로 한 번에 이동한다.

$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

```
직관:
  f''(x) > 0 (위로 볼록) → 최솟값이 어느 방향인지, 얼마나 이동해야 하는지 정확히 앎
  → 이동 크기를 자동으로 조절 → 큰 보폭으로 빠르게 수렴

경사하강법과의 차이:
  경사하강법: 기울기(1차 미분)만 보고 내려감 → "방향만 앎"
  뉴턴법:    기울기 + 곡률(2차 미분) 보고 내려감 → "방향과 보폭 모두 앎"
```

**장점**: 수렴이 매우 빠르다 (이차 수렴 — 오차가 제곱으로 줄어듦)  
**단점**:

- 2차 미분(헤시안 행렬)을 계산해야 해서 **고차원에서 계산 비용이 엄청남** ($n^2$ 크기 행렬)
- 초기값이 나쁘면 발산할 수 있다
- 안장점(saddle point)에서 잘못된 방향으로 갈 수 있다

```python
from scipy.optimize import fsolve
import numpy as np

# 뉴턴법으로 방정식의 근 찾기: f(x) = x^2 - 2 = 0 (√2 구하기)
def f(x):  return x**2 - 2
def df(x): return 2*x

# 수동 구현
x = 1.0   # 초기값
for i in range(10):
    x = x - f(x) / df(x)
    print(f'iter {i+1}: x = {x:.10f}')   # 매우 빠르게 √2에 수렴

# scipy로 다차원 뉴턴법
result = fsolve(f, x0=1.0, full_output=True)
print(f'근: {result[0][0]:.8f}')  # ≈ 1.41421356
```

---

### 경사하강법 (Gradient Descent)

**1차 미분(기울기)만 사용**해서 함수의 최솟값을 향해 조금씩 내려가는 방법이다.  
머신러닝 모델 학습의 핵심 알고리즘이다.

$$x_{n+1} = x_n - \alpha \cdot \nabla f(x_n)$$

- $\alpha$: 학습률(learning rate) — 한 번에 이동하는 보폭 크기
- $\nabla f(x_n)$: 기울기 (gradient) — 오르막 방향

```
직관:
  산 위에서 눈을 감고 가장 가파른 내리막 방향으로 한 걸음씩 내려간다

  α 가 너무 크면: 보폭이 너무 커서 계곡을 뛰어넘고 발산
  α 가 너무 작으면: 너무 조금씩 이동해서 수렴이 너무 느림
  α 가 적당하면: 안정적으로 최솟값으로 수렴
```

```
손실(Loss)
   │\
   │ \
   │  \
   │   \  ← 기울기가 음수 → 오른쪽으로 이동
   │    \_____
   └──────────── x
        ↑
     최솟값
```

```python
import numpy as np

# 경사하강법 직접 구현: f(x) = (x-3)^2 의 최솟값 찾기
def f(x):  return (x - 3)**2
def df(x): return 2*(x - 3)   # 1차 미분 (기울기)

x     = 0.0   # 초기값
alpha = 0.1   # 학습률

for i in range(20):
    grad = df(x)
    x    = x - alpha * grad
    print(f'iter {i+1:2d}: x={x:.4f}, f(x)={f(x):.6f}')

# scipy로 경사 기반 최적화
from scipy.optimize import minimize

result = minimize(f, x0=0.0, method='BFGS',
                  jac=df)   # jac: 기울기 함수 직접 제공
print(f'최적 x: {result.x[0]:.6f}')  # ≈ 3.0
```

### 경사하강법의 변형

|방법|업데이트 기준|특징|
|---|---|---|
|**Batch GD**|전체 데이터|안정적이지만 느림|
|**Stochastic GD (SGD)**|샘플 1개|빠르지만 노이즈가 많음|
|**Mini-batch GD**|일부 데이터 (batch size)|Batch + SGD의 절충 (실무 표준)|
|**Momentum**|이전 이동 방향을 누적|관성으로 수렴 안정화|
|**Adam**|Momentum + 적응형 학습률|딥러닝 기본 옵티마이저|

---

### 뉴턴법 vs 경사하강법 비교

|구분|뉴턴법|경사하강법|
|---|---|---|
|사용 정보|1차 + 2차 미분|1차 미분만|
|수렴 속도|매우 빠름 (이차 수렴)|느림 (선형 수렴)|
|계산 비용|높음 ($n^2$ 헤시안 행렬)|낮음|
|학습률|불필요 (자동 조절)|수동 설정 필요|
|고차원|비실용적|실용적|
|머신러닝 적용|소규모 문제, 통계|대규모 딥러닝|
|SciPy 구현|`Newton-CG`, `trust-ncg`|`CG`, `BFGS` (근사)|

```
→ 실무에서는 대부분 경사하강법 기반을 사용한다.
  뉴턴법의 2차 미분 비용을 줄인 준뉴턴법(BFGS, L-BFGS-B)이 절충안으로 자주 쓰인다.

  BFGS: 헤시안 행렬을 직접 계산하지 않고 기울기 정보로 근사 → 빠름 + 저렴
  L-BFGS-B: BFGS의 메모리 효율 버전 → 수백만 파라미터도 가능
```

```python
# 준뉴턴법 (BFGS) vs 경사하강법 (CG) 비교
from scipy.optimize import minimize

# Rosenbrock 함수 (골짜기 형태의 어려운 최적화 문제)
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

x0 = [0, 0]

for method in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B']:
    result = minimize(rosenbrock, x0, method=method,
                      jac='2-point')
    print(f'{method:12s}: 반복={result.nit:4d}, 최적점={result.x}')
```

---

### 최적화 방법 비교

|방법|특징|적합한 경우|
|---|---|---|
|**Nelder-Mead**|기울기 불필요, 강건함|미분 불가능한 함수|
|**CG**|켤레 기울기법, 1차 미분|대규모 문제|
|**BFGS**|준뉴턴법 (헤시안 근사)|부드러운 함수, 빠른 수렴|
|**L-BFGS-B**|BFGS의 메모리 효율 버전|고차원, 머신러닝|
|**Newton-CG**|뉴턴법 + 켤레 기울기|중간 규모, 정밀한 수렴|
|**TNC**|제약 조건 있는 최적화|변수 범위 제한 있을 때|

---

## 6. scipy.fft — 고속 푸리에 변환 (FFT)

### 푸리에 변환이란

**시간 영역의 신호를 주파수 영역으로 변환**하는 방법이다.  
"이 신호 안에 어떤 주파수 성분이 얼마나 들어있는가"를 분석한다.

```
시간 영역: 시간에 따른 진폭 변화    → "언제 얼마나 크게 진동했나"
주파수 영역: 주파수별 에너지 분포   → "어떤 주파수가 얼마나 강한가"

예: 음악 파일
   시간 영역: 파형 (음압의 시간 변화)
   주파수 영역: 스펙트럼 (고음, 중음, 저음 성분 비율)
```

수학적으로 오일러 공식 $e^{i\theta} = \cos\theta + i\sin\theta$가 기반이다.  
신호를 다양한 주파수의 사인파와 코사인파의 합으로 분해한다.

### 활용 분야

- **신호 처리**: 노이즈 제거, 필터링
- **영상 처리**: 이미지 압축(JPEG), 엣지 검출
- **시계열 분석**: 계절성·주기 탐지
- **음성 인식**: 음성 특성 추출 (MFCC)
- **통신**: 변조/복조 처리

```python
from scipy.fft import fft, ifft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

# 신호 생성: 50Hz와 120Hz 사인파의 합 + 노이즈
fs = 1000                           # 샘플링 주파수 (1000 Hz)
t  = np.linspace(0, 1, fs)          # 1초, 1000개 포인트
signal = (np.sin(2 * np.pi * 50 * t)   # 50Hz 성분
        + np.sin(2 * np.pi * 120 * t)  # 120Hz 성분
        + 0.5 * np.random.randn(fs))   # 가우시안 노이즈

# FFT 수행
fft_result  = fft(signal)           # 복소수 배열
frequencies = fftfreq(fs, d=1/fs)   # 각 bin의 주파수

# 진폭 스펙트럼 (절댓값의 절반 — 양의 주파수만)
amplitude = np.abs(fft_result) / fs
n = fs // 2

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(t[:200], signal[:200])
axes[0].set_title('시간 영역 신호')
axes[0].set_xlabel('시간 (초)')

axes[1].plot(frequencies[:n], amplitude[:n])
axes[1].set_title('주파수 영역 (스펙트럼)')
axes[1].set_xlabel('주파수 (Hz)')
axes[1].axvline(x=50,  color='red',   linestyle='--', label='50Hz')
axes[1].axvline(x=120, color='green', linestyle='--', label='120Hz')
axes[1].legend()

plt.tight_layout()
plt.show()

# 노이즈 제거: 특정 주파수만 남기고 역변환
fft_filtered = fft_result.copy()
fft_filtered[np.abs(frequencies) > 150] = 0   # 150Hz 이상 제거
signal_clean = np.real(ifft(fft_filtered))      # 역FFT
```

---

## 7. scipy.interpolate — 보간법

알려진 데이터 포인트 사이의 **빈 값을 추정**하는 방법이다.  
(스플라인 보간법은 Matplotlib 시각화 정리 문서에서 자세히 다뤘다.)

```python
from scipy.interpolate import interp1d, CubicSpline, RBFInterpolator
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 2, 1, 3, 2, 4])
x_new = np.linspace(0, 5, 200)

# 선형 보간
f_linear = interp1d(x, y, kind='linear')

# 3차 스플라인 (자연스러운 곡선)
f_cubic  = CubicSpline(x, y)

# 다양한 보간 방법 비교
for kind in ['linear', 'quadratic', 'cubic']:
    f = interp1d(x, y, kind=kind)

# 2D 불규칙 격자 보간 (RBF: 방사 기저 함수)
# 예: 지형 고도 데이터의 임의 위치 추정
points = np.random.rand(50, 2)   # 50개 무작위 지점
values = np.sin(points[:, 0]) * np.cos(points[:, 1])
rbf = RBFInterpolator(points, values)

# 새 격자에서 값 예측
grid_x, grid_y = np.meshgrid(np.linspace(0,1,20), np.linspace(0,1,20))
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
grid_values = rbf(grid_points).reshape(20, 20)
```

---

## 8. scipy.ndimage — 다차원 이미지 처리

NumPy 배열로 표현된 이미지(2D/3D 배열)에 필터링, 변환, 측정 등의 연산을 적용한다.

```python
from scipy import ndimage
import numpy as np

# 이미지를 NumPy 배열로 불러오기
from skimage import data
img = data.camera()   # 흑백 이미지 (512×512 uint8 배열)

# 가우시안 블러 — 노이즈 제거 (sigma: 흐림 정도)
img_blur   = ndimage.gaussian_filter(img, sigma=2)

# 엣지 검출 — 경계선 추출
img_edge   = ndimage.sobel(img)

# 회전
img_rotate = ndimage.rotate(img, angle=45, reshape=False)

# 레이블링 — 연결된 영역 찾기 (객체 검출의 기초)
binary = img > 128                        # 이진화
labeled, n_features = ndimage.label(binary)
print(f'검출된 영역 수: {n_features}')

# 각 영역의 무게중심
centers = ndimage.center_of_mass(binary, labeled,
                                  range(1, n_features+1))
```

---

### 컨볼루션 (Convolution)

컨볼루션은 이미지 처리와 딥러닝(CNN)의 핵심 연산이다.  
**커널(kernel, 필터)이라는 작은 행렬을 이미지 위로 슬라이딩하면서 가중합을 계산**한다.

#### 컨볼루션 연산 원리

```
입력 이미지 (5×5):           커널 (3×3):
┌─────────────────┐          ┌───────────┐
│  1  2  3  4  5  │          │ 0  -1   0 │
│  6  7  8  9 10  │    *     │-1   4  -1 │  →  출력 이미지 (3×3)
│ 11 12 13 14 15  │          │ 0  -1   0 │
│ 16 17 18 19 20  │          └───────────┘
│ 21 22 23 24 25  │
└─────────────────┘

출력 한 칸 계산 (왼쪽 위):
  0×1 + (-1)×2 + 0×3
+ (-1)×6 + 4×7 + (-1)×8
+ 0×11 + (-1)×12 + 0×13
= -2 - 6 + 28 - 8 - 12 = 0
```

커널이 이미지 위를 한 칸씩 이동(stride)하면서 겹치는 영역과 원소별 곱을 구한 뒤 합산한다.  
커널의 값이 곧 **"어떤 특징을 감지할 것인가"** 를 결정한다.

#### 수식

$$(f * g)(x, y) = \sum_m \sum_n f(m, n) \cdot g(x-m, y-n)$$

- $f$: 입력 이미지
- $g$: 커널 (필터)
- 출력 크기: $(H - k + 1) \times (W - k + 1)$ (패딩 없을 때, $k$: 커널 크기)

#### 목적별 커널 종류

|커널 이름|행렬|효과|
|---|---|---|
|**블러 (평균)**|$\frac{1}{9}\begin{bmatrix}1&1&1\1&1&1\1&1&1\end{bmatrix}$|노이즈 제거, 부드럽게|
|**가우시안 블러**|가우시안 분포 값|자연스러운 블러|
|**샤프닝**|$\begin{bmatrix}0&-1&0\-1&5&-1\0&-1&0\end{bmatrix}$|경계 강조, 선명하게|
|**엣지 검출 (Sobel X)**|$\begin{bmatrix}-1&0&1\-2&0&2\-1&0&1\end{bmatrix}$|수평 경계선 감지|
|**엣지 검출 (Sobel Y)**|$\begin{bmatrix}-1&-2&-1\0&0&0\1&2&1\end{bmatrix}$|수직 경계선 감지|
|**라플라시안**|$\begin{bmatrix}0&1&0\1&-4&1\0&1&0\end{bmatrix}$|모든 방향 엣지 감지|

```python
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

img = data.camera().astype(float)

# ── 커널 직접 정의 ──────────────────────────────────
blur_kernel    = np.ones((5, 5)) / 25            # 평균 블러
sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])          # 샤프닝
sobel_x        = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])           # 수평 엣지
sobel_y        = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]])           # 수직 엣지
laplacian      = np.array([[0,  1, 0],
                            [1, -4, 1],
                            [0,  1, 0]])            # 라플라시안

# ── 컨볼루션 적용 ──────────────────────────────────
img_blur    = ndimage.convolve(img, blur_kernel)
img_sharp   = ndimage.convolve(img, sharpen_kernel)
img_edge_x  = ndimage.convolve(img, sobel_x)
img_edge_y  = ndimage.convolve(img, sobel_y)
img_edge    = np.hypot(img_edge_x, img_edge_y)   # Sobel X + Y 합성
img_lap     = ndimage.convolve(img, laplacian)

# ── 시각화 ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, image, title in zip(
    axes.ravel(),
    [img, img_blur, img_sharp, img_edge_x, img_edge, img_lap],
    ['원본', '블러', '샤프닝', 'Sobel X', 'Sobel 합성', '라플라시안']
):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

# ── 가우시안 블러 (별도 함수로 더 정교하게) ──────────
img_gaussian = ndimage.gaussian_filter(img, sigma=3)
```

#### 컨볼루션과 상관관계(Correlation)의 차이

```python
# convolve: 커널을 180도 회전 후 슬라이딩 (수학적 컨볼루션)
# correlate: 커널을 그대로 슬라이딩 (딥러닝에서 실제로 사용하는 방식)
# 대칭 커널(블러, 라플라시안 등)에서는 결과 동일

img_conv  = ndimage.convolve(img, sobel_x)
img_corr  = ndimage.correlate(img, sobel_x)
# sobel_x는 비대칭이므로 두 결과가 다름
```

딥러닝 CNN에서 말하는 "컨볼루션"은 엄밀히는 **상관관계(cross-correlation)** 를 사용한다.  
커널을 회전하지 않고 그대로 슬라이딩하지만, 학습으로 커널 값이 결정되므로 수학적 차이는 실질적으로 의미 없다.

#### CNN에서의 컨볼루션

```
입력 이미지
   ↓ 컨볼루션 레이어 (여러 커널)
특징 맵 (Feature Map)
   ↓ 활성화 함수 (ReLU)
   ↓ 풀링 (MaxPooling) — 크기 축소
   ↓ 다음 컨볼루션 레이어
   ...
   ↓ Flatten → Fully Connected → 출력
```

초기 레이어는 엣지, 선, 색상 같은 저수준 특징을 감지하고,  
깊어질수록 눈, 바퀴, 얼굴처럼 고수준 특징을 감지한다.

---

### 이미지를 배열로 이해하기

```
흑백 이미지 = 2D NumPy 배열 (height × width)
컬러 이미지 = 3D NumPy 배열 (height × width × 3채널(RGB))
동영상      = 4D NumPy 배열 (frame × height × width × 채널)

픽셀값 0~255 (uint8)
  0 = 검정, 255 = 흰색 (흑백)
  (255,0,0) = 빨강 (RGB)
```

---

## 9. SciPy 모듈 전체 구조 정리

```
scipy
   ├─ integrate     수치 적분 (quad, dblquad, trapezoid)
   ├─ linalg        선형대수 (inv, solve, lu, qr, svd, eig)
   ├─ optimize      최적화, 방정식 근 (minimize, fsolve, brentq)
   ├─ stats         확률 분포, 통계 검정 (norm, t, chi2, ttest)
   ├─ fft           고속 푸리에 변환 (fft, ifft, fftfreq)
   ├─ interpolate   보간법 (interp1d, CubicSpline, RBFInterpolator)
   ├─ ndimage       이미지 처리 (gaussian_filter, sobel, label)
   ├─ signal        신호 처리 (filtfilt, spectrogram, find_peaks)
   ├─ sparse        희소행렬 (csr_matrix, csc_matrix)
   └─ special       특수 함수 (gamma, bessel, erf, factorial)
```

---

## 10. NumPy vs SciPy 역할 분담

|구분|NumPy|SciPy|
|---|---|---|
|핵심 역할|자료구조(ndarray), 기본 연산|고급 과학 계산 알고리즘|
|선형대수|기본 (inv, eig, svd)|확장 (lu, qr, cholesky, 수치 안정성 ↑)|
|통계|기초 (mean, std, percentile)|완전한 확률 분포, 가설 검정|
|최적화|없음|minimize, root finding|
|적분/미분|없음|quad, derivative|
|신호 처리|fft 기본|완전한 DSP 도구|

```
NumPy: "데이터를 배열로 표현하고 기본 연산"
SciPy: "그 위에서 수학/과학 문제를 풀기 위한 알고리즘"
```