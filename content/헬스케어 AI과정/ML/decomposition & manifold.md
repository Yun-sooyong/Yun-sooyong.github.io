---
title: 042 decomposition & manifold
tag:
  - 헬스케어 ai
  - ML
description: 260611 수업 내용 정리
---

# 차원 축소 (Dimensionality Reduction)

차원 축소는 **많은 변수(차원)를 가진 데이터를 더 적은 차원으로 압축**하면서 핵심 정보를 최대한 보존하는 기법이다.

단순히 데이터를 줄이는 것이 목적이 아니다. 고차원에서는 보이지 않던 **구조와 패턴**을 저차원에서 드러낼 수 있고, 모델이 불필요한 노이즈에 학습되는 것을 막는다.

```
척도 기반 분류:
  분산 기반: PCA, SparsePCA, KernelPCA  → 데이터가 가장 많이 퍼진 방향으로 축소
  거리 기반: MDS, UMAP                  → 점 간의 거리 관계를 보존하며 축소
  방향 기반: ICA                        → 통계적으로 독립인 신호를 분리
```

---

## 1. 왜 차원 축소가 필요한가

### 차원의 저주 (Curse of Dimensionality)

차원(변수)이 늘어날수록 데이터 공간이 **기하급수적으로 팽창**해서 다음 문제들이 생긴다.

```
같은 밀도를 유지하려면 필요한 데이터 수:
  1차원: 10개
  2차원: 100개  (10²)
  3차원: 1000개  (10³)
  10차원: 10,000,000,000개  (10¹⁰)
  
  → 현실적으로 고차원 공간을 채울 수 없음
  → 모든 점이 서로 "비슷하게 멀어짐" → 거리의 의미 퇴색
  → KNN, K-Means 같은 거리 기반 알고리즘 성능 저하
  → 과적합 위험 증가
```

구체적으로 단위 정육면체 안에 무작위로 점을 채울 때, 중심으로부터 일정 거리 안에 있는 점의 비율이 차원이 늘수록 급격히 줄어든다. 고차원에서는 데이터가 공간의 가장자리에 몰리는 현상이 발생한다.

차원이 높아지면 변수가 많아지고, 그 많은 변수 공간을 채울 데이터가 부족해서 **빈 공간이 많아지는 희소성 문제(Sparsity Problem)** 가 발생한다. 동시에 변수가 늘수록 모델의 복잡도가 증가해서 편향-분산 트레이드오프의 균형점을 찾기가 어려워진다. 변수를 줄이는 것이 분산을 낮추는 데 도움이 되지만, 너무 많이 줄이면 정보 손실로 편향이 높아진다.

```
차원 축소의 딜레마:
  변수 너무 많음 → 희소성 ↑, 복잡도 ↑, 분산 ↑ (과적합)
  변수 너무 적음 → 정보 손실 ↑, 편향 ↑ (과소적합)
  
  → Bias-Variance 균형점을 찾는 것이 차원 축소의 핵심
```

그리고 차원 축소 자체도 원래 데이터의 정보가 소실될 염려가 있다. PCA라면 "분산 보존"을, Manifold 방법이라면 "거리 보존"을 기준으로 삼아 최대한 중요한 정보를 유지하는 방향을 선택해야 한다.

### 차원 축소의 두 가지 방향

첫 번째는 **특성 추출(Feature Extraction)** 이다. 원래 변수들을 조합해서 완전히 새로운 변수(성분)를 만든다. PCA, ICA, NMF가 이 방향이다. 새 변수는 원래 변수의 이름이나 단위를 잃지만 더 적은 수로 더 많은 정보를 담는다.

두 번째는 **특성 선택(Feature Selection)** 이다. 원래 변수 중 중요한 것만 골라낸다. 분산 임계값, RFE, Lasso 계수 등이 이 방향이다. 원래 변수의 의미가 보존된다.

---

## 2. 고유값 분해 (Eigenvalue Decomposition)

### 고유값과 고유벡터의 의미

행렬 $A$를 벡터 $\mathbf{v}$에 곱했을 때 **방향은 바뀌지 않고 크기만 $\lambda$배 변하는 벡터**가 고유벡터이고, 그 배수 $\lambda$가 고유값이다.

$$A\mathbf{v} = \lambda\mathbf{v}$$

```
일반 벡터에 A를 곱하면:
  방향도 바뀌고 크기도 바뀜 → 변환 후 어디로 갔는지 추적하기 어려움

고유벡터에 A를 곱하면:
  방향은 그대로, 크기만 λ배 변함 → 변환의 "본질적인 방향"
  
  λ가 크다 = 행렬 A가 이 방향으로 데이터를 강하게 늘림 (중요한 방향)
  λ가 작다 = 행렬 A가 이 방향으로 데이터를 거의 안 변형 (중요하지 않은 방향)
```

### 정방행렬의 고유값 분해

$$A = V \Lambda V^{-1}$$

- $V$: 고유벡터를 열로 가진 행렬 (새로운 축의 방향)
- $\Lambda$: 고유값을 대각 성분으로 가진 대각행렬 (각 축의 중요도)

**PCA에서의 활용**: 공분산 행렬은 항상 정방행렬이고 대칭행렬이므로 고유값이 항상 실수이고 고유벡터가 서로 직교한다. 이 성질이 PCA가 직교 축을 만드는 이유다.

그러나 이것이 PCA의 근본적인 한계이기도 하다. 공분산 행렬을 거쳐야만 고유값 분해를 쓸 수 있기 때문에 PCA는 내부적으로 정방행렬 연산에 의존한다. 반면 실제 데이터 행렬은 행(샘플) ≠ 열(특성)인 **비정방 행렬**이 대부분이다. 이 한계를 직접 해결한 것이 다음 섹션의 SVD다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D 예시: 분산이 큰 방향과 작은 방향 찾기
np.random.seed(42)
X = np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], 200)

# 공분산 행렬
C = np.cov(X.T)
print(f'공분산 행렬:\n{C}')

# 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(C)
# 고유값 큰 순서로 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f'고유값: {eigenvalues}')       # [λ₁, λ₂] → λ₁ > λ₂
print(f'고유벡터:\n{eigenvectors}')   # 각 열이 주성분 방향

# 시각화: 데이터와 주성분 방향
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=20, label='데이터')

origin = np.mean(X, axis=0)
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    scale = np.sqrt(val) * 2   # 크기를 표준편차에 비례하게
    ax.annotate('', xy=origin + scale*vec,
                 xytext=origin - scale*vec,
                 arrowprops=dict(arrowstyle='<->', color=f'C{i+1}', lw=2))
    ax.text(origin[0] + scale*vec[0]*1.1, origin[1] + scale*vec[1]*1.1,
            f'PC{i+1}\n(λ={val:.2f})', fontsize=10, color=f'C{i+1}')

ax.set_aspect('equal')
ax.set_title('주성분 방향 (고유벡터) 시각화')
ax.legend()
plt.show()
# PC1 방향이 분산이 가장 큰 방향 (λ₁이 더 큼)
```

---

## 3. 특이값 분해 (SVD, Singular Value Decomposition)

고유값 분해의 가장 큰 한계는 **정방행렬에만 적용**된다는 것이다. 실제 데이터 행렬은 대부분 행(샘플) ≠ 열(특성)인 직사각형 행렬이다.

SVD는 이 한계를 극복한 **모든 행렬에 적용 가능한 일반화된 분해**다. 중요한 점은 PCA가 SVD의 특수한 경우이지, 별개의 방법이 아니라는 것이다. **특이값 분해 안에 PCA가 포함**된다.

### 수식

$$\underbrace{A}_{m \times n} = \underbrace{U}_{m \times m} \underbrace{\Sigma}_{m \times n} \underbrace{V^T}_{n \times n}$$

|기호|이름|크기|의미|
|---|---|---|---|
|$U$|왼쪽 특이벡터|$m \times m$|행(샘플) 공간의 직교 기저. "출력 방향"|
|$\Sigma$|특이값 행렬|$m \times n$|대각에 특이값 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$. "각 방향의 중요도"|
|$V^T$|오른쪽 특이벡터 전치|$n \times n$|열(특성) 공간의 직교 기저. "입력 방향"|

### 직관적 이해 — 변환의 분해

행렬 $A$가 표현하는 선형 변환을 세 단계로 분해한다.

```
A의 변환 = 입력 공간 회전(V^T)
          × 각 방향 스케일링(Σ)
          × 출력 공간 회전(U)

"어떤 방향으로 돌리고 (V^T)
 얼마나 늘리고 (Σ)
 다시 어떤 방향으로 돌려서 (U) 출력을 만드는가"
```

### 특이값의 의미

특이값 $\sigma_i$가 클수록 그 방향이 데이터에서 중요한 정보를 담고 있다.  
$\sigma_i = 0$에 가까우면 그 방향은 정보가 거의 없는 노이즈에 가깝다.

$$A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \cdots + \sigma_r \mathbf{u}_r \mathbf{v}_r^T$$

행렬 $A$는 **랭크-1 행렬들의 가중합**으로 분해된다. $\sigma_i$가 각 랭크-1 성분의 중요도다.

### Thin SVD (경제적 SVD)

$A$가 $m \times n$ 행렬이고 $m > n$이면 $\Sigma$의 하단 $m-n$행은 모두 0이다. 이를 제거한 것이 Thin SVD다.

$$A_{m \times n} = U_{m \times r} \cdot \Sigma_{r \times r} \cdot V^T_{r \times n}, \quad r = \text{rank}(A)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# ── SVD 전체 예시 ──────────────────────────────────────
np.random.seed(42)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [2, 4, 6]], dtype=float)   # 4×3 직사각형 행렬

# Full SVD
U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=True)
print(f'Full SVD:')
print(f'  U:  {U_full.shape}')   # (4, 4)
print(f'  s:  {s_full.shape}')   # (3,) — 특이값 3개
print(f'  Vt: {Vt_full.shape}')  # (3, 3)

# Thin SVD (경제적)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f'Thin SVD:')
print(f'  U:  {U.shape}')   # (4, 3)
print(f'  s:  {s}')          # 특이값 (크기 순)
print(f'  Vt: {Vt.shape}')  # (3, 3)

# 재조합 검증: A = U @ diag(s) @ Vt
A_reconstructed = U @ np.diag(s) @ Vt
print(f'재조합 오차: {np.max(np.abs(A - A_reconstructed)):.2e}')

# ── 특이값의 중요도 ────────────────────────────────────
energy = s**2 / np.sum(s**2) * 100
cumulative = np.cumsum(energy)
print(f'\n특이값 에너지 비율:')
for i, (sv, e, ce) in enumerate(zip(s, energy, cumulative)):
    print(f'  σ{i+1} = {sv:.3f}: {e:.1f}% (누적 {ce:.1f}%)')
```

### 저랭크 근사 (Low-Rank Approximation) — 핵심 응용

상위 $k$개 특이값만 사용해서 행렬을 **근사**하는 것이 SVD의 핵심 응용이다.

$$A \approx A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$

에크하르트-영 정리에 의해 이것이 **최적 저랭크 근사** (Frobenius 노름 기준) 임이 보장된다.

```python
# 이미지 압축으로 저랭크 근사 시각화
from skimage import data

img = data.camera().astype(float)   # 512×512 흑백 이미지
U, s, Vt = np.linalg.svd(img, full_matrices=False)

# 에너지 누적 확인
total_energy = np.sum(s**2)
cumulative_energy = np.cumsum(s**2) / total_energy

k_90 = np.searchsorted(cumulative_energy, 0.90) + 1
k_99 = np.searchsorted(cumulative_energy, 0.99) + 1
print(f'90% 에너지 포함: {k_90}개 / 전체 {len(s)}개 특이값')
print(f'99% 에너지 포함: {k_99}개 / 전체 {len(s)}개 특이값')

# 여러 k로 압축 비교
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
k_values = [1, 5, 20, 50, 512]

for ax, k in zip(axes, k_values):
    # k개 성분으로 근사
    img_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    img_k = np.clip(img_k, 0, 255)

    compression_ratio = k * (1 + 512 + 512) / (512 * 512) * 100
    energy_pct = cumulative_energy[k-1] * 100

    ax.imshow(img_k, cmap='gray')
    ax.set_title(f'k={k}\n에너지 {energy_pct:.1f}%\n용량 {compression_ratio:.1f}%')
    ax.axis('off')

plt.suptitle('SVD 저랭크 근사를 이용한 이미지 압축', fontsize=12)
plt.tight_layout()
plt.show()

# ── 추천 시스템에서의 활용 (행렬 완성) ────────────────
# 사용자-아이템 평점 행렬 (일부 결측)
# SVD로 잠재 요인 추출 → 결측값 예측
ratings = np.array([[5, 3, 0, 1],
                     [4, 0, 4, 1],
                     [1, 1, 0, 5],
                     [1, 0, 5, 4],
                     [0, 1, 5, 4]], dtype=float)

U_r, s_r, Vt_r = np.linalg.svd(ratings, full_matrices=False)
ratings_approx = U_r[:, :2] @ np.diag(s_r[:2]) @ Vt_r[:2, :]
print(f'\n원본 행렬:\n{ratings}')
print(f'2-랭크 근사 (0인 부분이 예측값):\n{ratings_approx.round(2)}')
```

### SVD와 PCA의 관계

PCA를 고유값 분해로 구현하는 것보다 **SVD로 구현하는 것이 더 수치 안정적**이다.

```
PCA (고유값 분해 방식):
  1. 공분산 행렬 C = XᵀX / n 계산
  2. C의 고유값 분해 → 고유벡터 = 주성분
  
  단점: X를 제곱하면서 수치 오차가 증폭됨

PCA (SVD 방식, sklearn의 실제 구현):
  1. X를 직접 SVD 분해: X = U Σ Vᵀ
  2. 오른쪽 특이벡터 V = 주성분 방향
  3. 투영: Z = XV = UΣ (주성분 점수)
  
  장점: 더 수치 안정적, sklearn PCA가 이 방식 사용
  
관계: X의 특이값 σᵢ와 공분산 행렬의 고유값 λᵢ
  λᵢ = σᵢ² / n
```

```python
from sklearn.decomposition import PCA
import numpy as np

X = np.random.randn(100, 5)
X -= X.mean(axis=0)   # 중심화

# 방법 1: 고유값 분해
C = X.T @ X / len(X)
eigenvalues, eigenvectors = np.linalg.eigh(C)  # eigh: 대칭행렬 전용
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
eigenvalues  = eigenvalues[idx]

# 방법 2: SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)
V_svd = Vt.T   # 오른쪽 특이벡터

# 방법 3: sklearn PCA (내부적으로 SVD 사용)
pca = PCA(n_components=5)
pca.fit(X)

# 세 방법 비교 (부호 차이는 무시)
print('주성분 방향 비교 (절댓값 기준):')
for i in range(3):
    diff = np.max(np.abs(np.abs(eigenvectors[:, i]) - np.abs(V_svd[:, i])))
    print(f'  PC{i+1} 고유값분해 vs SVD 차이: {diff:.2e}')

# 특이값과 고유값 관계: λᵢ = σᵢ² / n
print(f'\n특이값²/n:     {(s**2 / len(X))[:3]}')
print(f'고유값:        {eigenvalues[:3]}')
print(f'설명분산비율:  {pca.explained_variance_ratio_[:3]}')
```

### SVD 활용 분야 정리

|분야|활용 방식|핵심 아이디어|
|---|---|---|
|**PCA**|X에 SVD 적용|상위 k 특이벡터가 주성분|
|**이미지 압축**|픽셀 행렬에 SVD|상위 k개 성분만 보존|
|**추천 시스템 (MF)**|평점 행렬에 SVD|잠재 요인(사용자 취향, 아이템 특성) 추출|
|**LSA**|TF-IDF 행렬에 TruncatedSVD|단어-문서 잠재 의미 추출|
|**노이즈 제거**|신호 행렬에 SVD|작은 특이값(노이즈) 제거|
|**의사역행렬**|$A^+ = V\Sigma^{-1}U^T$|역행렬 없는 경우 최소제곱 해|
|**행렬 랭크 추정**|0에 가까운 특이값 제거|수치적 랭크 결정|

```python
# 의사역행렬 (Pseudoinverse) — 역행렬이 없는 경우
from scipy.linalg import pinv

A = np.array([[1, 2, 3],
              [4, 5, 6]])   # 2×3: 역행렬 없음

# Moore-Penrose 의사역행렬
A_pinv = pinv(A)            # 내부적으로 SVD 사용
print(f'의사역행렬:\n{A_pinv.round(4)}')   # 3×2

# 최소제곱 해: Ax = b
b = np.array([1, 2])
x = A_pinv @ b
print(f'최소제곱 해: {x}')
print(f'검증 A@x: {A @ x}')   # b에 가장 가까운 해
```

---

## 4. PCA (Principal Component Analysis) — 공간 유지

PCA는 **직선(선형 변환)으로 데이터를 바라본다**. 데이터를 고차원 공간에서 가장 분산이 큰 직선 방향으로 투영한다. 이 때문에 비선형 구조를 가진 데이터에서는 중요한 패턴을 놓친다.

"공간 유지"라는 표현은 PCA가 고차원 공간에서 데이터의 구조(거리와 분산의 관계)를 저차원에서도 가능한 한 유지한다는 의미다.

### PCA의 작동 원리 5단계

```
1. 데이터 표준화:  평균=0, 분산=1로 변환 (스케일 영향 제거)

2. 공분산 행렬 계산:
   C = (1/n) XᵀX  → p×p 정방행렬

3. SVD (또는 고유값 분해):
   X = U Σ Vᵀ
   오른쪽 특이벡터 V의 열 = 주성분 방향

4. 고유값(분산) 큰 순으로 k개 선택:
   전체 분산의 95% 이상을 설명하는 최소 k 선택

5. 데이터를 k개 주성분으로 투영:
   Z = X @ V_k   → (n×p) 행렬이 (n×k)로 축소
```

### PCA가 분산을 최대화하는 이유

주성분 1(PC1)은 다음 최적화 문제의 해다.

$$\mathbf{w}_1 = \arg\max_{|\mathbf{w}|=1} \text{Var}(X\mathbf{w}) = \arg\max_{|\mathbf{w}|=1} \mathbf{w}^T C \mathbf{w}$$

제약 조건 $|\mathbf{w}|=1$에서 이 문제의 해가 공분산 행렬 $C$의 **가장 큰 고유값에 대응하는 고유벡터**임이 라그랑주 승수법으로 증명된다. 그래서 고유값 분해가 PCA의 수학적 기반이 된다.

### 변수가 주성분에 미치는 기여도 (Loading)

로딩 행렬은 각 원본 변수가 각 주성분에 얼마나 기여하는지 보여준다.

```
예시:
  PC1: 키(0.8), 몸무게(0.7), 나이(0.1), 연봉(0.2)
  → PC1은 주로 "신체 크기" 요인 (키와 몸무게 기여 높음)

  PC2: 키(0.1), 몸무게(-0.2), 나이(0.9), 연봉(0.8)
  → PC2는 주로 "경력/나이" 요인

로딩이 양수: 해당 변수가 증가하면 주성분값도 증가
로딩이 음수: 해당 변수가 증가하면 주성분값은 감소
```

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 유방암 데이터 (30개 특성)
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# 표준화 필수
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# ── 전체 PCA 분석 ──────────────────────────────────────
pca_full = PCA()
pca_full.fit(X_sc)

evr        = pca_full.explained_variance_ratio_
cumulative = np.cumsum(evr)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 스크리 플롯
axes[0].bar(range(1, 11), evr[:10], alpha=0.7, label='개별 설명 분산')
axes[0].plot(range(1, 11), cumulative[:10], 'ro-', label='누적 설명 분산')
axes[0].axhline(y=0.95, color='green', linestyle='--', label='95% 기준선')
axes[0].set_xlabel('주성분 번호')
axes[0].set_ylabel('설명 분산 비율')
axes[0].set_title('스크리 플롯')
axes[0].legend()

# 95% 분산에 필요한 성분 수
n_95 = np.searchsorted(cumulative, 0.95) + 1
print(f'95% 분산 유지: {n_95}개 성분 필요 (전체 {X.shape[1]}개)')

# ── 2D 시각화 ──────────────────────────────────────────
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_sc)

axes[1].scatter(X_2d[y==0, 0], X_2d[y==0, 1], alpha=0.5, s=20, label='악성')
axes[1].scatter(X_2d[y==1, 0], X_2d[y==1, 1], alpha=0.5, s=20, label='양성')
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PCA 2D 시각화')
axes[1].legend()

# ── Loading 히트맵 (변수 기여도) ──────────────────────
pca_k = PCA(n_components=5)
pca_k.fit(X_sc)
loadings = pd.DataFrame(
    pca_k.components_,
    index=[f'PC{i+1}' for i in range(5)],
    columns=feature_names
)

im = axes[2].imshow(loadings.values, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
axes[2].set_xticks(range(len(feature_names)))
axes[2].set_xticklabels(feature_names, rotation=90, fontsize=7)
axes[2].set_yticks(range(5))
axes[2].set_yticklabels([f'PC{i+1}' for i in range(5)])
axes[2].set_title('변수 → 주성분 기여도 (Loading)')
plt.colorbar(im, ax=axes[2], fraction=0.02)

plt.tight_layout()
plt.show()

# 차원 축소
pca_final = PCA(n_components=0.95)
X_reduced = pca_final.fit_transform(X_sc)
print(f'차원 축소: {X_sc.shape} → {X_reduced.shape}')
```

---

## 5. PCA 단점 극복 — 변형 PCA 5종

### KernelPCA — 비선형 PCA

PCA는 선형 변환만 가능하다. 데이터가 비선형 구조를 가질 때 PCA는 중요한 정보를 놓친다.

KernelPCA는 **커널 트릭(Kernel Trick)** 을 사용한다. 데이터를 명시적으로 고차원으로 변환하지 않고, 커널 함수를 통해 고차원에서의 내적값만 계산해서 비선형 PCA를 수행한다.

대표적인 커널 3가지:

```
방사형 커널 (RBF, Radial Basis Function):
  K(x, z) = exp(-γ‖x-z‖²)
  → 두 점이 가까울수록 K≈1, 멀수록 K≈0
  → 지역적 유사도 반영. 가장 범용적이라 기본값으로 많이 사용

다항식 커널 (Polynomial):
  K(x, z) = (γxᵀz + r)^d
  → d차 다항 변환에 해당
  → "x의 2차항, 교차항까지 고려한 PCA"와 동일

시그모이드 커널 (Sigmoid):
  K(x, z) = tanh(γxᵀz + r)
  → 신경망의 활성화 함수와 유사한 구조
```

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

X_circles, y_circles = make_circles(n_samples=200, noise=0.1, random_state=42)

pca_linear = PCA(n_components=2)
X_pca      = pca_linear.fit_transform(X_circles)

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15, fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X_circles)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, X_plot, title in zip(axes,
    [X_circles, X_pca, X_kpca],
    ['원본 (동심원)', 'PCA 실패 (선형)', 'KernelPCA 성공 (rbf)']):
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_circles, cmap='bwr', s=20, alpha=0.8)
    ax.set_title(title)
plt.tight_layout()
plt.show()
```

### SparsePCA — 해석 가능한 희소 주성분

일반 PCA 주성분은 모든 변수의 선형 조합이라 "이 주성분이 무엇을 의미하는지" 해석하기 어렵다.

SparsePCA는 **L1 규제(Lasso)를 로딩 행렬에 적용**해서 대부분의 로딩을 0으로 강제한다. L1 패널티는 작은 계수를 정확히 0으로 만드는 성질이 있어 소수의 변수만으로 주성분을 구성하게 된다. 이렇게 변수를 단순화해서 "이 주성분이 무엇을 의미하는지" 해석 가능하게 만든다.

```
일반 PCA:   PC1 = 0.23×x₁ + 0.18×x₂ + 0.31×x₃ + 0.27×x₄ + ...  ← 모두 관여
SparsePCA:  PC1 = 0.00×x₁ + 0.00×x₂ + 0.89×x₃ + 0.00×x₄ + ...  ← x₃만 관여

"이 주성분은 x₃ 변수에 의해 결정된다"고 명확히 해석 가능
```

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=5, alpha=1.0, random_state=42, n_jobs=-1)
spca.fit(X_sc)

# 희소성 확인
total_loadings    = spca.components_.size
nonzero_loadings  = (spca.components_ != 0).sum()
print(f'비제로 로딩 비율: {nonzero_loadings/total_loadings:.1%}')
```

### IncrementalPCA — 메모리 초과 데이터 처리

일반 PCA는 공분산 행렬 계산을 위해 전체 데이터를 메모리에 올려야 한다.  
100만 행 × 1000열 데이터라면 8GB 이상이 필요할 수 있다.

IncrementalPCA는 **미니배치 단위로 순차 학습**해서 메모리 사용량을 크게 줄인다. numpy의 `np.memmap`(MMAP)과 함께 사용하면 디스크에 저장된 대용량 파일을 메모리에 모두 올리지 않고 일부씩 읽어가며 PCA를 처리할 수 있다.

```
일반 PCA:       전체 데이터 → 공분산 행렬 → 고유값 분해  (메모리: O(p²))
IncrementalPCA: 배치1 학습 → 배치2로 업데이트 → ...     (메모리: O(batch_size × p))
```

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

ipca = IncrementalPCA(n_components=10, batch_size=100)
for i in range(0, len(X_sc), 100):
    ipca.partial_fit(X_sc[i:i+100])
X_ipca = ipca.transform(X_sc)
print(f'IncrementalPCA: {X_sc.shape} → {X_ipca.shape}')

# MMAP과 함께 사용: 디스크의 대용량 파일을 일부씩 읽기
# X_mmap = np.load('large_data.npy', mmap_mode='r')   # 메모리에 올리지 않고 읽기
# for i in range(0, len(X_mmap), batch_size):
#     ipca.partial_fit(X_mmap[i:i+batch_size])
```

### RandomizedPCA — 근사 고속 PCA

전체 특이값을 계산하지 않고 **상위 k개만 근사적으로** 계산한다. Halko et al. (2011)의 확률적 알고리즘을 사용해서 정확도를 약간 희생하는 대신 속도가 훨씬 빠르다.

sklearn에서는 별도의 `RandomizedPCA` 클래스가 없다. `PCA(svd_solver='randomized')`로 사용한다. sklearn이 `n_components`가 작고 데이터가 크면 자동으로 이 방식을 선택하기도 한다.

```python
from sklearn.decomposition import PCA

pca_rand = PCA(n_components=10, svd_solver='randomized', random_state=42)
X_rand = pca_rand.fit_transform(X_sc)

# 비교: 전체 vs 근사
import time
for solver in ['full', 'randomized']:
    t0 = time.time()
    pca_s = PCA(n_components=10, svd_solver=solver)
    pca_s.fit(X_sc)
    print(f'{solver:12s}: {time.time()-t0:.4f}초')
```

### TruncatedSVD — 희소 행렬과 LSA

TF-IDF 행렬처럼 0이 대부분인 **희소 행렬(Sparse Matrix)** 에서는 일반 PCA가 비효율적이다.  
표준화를 하면 희소성이 깨지고, 메모리가 폭발한다.

TruncatedSVD는 **표준화 없이 희소 행렬에 직접 SVD를 적용**한다.  
텍스트에 적용하면 **LSA(Latent Semantic Analysis)** 라 부른다.

```
LSA (Latent Semantic Analysis):
  문서-단어 TF-IDF 행렬 (매우 희소)
         ↓ TruncatedSVD
  문서 → 잠재 의미 100차원 벡터
  
  "고양이"와 "야옹"은 다른 단어지만 같은 문서에 자주 등장
  → 잠재 공간에서 비슷한 위치 → 의미적 유사도 포착
```

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='train',
                           categories=['sci.med', 'sci.space', 'rec.sports.hockey'])
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(news.data)  # 희소 행렬

print(f'TF-IDF 행렬 크기: {X_tfidf.shape}')
print(f'밀도: {X_tfidf.nnz / (X_tfidf.shape[0]*X_tfidf.shape[1]):.4%}')  # 매우 희소

tsvd = TruncatedSVD(n_components=100, random_state=42)
X_lsa = tsvd.fit_transform(X_tfidf)
print(f'LSA 변환: {X_tfidf.shape} → {X_lsa.shape}')
print(f'설명 분산: {tsvd.explained_variance_ratio_.sum():.2%}')
```

---

## 6. NMF (Non-Negative Matrix Factorization)

비음수 행렬 분해다. "비음수 어떤 요소로 구성되었나"를 파악하는 것이 목적이다.

NMF가 필요한 이유는 PCA의 구조적 특성 때문이다. **PCA는 데이터를 표준화할 때 평균을 빼기 때문에 음수 값이 필연적으로 생긴다**. 주성분의 로딩도 양수와 음수가 섞인다. 텍스트(단어 빈도는 0 이상), 이미지(픽셀값은 0~255), 음향 신호(에너지는 0 이상)처럼 **물리적으로 음수가 존재할 수 없는 데이터**에서는 이 음수 성분이 해석을 방해하고 의미 없는 결과를 낳는다.

$$M_{m \times n} \approx W_{m \times k} \times H_{k \times n}, \quad W \geq 0, \quad H \geq 0$$

- $W$: 기저(basis) — $k$개의 패턴 (얼굴이라면 눈, 코, 입 등)
- $H$: 계수(coefficient) — 각 데이터가 각 패턴을 얼마나 포함하는지

PCA와의 핵심 차이는 음수를 허용하지 않는다는 것이다. 음수 없이 덧셈만으로 조합하므로 **부분들의 합(Parts-based Representation)** 으로 해석할 수 있다.

```
PCA: "이 얼굴 = +0.8 × 패턴A - 0.3 × 패턴B + ..."  ← 음수 조합
NMF: "이 얼굴 = 0.8 × 눈 패턴 + 0.6 × 코 패턴 + ..."  ← 모두 양수 조합

NMF가 더 직관적: "이 요소가 얼마나 들어 있는가"로 해석
```

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='train',
                           categories=['sci.med', 'sci.space', 'rec.sports.hockey'])
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(news.data)

nmf = NMF(n_components=3, random_state=42, max_iter=500)
W = nmf.fit_transform(X_tfidf)   # 문서 × 토픽 (각 문서의 토픽 비중)
H = nmf.components_               # 토픽 × 단어

feature_names_nmf = vectorizer.get_feature_names_out()
print('NMF 토픽별 상위 단어:')
for i, topic in enumerate(H):
    top_words = [feature_names_nmf[j] for j in topic.argsort()[:-11:-1]]
    print(f'  토픽 {i+1}: {", ".join(top_words)}')
```

---

## 7. ICA (Independent Component Analysis) — 방향 기반

**통계적으로 독립인 성분을 분리**하는 방법이다.

PCA가 "분산을 최대화하는 직교 방향"을 찾는다면, ICA는 "성분들이 서로 통계적으로 독립이 되는 방향"을 찾는다.

```
독립 vs 무상관:
  무상관: 선형 관계가 없음 (공분산 = 0)
  독립:   어떤 통계적 관계도 없음 (훨씬 강한 조건)
  
  독립 → 무상관 (O)
  무상관 → 독립 (X, 비선형 관계 있을 수 있음)
  
  PCA는 무상관 성분 생성 / ICA는 독립 성분 생성
```

칵테일 파티 문제(여러 목소리 분리), 뇌파(EEG) 분석에서 잡음 분리, fMRI 뇌 활동 패턴 분리 등에 활용된다.

```python
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 8, 2000)

# 원본 독립 신호 3개
s1 = np.sin(2 * t)
s2 = np.sign(np.sin(3 * t))
s3 = np.random.laplace(size=len(t))

S = np.column_stack([s1, s2, s3])
S /= S.std(axis=0)

# 혼합 행렬
A_mix = np.array([[1.0, 0.5, 1.5],
                   [1.0, 2.0, 1.0],
                   [1.0, 1.0, 2.0]])
X_mixed = S @ A_mix.T

ica = FastICA(n_components=3, random_state=42, max_iter=1000)
S_recovered = ica.fit_transform(X_mixed)

fig, axes = plt.subplots(3, 3, figsize=(15, 9))
for col, (sigs, title) in enumerate([
    (S.T, '원본 신호'),
    (X_mixed.T, '혼합 신호 (관측)'),
    (S_recovered.T, 'ICA 복원 신호')
]):
    for row in range(3):
        axes[row, col].plot(t[:300], sigs[row, :300], linewidth=0.8)
        if row == 0:
            axes[row, col].set_title(title, fontsize=11)
plt.tight_layout()
plt.show()
```

---

## 8. 비선형 Manifold — MDS, LLE, t-SNE, UMAP

고차원 데이터가 **저차원의 비선형 구조(다양체)** 위에 있다는 가정에서 출발한다.

```
스위스 롤 예시:
  3D 공간의 나선형 띠 → 실제로는 2D 평면을 말은 것
  PCA로 투영하면 말린 구조가 펼쳐지지 않음 (선형이라)
  Manifold 방법은 "펼치는" 능력이 있음
```

LLE, t-SNE, UMAP은 모두 **그래프 기반(Graph-based)** 방법이다. 데이터 포인트를 노드로, 유사한 점들 사이를 엣지로 연결한 그래프를 만들고, 이 그래프 구조를 저차원에서 재현한다. **KNN이 이 그래프를 만드는 역할**을 한다.

```
그래프 기반 manifold의 흐름:

데이터 포인트들
   ↓ KNN으로 이웃 관계 그래프 생성
각 점 → 가까운 k개 이웃과 엣지로 연결
   ↓ 이 그래프 구조를 저차원에서 보존
저차원 임베딩

가까운 이웃(연결된 노드)만 고려 → 지역적(LLE, t-SNE)
   : 클러스터 내부 세밀한 구조는 잘 보존하지만 클러스터 간 관계는 신뢰하기 어려움

먼 점까지 고려 → 전역적(Isomap, UMAP)
   : 전체 구조를 보존하려 해서 클러스터 간 거리도 어느 정도 의미 있음

한 가지 주의할 점이 있다. Manifold 방법들은 내부적으로 $n \times n$ 크기의 거리 행렬(Distance Matrix)을 사용한다. 샘플 수 × 샘플 수로 구성되어 정방 형태이지만, 이것은 PCA에서 쓰는 "특성 수 × 특성 수" 공분산 행렬과 전혀 다른 개념이다. Manifold의 핵심은 정방행렬이 아니라 **점 간 거리 관계의 보존**이다.
```

### MDS — 거리 관계 보존

고차원에서의 **점 간 거리를 저차원에서도 보존**한다.

```
두 점이 고차원에서 멀었으면 저차원에서도 멀게,
두 점이 고차원에서 가까웠으면 저차원에서도 가깝게 배치

스트레스 함수 = Σ(원본 거리 - 저차원 거리)² / Σ원본 거리²
→ 이 값을 최소화하도록 저차원 좌표 최적화
```

```python
from sklearn.manifold import MDS

mds = MDS(n_components=2, metric=True, dissimilarity='euclidean', random_state=42)
X_mds = mds.fit_transform(X_sc[:100])   # 계산 비용이 크므로 일부만
print(f'MDS 스트레스: {mds.stress_:.4f}')
```

### LLE — 지역 선형 관계 보존

각 점을 **이웃들의 선형 조합으로 표현하는 가중치**를 고차원에서 구하고, 저차원에서도 같은 가중치 관계가 성립하도록 임베딩한다.

```
고차원: xᵢ ≈ Σⱼ wᵢⱼ xⱼ  (이웃 j들의 가중 평균)
저차원에서도 이 가중치 wᵢⱼ를 그대로 유지
→ 지역 구조(이웃 관계)가 보존됨

단점: transform() 없어서 새 데이터 변환 불가
```

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=12,
                              method='standard', random_state=42)
X_lle = lle.fit_transform(X_sc[:200])
```

### t-SNE — 시각화 특화

t분포 기반 확률 모델로 고차원의 이웃 관계를 2D/3D에 재현한다.

```
원리:
  고차원: 이웃 관계를 정규분포 확률로 모델링
          Pᵢⱼ = exp(-‖xᵢ-xⱼ‖²/2σ²) / Σ ...

  저차원: 이웃 관계를 t분포(자유도=1) 확률로 모델링
          Qᵢⱼ = (1+‖yᵢ-yⱼ‖²)⁻¹ / Σ ...

  KL 발산 최소화: KL(P‖Q) → 고차원 관계를 저차원에서 재현

t분포를 사용하는 이유 (crowding problem 해결):
  고차원의 많은 점들을 2D에 압축할 때 중심에 뭉치는 문제
  t분포의 두꺼운 꼬리가 먼 점들을 더 멀리 밀어내 공간 활용
```

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 권장 방법: PCA로 먼저 압축 후 t-SNE (속도+안정성 향상)
pca_50 = PCA(n_components=min(50, X_sc.shape[1]))
X_pca50 = pca_50.fit_transform(X_sc)

tsne = TSNE(n_components=2, perplexity=30,
            learning_rate='auto', n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_pca50)

# 주의사항 4가지:
# 1. transform() 없음 — 매번 전체 데이터 재학습
# 2. 실행마다 다른 결과 — random_state 고정 필수
# 3. 클러스터 간 거리 무의미 — 상대적 위치만 의미
# 4. perplexity 조정 — 데이터 크기에 따라 5~50 범위
```

### UMAP — t-SNE의 개선판

위상수학(topology)과 리만 기하학 기반으로 만들어진 방법으로 t-SNE의 한계를 거의 모두 극복했다.

```
t-SNE vs UMAP 비교:

t-SNE:
  지역 구조 ✅  전역 구조 ❌  transform() ❌  속도: 느림
  클러스터 내부 구조는 잘 보존하지만 클러스터 간 관계 신뢰 불가

UMAP:
  지역 구조 ✅  전역 구조 ✅  transform() ✅  속도: 빠름
  클러스터 간 거리도 어느 정도 의미 있음

파라미터:
  n_neighbors: 지역 vs 전역 균형 조절
               작으면(5~15): 지역 세부 구조 강조
               크면(50~100): 전역 구조 강조
  min_dist:    저차원에서 점들이 얼마나 뭉치는지
               작으면(0.0~0.1): 촘촘히 뭉침
               크면(0.5~1.0): 고르게 펼침
```

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric='euclidean', random_state=42)
X_umap = reducer.fit_transform(X_sc)

# transform() 가능 — 새 데이터도 같은 공간에 투영
# X_new_umap = reducer.transform(X_test_sc)
```

---

## 9. Random Projection

**존슨-린덴스트라우스(JL) 보조정리**에 따라 무작위 행렬을 곱해도 점 간의 거리 관계가 근사적으로 보존된다.

```
JL 보조정리:
  n개의 점이 있을 때, 목표 차원 k ≥ 8 ln(n) / ε²이면
  임의의 두 점 사이 거리 오차가 ε 이하로 보장됨

  → 데이터 수만 알면 필요한 차원 수를 계산할 수 있음
  → PCA보다 훨씬 빠름 (최적화 없이 무작위 투영)
  → 해석은 어렵지만 다운스트림 ML 성능 유지
```

```python
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import numpy as np
import time

X_large = np.random.randn(1000, 500)

# 가우시안 랜덤 투영
t0 = time.time()
grp = GaussianRandomProjection(n_components='auto', eps=0.1, random_state=42)
X_grp = grp.fit_transform(X_large)
print(f'Gaussian RP: {X_large.shape} → {X_grp.shape}, {time.time()-t0:.3f}초')

# 희소 랜덤 투영 (가우시안보다 더 빠름)
t0 = time.time()
srp = SparseRandomProjection(n_components='auto', eps=0.1, random_state=42)
X_srp = srp.fit_transform(X_large)
print(f'Sparse RP:   {X_large.shape} → {X_srp.shape}, {time.time()-t0:.3f}초')

# PCA와 속도 비교
t0 = time.time()
pca = PCA(n_components=X_grp.shape[1])
X_pca_large = pca.fit_transform(X_large)
print(f'PCA:         {X_large.shape} → {X_pca_large.shape}, {time.time()-t0:.3f}초')
```

---

## 10. Feature Agglomerative Clustering (FAC)

일반 군집화가 **샘플(행)을 묶는다**면, FAC는 **특성(열)을 묶는다**.

유사한 패턴을 보이는 변수들을 하나로 합쳐 대표값(평균)으로 교체한다.

```
예: 30개 의학 특성 중 "세포 크기", "세포 면적", "세포 반지름"이
    서로 매우 상관이 높음 → 하나의 "크기" 변수로 합칠 수 있음

일반 군집화:  샘플 500개 → 3개 그룹으로 요약
FAC:          특성 30개 → 10개 그룹으로 요약
```

```python
from sklearn.cluster import FeatureAgglomeration

fa = FeatureAgglomeration(n_clusters=10,
                           linkage='ward',       # 연결법 선택
                           pooling_func=np.mean) # 묶인 변수들의 집계 방법
X_fa = fa.fit_transform(X_sc)
print(f'FAC: {X_sc.shape} → {X_fa.shape}')

# 어떤 변수들이 같은 그룹으로 묶였는지
labels = fa.labels_
for group in range(5):
    group_features = [feature_names[i] for i, l in enumerate(labels) if l == group]
    print(f'그룹 {group}: {group_features}')
```

---

## 11. 방법별 전체 비교와 선택 가이드

```
데이터가 선형 구조인가?
   ├─ YES
   │   ├─ 빠른 처리가 필요하다?      → Random Projection
   │   ├─ 텍스트/희소 행렬이다?      → TruncatedSVD (LSA)
   │   ├─ 대용량으로 메모리가 부족?  → IncrementalPCA
   │   ├─ 빠른 근사가 필요?          → RandomizedPCA
   │   ├─ 음수가 없어야 한다?        → NMF (부분 합성)
   │   ├─ 신호 분리가 목적이다?      → ICA
   │   ├─ 변수를 줄이고 싶다?        → FAC
   │   ├─ 해석 가능성이 중요하다?    → SparsePCA
   │   └─ 일반적인 경우?             → PCA  ← 항상 첫 번째로 시도
   │
   └─ NO (비선형 구조)
       ├─ 시각화만 필요하다?         → t-SNE (정밀) 또는 UMAP (빠름)
       ├─ 새 데이터 변환도 필요?     → UMAP, KernelPCA
       ├─ 전역 거리 관계도 중요?     → UMAP, MDS
       ├─ 지역 선형 구조 보존?       → LLE
       └─ 비선형 선형 분리 필요?     → KernelPCA
```

|방법|유형|척도|새 데이터|해석 가능성|속도|핵심 한줄 설명|
|---|---|---|---|---|---|---|
|PCA|선형|분산|✅|중간|빠름|분산 최대화 방향으로 투영|
|KernelPCA|비선형|분산|✅|낮음|보통|커널 트릭으로 비선형 PCA|
|SparsePCA|선형|분산|✅|높음|느림|소수 변수만으로 주성분 구성|
|IncrementalPCA|선형|분산|✅|중간|빠름|미니배치로 메모리 절약|
|RandomizedPCA|선형|분산|✅|중간|매우 빠름|근사 고속 계산|
|TruncatedSVD|선형|분산|✅|낮음|빠름|희소 행렬 전용, LSA|
|NMF|선형|분산|✅|높음|보통|음수 없는 부분 합성|
|ICA|선형|방향|✅|중간|보통|통계적 독립 신호 분리|
|MDS|비선형|거리|✅|낮음|느림|거리 관계 보존|
|LLE|비선형|거리|❌|낮음|보통|지역 선형 관계 보존|
|t-SNE|비선형|거리|❌|낮음|느림|2D/3D 시각화 특화|
|UMAP|비선형|거리|✅|낮음|빠름|t-SNE 단점 극복, 실무 표준|
|Random Projection|선형|거리|✅|없음|매우 빠름|JL 보조정리 기반 무작위 투영|
|FAC|선형|분산|✅|높음|빠름|변수(열) 군집화|