---
title: 028 matplotlib
tag:
  - 국비교육
  - numpy
description: 2605013 수업 내용 정리
---

# Matplotlib 시각화(EDA): 도화지와 축

데이터를 시각적으로 표현하는 Python의 핵심 시각화 라이브러리다.  
NumPy의 배열 데이터와 Pandas의 DataFrame을 그대로 받아 그래프로 변환할 수 있어, 세 라이브러리는 데이터 분석 파이프라인에서 함께 사용된다.

---

## 1. 무엇을 시각화할 것인가 — 정보 표현 목적

그래프를 선택하기 전에 **무엇을 보여주고 싶은지**를 먼저 결정해야 한다.

|목적|그래프 종류|
|---|---|
|**비교**|Bar (막대 그래프)|
|**추세**|Line (선 그래프)|
|**관계**|Scatter (산점도)|
|**분포**|Histogram, Boxplot|

---

## 2. 도식화 방법 — 상태기반 vs 무상태기반

Matplotlib은 두 가지 방식으로 그래프를 그릴 수 있다.

### 상태기반 방식 (pyplot)

`plt.` 함수를 순서대로 호출해서 현재 활성화된 그래프에 누적 적용한다.  
코드가 간결하지만 여러 그래프를 동시에 다루면 복잡해질 수 있다.

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.title("제목")
plt.xlabel("x축")
plt.ylabel("y축")
plt.show()
```

### 무상태기반 방식 (객체지향)

`fig`와 `ax` 객체를 명시적으로 만들어 다룬다.  
서브플롯이 여러 개이거나 세밀한 제어가 필요할 때 적합하다. **실무에서 권장되는 방식**이다.

```python
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title("제목")
ax.set_xlabel("x축")
ax.set_ylabel("y축")
plt.show()
```

---

## 3. Matplotlib의 3개 레이어 구조

Matplotlib 내부는 세 개의 레이어로 나뉜다.

|레이어|역할|
|---|---|
|**Backend Layer**|실제 화면/파일에 출력하는 하위 엔진 (화면 출력, PNG 저장 등)|
|**Artist Layer**|그래프의 모든 시각 요소를 담당 (선, 점, 텍스트, 축 등 객체)|
|**Scripting Layer (pyplot)**|사용자가 직접 다루는 인터페이스. Artist Layer를 편하게 조작|

일반적으로 `plt.`로 호출하는 것이 Scripting Layer이고, `fig`, `ax` 객체가 Artist Layer에 해당한다.

---

## 4. Figure와 Axes — 도화지와 축

### Figure — 캔버스 (도화지)

그래프 전체를 담는 최상위 컨테이너다.  
화면 분할, 전체 크기 조정, 파일 저장 등 전체 레이아웃을 다룰 때 필요하다.

```python
fig = plt.figure(figsize=(10, 6))   # 가로 10인치, 세로 6인치
fig, axes = plt.subplots(2, 3)      # 2행 3열 서브플롯
```

### Axes — 분할 영역 (실제 그래프 영역)

Figure 안에 배치된 개별 그래프 영역이다.  
축, 점, 선, 마커, 레이블, 범례 등 도식화에 필요한 모든 요소를 여기서 다룬다.

하나의 Figure 안에 여러 Axes를 배치할 수 있다.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x, y1)
ax1.set_title("그래프 1")

ax2.scatter(x, y2)
ax2.set_title("그래프 2")
```

### Figure vs Axes 구분

```
Figure (캔버스 전체)
   └─ Axes (그래프 영역 1)
         ├─ axis (x축, y축)
         ├─ Title
         ├─ Legend
         ├─ Grid
         ├─ Spines (경계선)
         └─ 그래픽 객체 (점, 선, 면, 마커...)
   └─ Axes (그래프 영역 2)
         └─ ...
```

---

## 5. 그래프 구성 요소 (Anatomy of a Figure)

|구성 요소|설명|
|---|---|
|**Title**|그래프 제목|
|**axis**|x축, y축|
|**Major tick**|주요 눈금|
|**Minor tick**|보조 눈금|
|**Tick label**|눈금 레이블|
|**Axis label**|축 레이블 (단위나 변수명)|
|**Grid**|격자선|
|**Spines**|그래프 경계선 (위/아래/왼/오른쪽)|
|**Legend**|범례|
|**Markers**|산점도의 점|
|**Line**|선 그래프의 선|

```python
ax.set_title("그래프 제목")
ax.set_xlabel("x축 레이블")
ax.set_ylabel("y축 레이블")
ax.legend()
ax.grid(True)
ax.spines['top'].set_visible(False)    # 위쪽 경계선 제거
ax.spines['right'].set_visible(False)  # 오른쪽 경계선 제거
ax.tick_params(axis='x', rotation=45) # 눈금 레이블 회전
```

---

## 6. 그래프 종류 선택 가이드

데이터 특성과 목적에 따라 그래프를 선택한다.

### 범주형 데이터 — 비교

범주(그룹)별로 값의 크기를 비교할 때 사용한다.  
항목 수가 많으면 가로 막대(`barh`)가 레이블을 읽기 편하고, 비율 구성이 목적이라면 Stacked Bar나 Pie가 적합하다.  
단, Pie Chart는 항목이 많거나 값 차이가 작으면 오히려 읽기 어려워지므로 3~5개 이하일 때 쓰는 것이 좋다.

```python
# Bar Plot
ax.bar(categories, values)
ax.barh(categories, values)          # 가로 막대 — 레이블이 길 때 유리
ax.bar(x, y1, label='A')
ax.bar(x, y2, bottom=y1, label='B')  # Stacked Bar — 전체 대비 구성 비율

# Pie Chart — 전체 중 각 항목의 비율
ax.pie(values, labels=labels, autopct='%1.1f%%')
```

### 수치형 분포

단일 변수의 값이 어떻게 퍼져 있는지 파악할 때 사용한다.  
Histogram은 구간(bin)으로 쪼개서 빈도를 보여주고, KDE는 그것을 부드러운 곡선으로 표현한다.  
Box Plot은 이상치 탐지에 강하고, Violin Plot은 분포 모양까지 함께 보여준다.

```python
# Histogram — 구간별 빈도. bin 수에 따라 인상이 크게 달라짐
ax.hist(data, bins=20, edgecolor='white')

# KDE (커널 밀도 추정) — 히스토그램을 부드럽게 연속 곡선으로 표현
import seaborn as sns
sns.kdeplot(data, ax=ax)

# Box Plot — 중위수, IQR, 이상치를 한눈에 파악
ax.boxplot(data)

# Violin Plot — 박스플롯 + KDE. 분포 모양까지 함께 확인
ax.violinplot(data)
```

### 수치형 관계

두 변수 사이의 관계(상관성, 패턴, 군집)를 파악할 때 사용한다.  
Scatter는 가장 기본적인 관계 시각화이고, Bubble은 세 번째 변수를 점의 크기로 추가 표현한다.  
Heatmap은 상관 행렬처럼 변수 쌍이 많을 때 전체 관계를 한 번에 보는 데 유용하다.

```python
# Scatter Plot — 두 변수의 관계. 색·크기로 추가 변수 표현 가능
ax.scatter(x, y, c=colors, s=sizes, alpha=0.6)

# Bubble Chart — Scatter에서 크기(s)로 세 번째 변수 시각화
ax.scatter(x, y, s=sizes*100, alpha=0.5)

# Heatmap — 상관 행렬 등 변수 간 관계를 색상 강도로 표현
import seaborn as sns
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
```

### 시계열

시간 흐름에 따른 변화나 추세를 볼 때 사용한다.  
Line Plot이 가장 기본이고, Area Chart는 누적량이나 범위를 강조할 때 쓴다.  
OHLC는 주식처럼 구간 내 시가/고가/저가/종가가 있는 데이터에 특화된 형식이다.

```python
# Line Plot — 시간에 따른 수치 변화. 연속성 강조
ax.plot(dates, values, linewidth=2)

# Area Chart — 선 아래 면적을 채워 누적량이나 범위 강조
ax.fill_between(dates, values, alpha=0.3)

# OHLC (주가 차트) — 시가/고가/저가/종가 네 값을 동시에 표현
# mplfinance 라이브러리 사용
```

### 3차원/공간

두 개의 독립변수로 하나의 종속변수를 표현하거나, 지리적 데이터를 시각화할 때 사용한다.  
Surface는 연속적인 3D 면을, Contour는 그 면을 위에서 내려다본 등고선 형태로 보여준다.  
2D로 표현하기 어려운 패턴이나 지형, 최적화 함수 시각화에 유용하다.

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)     # Surface — 연속 3D 곡면
ax.contour(X, Y, Z)          # Contour — 3D 면을 2D 등고선으로 투영
```

---

## 7. 출력 이미지 형식

### Vector 이미지

수학적인 **수식(점, 선, 곡선, 도형)으로 그래픽을 표현**하는 방식이다.  
이미지를 저장할 때 "이 좌표에서 저 좌표까지 이런 색의 선을 그려라"는 명령을 저장하기 때문에, 얼마나 확대하든 다시 계산해서 그려내므로 **해상도에 무관하게 선명하다.**

- 확대/축소해도 깨지지 않음
- 파일 크기가 작다 (복잡한 그래픽은 예외)
- 텍스트, 로고, 아이콘, 다이어그램처럼 선과 면 중심의 그래픽에 적합
- 형식: **SVG, PDF, EPS, AI, CGM, CDR**

```
저장 방식: "A좌표 → B좌표까지 검은 선, 굵기 2px"
→ 확대해도 새로 계산해서 그림 → 항상 선명
```

### Raster 이미지

**픽셀(Pixel)의 격자**로 이미지를 표현하는 방식이다.  
이미지를 저장할 때 각 픽셀의 색상 값을 그대로 저장하기 때문에, 확대하면 픽셀이 보이면서 깨진다.  
해상도는 **DPI(Dots Per Inch)** 로 결정된다.

- 사진, 복잡한 색상 표현에 적합
- 파일 크기가 상대적으로 크다
- 확대하면 계단 현상(aliasing) 발생
- 형식: **PNG, JPEG, TIFF, BMP, GIF, PCX**

```
저장 방식: [픽셀1: (255,0,0)][픽셀2: (254,1,0)]...
→ 확대하면 픽셀 자체가 커짐 → 깨짐
```

### Vector vs Raster 비교

|구분|Vector|Raster|
|---|---|---|
|표현 방식|수식 (점, 선, 면)|픽셀 격자|
|확대 시|선명함 유지|깨짐|
|파일 크기|작음 (단순 그래픽)|상대적으로 큼|
|색상 표현|단색, 그라데이션|사진 수준 표현 가능|
|적합한 용도|로고, 아이콘, 다이어그램, 그래프|사진, 스크린샷, 복잡한 이미지|
|대표 형식|SVG, PDF, EPS|PNG, JPEG, TIFF|

### DPI (Dots Per Inch)

Raster 이미지의 해상도 단위다. 1인치 안에 몇 개의 픽셀이 들어가는지를 나타낸다.  
DPI가 높을수록 더 촘촘하게 픽셀이 찍혀 선명해지지만, 파일 크기도 커진다.

|용도|권장 DPI|
|---|---|
|화면 표시 (웹, 발표)|72 ~ 96|
|일반 인쇄|150 ~ 200|
|보고서 / 논문|300 이상|
|고품질 인쇄|600 이상|

```python
# 저장 방식 선택
fig.savefig("plot.png", dpi=300, bbox_inches='tight')  # Raster, 고해상도
fig.savefig("plot.svg")                                 # Vector, 확대 무관
fig.savefig("plot.pdf", bbox_inches='tight')            # Vector, 논문/보고서용
```

`bbox_inches='tight'`는 여백을 자동으로 맞춰 잘리는 부분 없이 저장해준다.

---

## 8. NumPy와 Matplotlib의 연결

NumPy 배열은 Matplotlib의 기본 입력 형식이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 수식 시각화
x = np.linspace(-np.pi, np.pi, 300)  # -π ~ π 사이 300개 균등 간격
y_sin = np.sin(x)
y_cos = np.cos(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y_sin, label='sin(x)')
ax.plot(x, y_cos, label='cos(x)', linestyle='--')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.legend()
ax.set_title("삼각함수")
plt.show()
```

### NumPy로 자주 쓰는 데이터 생성

```python
np.linspace(0, 10, 100)    # 균등 간격 100개
np.arange(0, 10, 0.1)      # 0.1 간격으로 생성
np.random.randn(1000)       # 정규분포 난수 1000개
np.random.rand(100, 2)      # 0~1 균일분포 난수 (산점도용)
```

---

## 9. Pandas와 Matplotlib의 연결

Pandas DataFrame은 `.plot()` 메서드로 Matplotlib 그래프를 바로 그릴 수 있다.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'sales': [100, 150, 130, 170],
    'cost':  [80,  100, 90,  120]
})

# DataFrame.plot() — Matplotlib을 내부적으로 사용
df.plot(x='month', y=['sales', 'cost'], kind='bar', figsize=(8, 5))
plt.show()

# kind 옵션
# 'line', 'bar', 'barh', 'hist', 'box', 'kde', 'scatter', 'pie'
```

### EDA에서 자주 쓰는 Pandas 시각화 패턴

```python
# 수치형 변수 분포 한 번에 확인
df.hist(bins=20, figsize=(12, 8))
plt.tight_layout()

# 상관 행렬 히트맵
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)

# 박스플롯으로 이상치 확인
df.boxplot(figsize=(10, 6))

# 산점도 행렬 (변수 간 관계 한눈에 보기)
pd.plotting.scatter_matrix(df, figsize=(10, 10), diagonal='kde')
```

---

## 10. 보간법 (Interpolation)

데이터 포인트 사이의 빈 구간을 채우거나, 측정값들을 부드러운 곡선으로 연결하는 방법이다.  
시각화에서는 선 그래프를 더 자연스럽게 표현하거나, 결측 구간을 추정할 때 사용한다.

### 보간법의 종류

|방법|특징|적합한 경우|
|---|---|---|
|선형 보간|두 점 사이를 직선으로 연결|단순하고 빠름. 급격한 변화 구간에서 부자연스러움|
|스플라인 보간|구간별 다항식으로 부드럽게 연결|부드러운 곡선이 필요한 경우|
|다항식 보간|모든 점을 지나는 하나의 다항식|점이 적을 때. 점이 많으면 진동(룽게 현상) 발생|

---

### 스플라인 보간법 (Spline Interpolation)

데이터 포인트들을 **구간별 저차 다항식(주로 3차)으로 연결**하되, 연결점(knot)에서 **기울기와 곡률이 연속**이 되도록 맞추는 방법이다.

하나의 고차 다항식으로 전체를 연결하면 양 끝에서 심하게 튀는 **룽게 현상(Runge's phenomenon)** 이 발생한다. 스플라인은 구간을 나눠서 이 문제를 피한다.

```
데이터 포인트:  x0  x1  x2  x3  x4
                 ●───●───●───●───●
                  구간1 구간2 구간3 구간4

각 구간에 3차 다항식 S₁(x), S₂(x), S₃(x), S₄(x)를 따로 만들되
연결점에서 값, 1차 미분(기울기), 2차 미분(곡률)이 같아야 함
→ 전체가 자연스럽게 이어진 부드러운 곡선
```

### 3차 스플라인 (Cubic Spline)의 조건

각 구간의 3차 다항식 $S_i(x) = a_i + b_i x + c_i x^2 + d_i x^3$ 에 대해:

- **연속성**: 연결점에서 좌우 값이 같음 ($S_i(x_{i+1}) = S_{i+1}(x_{i+1})$)
- **1차 미분 연속**: 연결점에서 기울기가 같음 (꺾임 없음)
- **2차 미분 연속**: 연결점에서 곡률이 같음 (자연스러운 휨)

이 조건들이 맞춰지면 전체 곡선이 끊김 없이 매끄럽게 이어진다.

### 경계 조건 종류

|종류|설명|
|---|---|
|Natural Spline|양 끝의 2차 미분 = 0. 끝점에서 직선처럼 뻗어나감|
|Clamped Spline|양 끝의 기울기를 직접 지정|
|Not-a-Knot|첫/마지막 두 구간의 3차 계수를 같게 설정. scipy 기본값|

### Python 코드 (scipy)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

# 원본 데이터 (듬성듬성한 측정값)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 2, 1, 3, 2, 4])

# 보간에 사용할 촘촘한 x축
x_fine = np.linspace(0, 5, 300)

# 선형 보간
linear = interp1d(x, y, kind='linear')

# 3차 스플라인 보간
cs = CubicSpline(x, y)

# 비교 시각화
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x, y, color='black', zorder=5, label='원본 데이터')
ax.plot(x_fine, linear(x_fine), linestyle='--', label='선형 보간')
ax.plot(x_fine, cs(x_fine), linewidth=2, label='3차 스플라인')
ax.legend()
ax.set_title("보간법 비교")
plt.show()
```

### Pandas에서 결측값 보간

Pandas의 `interpolate()` 메서드로 DataFrame의 결측값을 보간할 수 있다.

```python
import pandas as pd

df = pd.DataFrame({'value': [1.0, None, None, 4.0, None, 6.0]})

df['linear']  = df['value'].interpolate(method='linear')   # 선형
df['spline']  = df['value'].interpolate(method='spline', order=3)  # 3차 스플라인
df['polynomial'] = df['value'].interpolate(method='polynomial', order=2)  # 2차 다항식

print(df)
```

### 시각화 오류 해결에서의 활용

앞서 EDA 시각화 오류 부분에서 언급한 **빈 데이터(Gap) 문제**를 스플라인 보간으로 해결할 수 있다.

```python
# geom_smooth 대신 직접 스플라인으로 보간 후 시각화
from scipy.interpolate import CubicSpline

# 결측 구간 제거 후 보간
mask = ~np.isnan(y_with_gaps)
cs = CubicSpline(x[mask], y_with_gaps[mask])

ax.scatter(x[mask], y_with_gaps[mask], label='실측값')
ax.plot(x_fine, cs(x_fine), alpha=0.7, label='스플라인 보간')
```

### 주의사항

- 스플라인은 데이터 포인트를 **정확히 통과**한다. 노이즈가 많은 데이터에 쓰면 노이즈까지 그대로 따라간다. 이 경우 보간 대신 `geom_smooth`처럼 **회귀 기반 평활화(smoothing)** 가 더 적합하다.
- 외삽(extrapolation), 즉 데이터 범위 바깥으로 곡선을 연장하면 예측이 급격히 틀려질 수 있어 주의해야 한다.

---

## 11. 전체 흐름 정리

```
데이터 준비 (NumPy array / Pandas DataFrame)
   ↓
목적 결정
   ├─ 비교  → Bar
   ├─ 추세  → Line
   ├─ 관계  → Scatter / Heatmap
   └─ 분포  → Histogram / Box / Violin
   ↓
Figure / Axes 생성
   fig, ax = plt.subplots()
   ↓
그래프 그리기
   ax.plot() / ax.scatter() / ax.hist() ...
   ↓
구성 요소 설정
   제목, 축 레이블, 범례, 눈금, 경계선
   ↓
저장 또는 출력
   fig.savefig() / plt.show()
```