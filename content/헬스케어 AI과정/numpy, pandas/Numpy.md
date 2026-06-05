---
title: 027 numpy
tag:
  - 헬스케어 ai
  - numpy
description: 2605012 수업 내용 정리
---

# 기본 데이터 타입과 NumPy: ndarray

---

## 1. Python 과학 계산 생태계

Python은 범용 언어지만, 과학 계산과 데이터 분석에 필요한 기능들은 별도 라이브러리로 제공된다.

```
Python Core
   ├─ NumPy      핵심 수치 계산 (배열, 행렬, 선형대수)
   ├─ SciPy      NumPy 기반 고급 과학 계산 (통계, 최적화, 신호처리)
   └─ Matplotlib 시각화
```

이 생태계는 MATLAB과 기능적으로 대응된다. MATLAB에서 하던 행렬 연산, 수치 계산, 시각화를 Python에서 이 세 라이브러리로 대체할 수 있다.

- **NumPy**: MATLAB의 행렬 연산에 대응
- **SciPy**: MATLAB의 Toolbox에 대응
- **Matplotlib**: MATLAB의 plot 기능에 대응

---

## 2. ndarray란

NumPy의 핵심 자료구조다. **N-dimensional Array**, 즉 N차원 배열을 의미한다.

Python의 기본 리스트와 다르게 **단일 데이터 타입만 저장**하고, 메모리에 **연속적으로 배치**되기 때문에 연산이 매우 빠르다.

```python
import numpy as np

a = np.array([1, 2, 3])          # 1D array (Vector)
b = np.array([[1, 2, 3],
              [4, 5, 6]])         # 2D array (Matrix)
c = np.zeros((2, 3, 4))          # 3D array (Tensor)
```

### Python 리스트 vs ndarray

|구분|Python 리스트|NumPy ndarray|
|---|---|---|
|데이터 타입|혼합 가능|단일 타입만|
|메모리|불연속적|연속적|
|연산 속도|느림|빠름|
|벡터화 연산|불가|가능|
|메모리 효율|낮음|높음|

---

## 3. 배열의 차원 구조

```
1D array (Vector)    → [1, 2, 3]
2D array (Matrix)    → [[1, 2, 3],
                         [4, 5, 6]]
3D array (Tensor)    → 행렬을 여러 겹 쌓은 구조
```

축(axis)의 방향:

- **axis 0**: 행 방향 (아래쪽)
- **axis 1**: 열 방향 (오른쪽)
- **axis 2**: 깊이 방향 (3D에서 추가)

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

np.sum(arr, axis=0)   # 열별 합계 → [5, 7, 9]
np.sum(arr, axis=1)   # 행별 합계 → [6, 15]
```

---

## 4. ndarray의 주요 속성

속성은 배열의 메타 정보를 담고 있다. 함수 호출 없이 `.`으로 바로 접근한다.

|속성|설명|예시|
|---|---|---|
|`ndim`|차원 수|`2` (2D 배열)|
|`dtype`|데이터 타입|`int64`, `float32` 등|
|`shape`|각 차원의 크기|`(2, 3)` → 2행 3열|
|`strides`|다음 원소로 이동하기 위한 바이트 수|메모리 배치 방식 결정|

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.ndim)    # 2
print(arr.dtype)   # int64
print(arr.shape)   # (2, 3)
print(arr.strides) # (24, 8) — int64는 8바이트
```

### strides와 메모리 구조

ndarray는 실제 메모리에 **1차원으로 연속 저장**된다.  
strides는 "다음 행으로 가려면 몇 바이트를 건너뛰어야 하는가"를 알려준다.

```
메모리: [1][2][3][4][5][6]  ← 연속적으로 저장

shape=(2,3), strides=(24, 8)
→ 다음 행: 24바이트(= 3개 × 8바이트) 이동
→ 다음 열: 8바이트 이동
```

이 구조 덕분에 전치(`transpose`)나 슬라이싱이 데이터 복사 없이 strides만 바꿔서 빠르게 처리된다.

---

## 5. NumPy 데이터 타입

ndarray는 **단일 타입**만 저장하기 때문에 타입 선택이 성능과 메모리에 직접 영향을 준다.

|기본 타입|NumPy 타입|설명|
|---|---|---|
|Boolean|`bool`|True / False|
|Integer|`int8`, `int16`, `int32`, `int64`, `int128`, `int`|부호 있는 정수|
|Unsigned Integer|`uint8`, `uint16`, `uint32`, `uint64`, `uint128`, `uint`|부호 없는 정수 (0 이상)|
|Float|`float32`, `float64`, `float`, `longfloat`|실수|
|Complex|`complex64`, `complex128`, `complex`|복소수|
|Strings|`str`, `unicode`|문자열|
|Object|`object`|임의의 Python 객체|
|Records|`void`|구조체형 데이터|

### 타입 선택 기준

- **메모리 절약이 중요한 경우**: `int8`, `float32` 등 작은 타입 사용
- **정밀도가 중요한 경우**: `float64` (기본값)
- **이미지 처리**: 픽셀값 0~255는 `uint8`

```python
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 타입 명시
arr.astype(np.int64)                                  # 타입 변환
```

---

## 6. NumPy 주요 서브모듈

|서브모듈|기능|
|---|---|
|`numpy.core`|핵심 데이터 구조 및 연산|
|`numpy.linalg`|행렬 연산, 역행렬, 고유값 분해, SVD|
|`numpy.random`|난수 생성 및 확률 분포|
|`numpy.fft`|푸리에 변환 (신호 처리)|
|`numpy.polynomial`|다항식 계산|
|`numpy.testing`|테스트 및 디버깅 도구|
|`numpy.ma`|결측값 마스킹 처리|

```python
# numpy.linalg 예시
A = np.array([[2, 1], [1, 3]])
np.linalg.inv(A)        # 역행렬
np.linalg.eig(A)        # 고유값, 고유벡터
np.linalg.svd(A)        # 특이값 분해

# numpy.random 예시
np.random.seed(42)
np.random.randn(3, 3)   # 정규분포 난수
np.random.randint(0, 10, size=(2, 4))  # 정수 난수
```

---

## 7. Ufunc (Universal Function)

ndarray의 각 원소에 **원소별로 빠르게 적용되는 함수**다.  
내부적으로 C로 구현되어 있어 Python 반복문보다 훨씬 빠르다.

```python
arr = np.array([1.0, 4.0, 9.0, 16.0])

np.sqrt(arr)    # [1.0, 2.0, 3.0, 4.0]
np.exp(arr)     # 각 원소에 e^x 적용
np.log(arr)     # 각 원소에 자연로그 적용
np.sin(arr)     # 각 원소에 sin 적용
```

R의 벡터화 연산과 같은 개념이다. 반복문 없이 배열 전체에 한 번에 연산이 적용된다.

---

## 8. 벡터화 연산과 브로드캐스팅

### 벡터화 연산

배열 전체에 한 번에 연산을 적용한다. 반복문보다 수십 배 빠르다.

```python
arr = np.array([1, 2, 3, 4])

arr + 1    # [2, 3, 4, 5]
arr * 2    # [2, 4, 6, 8]
arr ** 2   # [1, 4, 9, 16]
```

### 브로드캐스팅 (Broadcasting)

크기가 다른 배열끼리 연산할 때 작은 배열을 자동으로 확장해서 맞춘다.  
R의 recycling rule과 같은 개념이지만 더 유연하다.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)
b = np.array([10, 20, 30])  # shape (3,)

a + b
# [[11, 22, 33],
#  [14, 25, 36]]
# b가 (2, 3)으로 자동 확장되어 연산됨
```

---

## 9. 속성과 함수의 관계

ndarray는 **속성**과 **함수** 두 가지 방식으로 정보를 제공한다.

- **속성**: 배열 자체의 메타 정보 (`ndim`, `dtype`, `shape`, `strides`)
- **함수**: 배열에 계산을 수행 (`np.sum()`, `np.mean()`, `np.reshape()`)

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 속성 (계산 없이 바로 반환)
arr.shape    # (2, 3)
arr.dtype    # int64

# 함수 (계산 수행)
arr.reshape(3, 2)     # 형태 변경
arr.T                 # 전치
np.mean(arr, axis=0)  # 열별 평균
```

---

## 10. 전체 구조 정리

```
ndarray
   ├─ 속성
   │   ├─ ndim   : 차원 수
   │   ├─ dtype  : 데이터 타입
   │   ├─ shape  : 각 축의 크기
   │   └─ strides: 메모리 이동 단위
   │
   ├─ 차원
   │   ├─ 1D → Vector
   │   ├─ 2D → Matrix
   │   └─ 3D → Tensor
   │
   └─ 서브모듈
       ├─ linalg  : 선형대수
       ├─ random  : 난수/확률분포
       ├─ fft     : 신호처리
       ├─ ma      : 결측값 마스킹
       └─ Ufunc   : 원소별 고속 연산
```