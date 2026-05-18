---
title: 029 pandas
tag:
  - 국비교육
  - numpy
description: 2605013 수업 내용 정리
---

# Pandas

NumPy 기반의 고수준 데이터 분석 도구다.  
ndarray가 수치 계산에 특화된 저수준 구조라면, Pandas는 그 위에 **인덱스, 레이블, 결측값 처리, SQL 스타일 연산** 등을 얹어 실제 데이터 분석에 바로 쓸 수 있게 만든 구조다.

---

## 1. Pandas의 특징

- **자료구조**: NumPy(ndarray) + dict의 결합. 행과 열에 레이블(인덱스)을 붙인 구조
- **함수형 프로그래밍**: 벡터화 연산을 기본으로 사용해 반복문 없이 전체 열에 연산 적용
- **데이터베이스 스타일**: `group by`, `join`, `pivot table` 등 SQL과 유사한 연산 지원
- **엑셀 기능**: pivot table, 조건부 집계 등을 코드로 구현 가능

---

## 2. Pandas, NumPy, R의 관계

### 계층 구조

세 도구는 서로 다른 층위에서 동작한다.

```
R 생태계                    Python 생태계
─────────────────           ─────────────────────────────
data.frame (base R)    ←→   Pandas DataFrame
  벡터 (base R)        ←→   NumPy ndarray
  행렬 (base R)        ←→   NumPy ndarray (2D)
  dplyr / tidyr        ←→   Pandas (groupby, merge, pivot)
  ggplot2              ←→   Matplotlib / Seaborn
```

Python에서 Pandas는 R의 data.frame + dplyr + tidyr을 하나로 합쳐놓은 위치에 있다.

### NumPy와 Pandas의 관계

NumPy는 **저수준 수치 계산 엔진**이고, Pandas는 그 위에 올라탄 **고수준 데이터 분석 도구**다.

```
Pandas DataFrame
   └─ 내부적으로 NumPy ndarray로 저장
         └─ 연속 메모리 블록 (고속 연산 가능)
```

|구분|NumPy|Pandas|
|---|---|---|
|핵심 자료구조|ndarray|Series, DataFrame|
|데이터 타입|단일 타입|열마다 다른 타입 가능|
|인덱스|정수 위치(0, 1, 2...)|레이블 인덱스 (날짜, 문자열 등)|
|결측값 처리|기본 지원 없음|`NaN` 기본 지원|
|주요 용도|행렬 연산, 수치 계산|표 형태 데이터 분석|
|SQL 스타일|없음|groupby, merge, pivot|

Pandas 연산의 실제 계산은 NumPy가 처리하기 때문에, Pandas를 쓰면서도 NumPy 수준의 속도를 낼 수 있다. `df.values`나 `df.to_numpy()`로 DataFrame을 ndarray로 꺼낼 수 있다.

```python
import numpy as np
import pandas as pd

arr = np.array([[1, 2, 3], [4, 5, 6]])
df  = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# DataFrame → ndarray
df.to_numpy()   # array([[1, 2, 3], [4, 5, 6]])

# ndarray → DataFrame
pd.DataFrame(arr, columns=['A', 'B', 'C'])

# NumPy 함수는 Pandas에 바로 적용 가능
np.sqrt(df)
np.log(df['A'])
```

### Pandas와 R의 관계

Pandas는 R의 데이터 분석 패러다임을 Python으로 옮겨온 도구다.  
Hadley Wickham의 tidyverse(dplyr, tidyr)와 목적이 거의 동일하며, 문법도 대응 관계가 뚜렷하다.

|기능|R (base + tidyverse)|Pandas|
|---|---|---|
|표 자료구조|`data.frame`, `tibble`|`DataFrame`|
|열 선택|`df %>% select(col)`|`df['col']`|
|행 필터링|`df %>% filter(x > 0)`|`df[df['x'] > 0]`|
|정렬|`df %>% arrange(x)`|`df.sort_values('x')`|
|파생 변수|`df %>% mutate(z = x + y)`|`df['z'] = df['x'] + df['y']`|
|그룹 집계|`df %>% group_by(g) %>% summarise(m = mean(x))`|`df.groupby('g')['x'].mean()`|
|긴 형태 변환|`pivot_longer()`|`melt()`|
|넓은 형태 변환|`pivot_wider()`|`pivot_table()`|
|두 표 결합|`left_join()`|`merge(..., how='left')`|
|결측값 제거|`na.omit()`|`dropna()`|
|결측값 채우기|`replace_na()`|`fillna()`|
|범주형|`factor()`|`pd.Categorical()`|
|문자열 처리|`stringr::str_*`|`df['col'].str.*`|

### 핵심 차이점

**인덱스 시작**

- R: 1부터 시작
- Python(NumPy/Pandas): 0부터 시작

```python
# Python
df.iloc[0]   # 첫 번째 행 (0-based)

# R
df[1, ]      # 첫 번째 행 (1-based)
```

**파이프 문법**

R의 `%>%`와 Python의 메서드 체인은 같은 역할을 한다.

```r
# R
df %>%
  filter(score >= 80) %>%
  group_by(grade) %>%
  summarise(avg = mean(score))
```

```python
# Python (메서드 체인)
(df
  .query('score >= 80')
  .groupby('grade')['score']
  .mean()
)
```

**벡터화 연산 방식**

R은 언어 자체가 벡터 중심이라 기본 연산이 곧 벡터화다.  
Python은 기본 리스트가 벡터화를 지원하지 않아, NumPy ndarray나 Pandas Series를 써야 벡터화 연산이 가능하다.

```r
# R — 기본 벡터에 바로 적용
x <- c(1, 2, 3, 4)
x * 2   # [2, 4, 6, 8]
```

```python
# Python — 리스트는 벡터화 불가
x = [1, 2, 3, 4]
x * 2   # [1, 2, 3, 4, 1, 2, 3, 4]  ← 리스트 반복

# NumPy ndarray나 Pandas Series를 써야 벡터화 가능
x = np.array([1, 2, 3, 4])
x * 2   # [2, 4, 6, 8]
```

### 언제 무엇을 쓸까

|상황|추천 도구|
|---|---|
|통계 분석, 논문, 학술 연구|R|
|행렬 연산, 선형대수, 수치 계산|NumPy|
|표 데이터 전처리 및 분석|Pandas|
|머신러닝 모델 학습|NumPy + scikit-learn|
|대용량 데이터 처리|Pandas + Parquet / PySpark|
|시각화|R(ggplot2) / Python(Matplotlib, Seaborn)|

세 도구는 경쟁 관계가 아니라 **서로 다른 층위에서 보완하는 관계**다.  
실무에서는 NumPy로 수치 연산 기반을 만들고, Pandas로 데이터를 전처리하고, Matplotlib으로 시각화하는 흐름으로 함께 사용한다.

---

## 3. 자료구조: Series와 DataFrame

### Series

1차원 레이블 배열이다. NumPy 배열에 인덱스(레이블)를 붙인 구조로 이해하면 된다.

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# a    10
# b    20
# c    30
```

- 하나의 열(column)이 곧 하나의 Series
- dtype이 하나로 고정됨 (NumPy 배열과 동일)
- 인덱스로 값에 접근 가능: `s['a']` → 10

### DataFrame

여러 Series를 열로 묶은 2차원 표 형태 구조다.  
행에는 **Row Index**, 열에는 **Column Index**가 붙는다.

```
         Column Index
          A    B    C
Row  0   1.0  2.0  3.0
Index 1   4.0  5.0  6.0
      2   7.0  8.0  9.0

각 열(A, B, C)은 독립적인 Series
내부적으로는 NumPy ndarray로 저장됨
```

```python
df = pd.DataFrame({
    'name':  ['Kim', 'Lee', 'Park'],
    'score': [85, 92, 78],
    'grade': ['B', 'A', 'C']
})
```

- 열마다 dtype이 달라도 됨 (문자열 열 + 숫자 열 + 논리형 열 공존 가능)
- R의 data.frame과 동일한 개념

---

## 4. DataFrame과 SQL의 대응

SQL을 알면 Pandas를 더 직관적으로 이해할 수 있다.

|SQL|Pandas|설명|
|---|---|---|
|`SELECT`|`df[컬럼명]`|열 선택|
|`WHERE`|Boolean Indexing|조건에 맞는 행 필터링|
|`GROUP BY`|`groupby()`|그룹별 집계|
|`ORDER BY`|`sort_values()`|정렬|
|`JOIN`|`merge()`|두 DataFrame 결합|

```python
# SQL: SELECT name, score FROM df WHERE score >= 80 ORDER BY score DESC
df[['name', 'score']][df['score'] >= 80].sort_values('score', ascending=False)

# SQL: SELECT grade, AVG(score) FROM df GROUP BY grade
df.groupby('grade')['score'].mean()
```

---

## 5. 데이터 변환 (Data Transformation)

### 기본 전처리

```python
df.info()           # 열별 dtype, 결측값 수 확인
df.describe()       # 수치형 변수 기술통계
df.isnull().sum()   # 결측값 개수 확인

df.dropna()                         # 결측값 있는 행 제거
df.fillna(df.mean(numeric_only=True))  # 평균으로 채우기
df.drop_duplicates()                # 중복 행 제거
df.rename(columns={'old': 'new'})   # 열 이름 변경
df.astype({'score': float})         # dtype 변환
```

### 통계 함수와 상관 함수

```python
df['score'].mean()    # 평균
df['score'].std()     # 표준편차
df['score'].median()  # 중위수
df['score'].value_counts()  # 빈도 카운트

df.corr()             # 수치형 변수 간 상관 행렬
df.cov()              # 공분산 행렬
```

### 벡터화된 문자열 함수

Series에 `.str` 접근자를 붙이면 문자열 연산을 벡터화해서 적용할 수 있다.

```python
df['name'].str.upper()          # 대문자 변환
df['name'].str.contains('Kim')  # 특정 문자 포함 여부
df['name'].str.replace('a', 'A') # 문자 치환
df['name'].str.len()            # 문자 길이
df['name'].str.split('_')       # 구분자로 분리
```

### 데이터 요약 기능

```python
# 그룹별 복수 집계
df.groupby('grade').agg({
    'score': ['mean', 'std', 'count'],
    'name':  'count'
})

# 피벗 테이블 (엑셀의 피벗과 동일)
pd.pivot_table(df,
    values='score',
    index='grade',
    columns='gender',
    aggfunc='mean'
)
```

슬라이드의 크로스탭 예시 (`건조 우기 All` 표)처럼, 두 범주형 변수를 행/열로 배치하고 집계값을 채우는 방식이다.

### 시계열 전처리

```python
df['date'] = pd.to_datetime(df['date'])   # 날짜 타입으로 변환
df.set_index('date', inplace=True)        # 날짜를 인덱스로 설정

df.resample('M').mean()    # 월별 평균으로 리샘플링
df.resample('W').sum()     # 주별 합계

df['score'].rolling(window=7).mean()   # 7일 이동평균
df['score'].shift(1)                   # 1기간 전 값
df['score'].diff()                     # 전기 대비 변화량
```

### 데이터에 함수 적용

|함수|적용 대상|설명|
|---|---|---|
|`map()`|Series|원소 하나하나에 함수 또는 딕셔너리 적용|
|`apply()`|Series / DataFrame|행 또는 열 단위로 함수 적용|
|`applymap()`|DataFrame|모든 원소 하나하나에 함수 적용 (현재는 `map()` 권장)|

```python
# map: 값을 딕셔너리로 매핑
df['grade'].map({'A': 4.0, 'B': 3.0, 'C': 2.0})

# apply: 열 단위로 함수 적용
df[['score']].apply(lambda x: (x - x.mean()) / x.std())

# apply: 행 단위로 함수 적용 (axis=1)
df.apply(lambda row: row['score'] * 2 if row['grade'] == 'A' else row['score'], axis=1)
```

### 데이터베이스 연동 (query)

```python
import sqlalchemy

engine = sqlalchemy.create_engine('mysql+pymysql://user:pw@host/db')

# DB에서 읽기
df = pd.read_sql("SELECT * FROM table WHERE score > 80", engine)

# DB에 쓰기
df.to_sql('result_table', engine, if_exists='replace', index=False)
```

---

## 6. 직렬화와 역직렬화 (Serialization / De-Serialization)

직렬화는 **메모리에 있는 Python 객체를 저장하거나 전송할 수 있는 바이트 스트림으로 변환**하는 과정이다.  
역직렬화는 그 반대, 바이트 스트림을 다시 Python 객체로 복원하는 과정이다.

```
Object (메모리)
   ↓ 직렬화 (Serialization)
ByteStream
   ├─ File로 저장
   ├─ Memory에 보관
   └─ Database에 저장
   ↓ 역직렬화 (De-Serialization)
Object (메모리)
```

### 저장 포맷

|포맷|저장|불러오기|특징|
|---|---|---|---|
|CSV|`to_csv()`|`read_csv()`|텍스트 기반. 범용적. 타입 정보 손실|
|JSON|`to_json()`|`read_json()`|웹 API 연동. 중첩 구조 표현 가능|
|Pickle|`to_pickle()`|`read_pickle()`|Python 전용 이진 포맷. 타입 완벽 보존|
|Parquet|`to_parquet()`|`read_parquet()`|열 기반 압축 포맷. 대용량 데이터에 효율적|

```python
# 저장
df.to_csv('data.csv', index=False)
df.to_json('data.json', force_ascii=False)
df.to_pickle('data.pkl')
df.to_parquet('data.parquet')

# 불러오기
df = pd.read_csv('data.csv')
df = pd.read_json('data.json')
df = pd.read_pickle('data.pkl')
df = pd.read_parquet('data.parquet')
```

### 포맷 선택 기준

- **다른 도구나 언어와 공유**: CSV, JSON
- **Python 내부에서만 사용**: Pickle (타입 정보 완벽 보존)
- **대용량 데이터, 반복 읽기**: Parquet (압축률 높고 읽기 빠름)
- **엑셀과 연동**: `to_excel()` / `read_excel()`

---

---

## 7. 시계열 데이터 (Time Series)

시간 순서에 따라 기록된 데이터다. 주가, 기온, 트래픽, 센서 측정값처럼 **시간이 인덱스**가 되는 구조다.  
일반 데이터와 달리 **순서**, **주기성**, **추세**가 중요하다.

---

### 시계열의 구성 요소

시계열 데이터는 보통 다음 네 가지 요소가 합쳐진 형태로 이루어진다.

|구성 요소|설명|예시|
|---|---|---|
|**추세 (Trend)**|장기적으로 증가 또는 감소하는 방향|연간 매출 꾸준히 증가|
|**계절성 (Seasonality)**|일정 주기로 반복되는 패턴|여름마다 에어컨 판매 급증|
|**주기 (Cycle)**|계절성보다 긴 불규칙한 파동|경기 사이클|
|**잔차 (Residual)**|위 세 요소로 설명 안 되는 무작위 노이즈|예측 오차|

```
원본 시계열 = 추세 + 계절성 + 주기 + 잔차
```

---

### DatetimeIndex — 시간 인덱스

Pandas에서 시계열을 다루려면 인덱스를 **DatetimeIndex**로 설정해야 한다.

```python
import pandas as pd
import numpy as np

# 문자열을 날짜 타입으로 변환
df['date'] = pd.to_datetime(df['date'])

# 날짜를 인덱스로 설정
df = df.set_index('date')

# 날짜 범위 직접 생성
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')  # 일별
dates = pd.date_range(start='2024-01-01', periods=12, freq='ME')        # 월별 12개
```

**주요 freq 옵션**

|코드|의미|
|---|---|
|`'D'`|일별|
|`'W'`|주별|
|`'ME'`|월말|
|`'QE'`|분기말|
|`'YE'`|연말|
|`'h'`|시간별|
|`'min'`|분별|

---

### 시계열 인덱싱 및 슬라이싱

DatetimeIndex가 설정되면 날짜 문자열로 바로 접근할 수 있다.

```python
df['2024']              # 2024년 전체
df['2024-06']           # 2024년 6월
df['2024-01':'2024-06'] # 2024년 1~6월
df.loc['2024-03-01']    # 특정 날짜
```

---

### 리샘플링 (Resampling)

시간 단위를 변경하는 작업이다. 일별 데이터를 월별로 묶거나, 시간별 데이터를 일별로 묶는 식이다.

```python
# 다운샘플링 (세밀 → 거칠게): 일별 → 월별
df.resample('ME').mean()   # 월별 평균
df.resample('ME').sum()    # 월별 합계
df.resample('ME').agg({'price': 'mean', 'volume': 'sum'})  # 열마다 다른 집계

# 업샘플링 (거칠게 → 세밀하게): 월별 → 일별
df.resample('D').ffill()   # 앞 값으로 채우기 (forward fill)
df.resample('D').bfill()   # 뒤 값으로 채우기 (backward fill)
df.resample('D').interpolate('spline', order=3)  # 스플라인 보간
```

다운샘플링은 데이터를 집계해서 줄이는 것이고, 업샘플링은 빈 구간을 채워서 늘리는 것이다.

---

### 이동 통계 (Rolling / Expanding)

**이동 통계(Rolling)**: 고정된 크기의 창(window)을 슬라이딩하며 통계를 계산한다.  
노이즈가 많은 데이터의 전체적인 흐름(추세)을 파악하는 데 유용하다.

```python
# 7일 이동 평균
df['rolling_mean'] = df['value'].rolling(window=7).mean()

# 30일 이동 표준편차 (변동성 파악)
df['rolling_std'] = df['value'].rolling(window=30).std()

# 중심 이동 평균 (창의 중앙이 기준)
df['value'].rolling(window=7, center=True).mean()
```

```
원본:   [1, 3, 2, 5, 4, 6, 3]
window=3 이동 평균:
        [NaN, NaN, 2.0, 3.3, 3.7, 5.0, 4.3]
         ↑    ↑    ↑
         처음 2개는 창이 차지 않아 NaN
```

**누적 통계(Expanding)**: 시작점부터 현재까지 전체를 누적해서 계산한다.

```python
df['value'].expanding().mean()   # 누적 평균
df['value'].expanding().max()    # 누적 최댓값
```

---

### 시차와 변화량 (Shift / Diff)

```python
# 전기 대비 이동 (lag)
df['lag1'] = df['value'].shift(1)    # 1기간 전 값
df['lag7'] = df['value'].shift(7)    # 7기간 전 값 (주간 비교)

# 전기 대비 변화량
df['diff1'] = df['value'].diff(1)    # 1기간 전과의 차이
df['diff7'] = df['value'].diff(7)    # 7기간 전과의 차이 (주간 변화)

# 전기 대비 증감률 (%)
df['pct'] = df['value'].pct_change() * 100
```

---

### 날짜 속성 추출

DatetimeIndex 또는 datetime 열에서 날짜 구성 요소를 추출할 수 있다.

```python
df.index.year       # 연도
df.index.month      # 월
df.index.day        # 일
df.index.weekday    # 요일 (0=월요일, 6=일요일)
df.index.quarter    # 분기
df.index.dayofyear  # 연중 몇 번째 날

# 열에 datetime이 있는 경우
df['date'].dt.year
df['date'].dt.month
df['date'].dt.day_name()   # 'Monday', 'Tuesday' ...
```

---

### 시계열 분해 (Decomposition)

추세, 계절성, 잔차를 분리해서 각각 분석하는 방법이다.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# 가법 분해: 원본 = 추세 + 계절성 + 잔차
result = seasonal_decompose(df['value'], model='additive', period=12)

# 승법 분해: 원본 = 추세 × 계절성 × 잔차 (값이 클수록 변동폭도 큰 경우)
result = seasonal_decompose(df['value'], model='multiplicative', period=12)

result.plot()   # 4개 그래프로 시각화 (원본, 추세, 계절성, 잔차)
```

가법 분해는 계절 변동폭이 일정할 때, 승법 분해는 계절 변동폭이 추세에 비례해서 커질 때 사용한다.

---

### 정상성 (Stationarity)

시계열 분석 모델 대부분은 데이터가 **정상성(Stationarity)** 을 만족해야 한다.  
정상성이란 시간이 지나도 **평균, 분산, 자기공분산이 일정**한 성질이다.

```
정상 시계열: 평균과 분산이 시간에 따라 변하지 않음
비정상 시계열: 추세나 계절성이 있어 평균/분산이 변함
```

**ADF 검정 (Augmented Dickey-Fuller Test)** 으로 정상성을 확인한다.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['value'])
print(f'ADF 통계량: {result[0]:.4f}')
print(f'p-value:   {result[1]:.4f}')

# p < 0.05 → 정상 시계열 (귀무가설: 단위근 존재 = 비정상)
# p ≥ 0.05 → 비정상 시계열 → 차분(differencing) 필요
```

비정상 시계열은 **차분(Differencing)** 으로 정상화한다.

```python
df['diff'] = df['value'].diff(1)      # 1차 차분
df['diff2'] = df['value'].diff(1).diff(1)  # 2차 차분 (1차로 안 될 때)
```

---

### 자기상관 (Autocorrelation)

시계열은 **현재 값이 과거 값과 얼마나 상관이 있는지**를 분석하는 것이 중요하다.

- **ACF (Autocorrelation Function)**: 현재 값과 k기간 전 값의 상관관계
- **PACF (Partial ACF)**: 중간 시차의 영향을 제거한 순수 자기상관

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df['value'], lags=40, ax=ax1)
plot_pacf(df['value'], lags=40, ax=ax2)
plt.tight_layout()
```

ACF/PACF 그래프는 이후 ARIMA 모델의 파라미터(p, d, q)를 결정하는 데 사용된다.

---

### 시계열 예측 모델

|모델|특징|적합한 경우|
|---|---|---|
|**이동 평균 (MA)**|최근 값의 평균으로 예측|단순 추세, 빠른 예측|
|**지수 평활 (ETS)**|최근 값에 더 큰 가중치|추세 + 계절성이 있는 경우|
|**ARIMA**|자기회귀 + 차분 + 이동평균|정상화 후 선형 패턴|
|**SARIMA**|ARIMA + 계절성|명확한 계절 패턴|
|**Prophet**|Facebook 개발. 휴일 효과 포함|사업 데이터, 결측값 허용|

```python
# ARIMA 예시
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df['value'], order=(1, 1, 1))  # (p, d, q)
result = model.fit()
forecast = result.forecast(steps=30)  # 30기간 예측

# Prophet 예시
from prophet import Prophet

m = Prophet()
m.fit(df.rename(columns={'date': 'ds', 'value': 'y'}))
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
m.plot(forecast)
```

---

### R과 Python의 시계열 도구 비교

|기능|R|Python|
|---|---|---|
|시계열 자료구조|`ts`, `xts`, `zoo`|`pd.DatetimeIndex`|
|분해|`decompose()`, `stl()`|`seasonal_decompose()`|
|정상성 검정|`adf.test()` (tseries)|`adfuller()` (statsmodels)|
|ACF/PACF|`acf()`, `pacf()`|`plot_acf()`, `plot_pacf()`|
|ARIMA|`auto.arima()` (forecast)|`ARIMA()` (statsmodels)|
|예측 시각화|`autoplot()`|`matplotlib`|

---

## 8. 전체 구조 정리

```
Pandas
   ├─ 자료구조
   │   ├─ Series      : 1차원, 인덱스 있는 배열
   │   └─ DataFrame   : 2차원, 행/열 인덱스 있는 표
   │
   ├─ 데이터 변환
   │   ├─ 전처리      : dropna, fillna, rename, astype
   │   ├─ 통계        : mean, std, corr, cov
   │   ├─ 문자열      : str.upper, str.contains, str.split
   │   ├─ 그룹 집계   : groupby, pivot_table
   │   ├─ 함수 적용   : map, apply, applymap
   │   └─ 시계열      : (아래 참고)
   │
   ├─ 시계열
   │   ├─ DatetimeIndex : 날짜 인덱스 설정
   │   ├─ resample      : 시간 단위 변환 (다운/업샘플링)
   │   ├─ rolling       : 이동 통계 (이동평균 등)
   │   ├─ expanding     : 누적 통계
   │   ├─ shift / diff  : 시차, 변화량
   │   ├─ decompose     : 추세/계절성/잔차 분해
   │   ├─ adfuller      : 정상성 검정
   │   └─ ACF / PACF    : 자기상관 분석
   │
   ├─ SQL 스타일 연산
   │   SELECT → df[] / WHERE → boolean indexing
   │   GROUP BY → groupby() / ORDER BY → sort_values()
   │   JOIN → merge()
   │
   └─ 저장 포맷
       CSV / JSON / Pickle / Parquet
```