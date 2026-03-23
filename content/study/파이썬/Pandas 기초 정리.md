
# Pandas 기초 

## Pandas란?
데이터를 다루기 위한 Python 라이브러리 
엑셀 형식의 데이터를 다루는데 유용함 
- Series : 1차원 데이터 
- DataFrame : 2차원 엑셀 형식의 데이터 

---

## Python에서 Pandas 사용하기

### 설치 
```
pip install pandas
```

### 불러오기
```python

# as 를 사용하면 해당 라이브러리를 별칭을 지정할 수 있다
# pandas는 관습적으로 pd로 표기 
import pandas as pd 
```
---
## CSV 파일 불러오기 
```python
df = pd.read_csv("data.csv")

# 자주 사용하는 기본 메서드 
df.head()     # 위에서부터 5줄 확인  
df.tail()     # 아래서부터 5줄 확인
df.shape      # (행, 열)의 개수 확인
df.columns    # 컬럼들의 이름 확인
df.info()     # 데이터 타입, 결측치 확인
df.describe() # 데이터 통계 요약
```
---
## 열(Column) 선택
```python
# 세로방향 선택 
# 한 개의 열 선택 
df["age"]

# 여러개의 열 선택
df[["name", "age"]]
```
---
## 행(Row) 필터링 

### 조건 필터링
```python
# 60살 이상인 데이터만 필터링 
df[df["age"] >= 60] 

# 나이가 60이상이고 성별이 남성인 데이터만 필터링 
df[(df["age"] >= 60) & (df["gender"] == "M")]
```
주의 사항 : 
- and, or 이 아니라 &, | 을 사용 
- 조건마다 괄호로 감싸야 함 
---
## 결측치 확인 
```python
df.isnull()
df.isnull().sum()

# 특정 column의 결측치만 확인은 조건 필터링과 같은 방식 
df[df["age"].isull()]
```

## 결측치 처리 

### 결측치 제거 
```python
df.dropna()
```
### 특정 값으로 결측치 채우기 
```python
# 결측치를 전부 0으로 변경
df.fillna(0)
```
### 평균값으로 채우기
```python
# "age" 열에 있는 결측치를 "age" 열의 평균치로 채움
df["age"] = df["age"].fillna(df["age"].mean())

# 모든 column의 결측치를 각 열의 평균으로 채움 
df = df.fillna(df.mean())
```
### 앞, 뒤의 값으로 채우기
```python
# 결측치를 한 칸 앞의 데이터로 대체
df = df.ffill(axis=0)

# 결측치를 한 칸 뒤의 데이터로 대체
df = df.dfill(axis=0)
```
---
## 정렬 
```python
# age column의 값을 기준으로 오름차순으로 데이터를 정렬
df.sort_values("age") 
# age column의 값을 기준으로 내림차순으로 데이터를 정렬
df.sort_values("age", ascending=false)
```
---
## 그룹화와 집계

```python
# 그룹화 
df.groupby("gender") # gender를 기준으로 그룹화

# 그룹별 개수 세기
# gender를 기준으로 그룹화하고 그룹 내에 age의 개수를 셈 
df.groupby("gender")["age"].count()

# 그룹별 평균 구하기
# gender를 기준으로 그룹화하고 그룹 내에 age의 평균을 구함
df.groupby("gender")["age"].mean()

# 여러 통계값 함께 보기 
# agg 함수를 사용해서 여러 통계랑을 동시에 계산
# agg 를 사용하면 여러 집계 함수를 동시에 계산이 가능하고 사용자 정의 함수도 이용가능 
df.groupby("gender")["age"].agg(["count", "mean", "max", "min"])
```
---
## 새로운 컬럼 만들기 
```python

```
---
## 문자열 처리 
---
## 날짜 처리 
---
## 파일 저장