---
title : Pandas 기초 정리 
tags :
	- python
---

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